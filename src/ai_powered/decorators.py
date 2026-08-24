import asyncio
from functools import partial, wraps
from typing import Any, Awaitable, Callable, Generic, overload
from easy_sync import sync_compatible
from typing_extensions import ParamSpec, TypeVar
import json
import msgspec
import openai

from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import FunctionSimulator, ModelFeature
from ai_powered.llm.adapter_selector import FunctionSimulatorSelector
from ai_powered.llm.known_models import complete_model_config
from ai_powered.schema_deref import deref
from ai_powered.constants import DEBUG, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_FEATURES, OPENAI_MODEL_NAME, SYSTEM_PROMPT, SYSTEM_PROMPT_JSON_SYNTAX, SYSTEM_PROMPT_RETURN_SCHEMA
from ai_powered.colors import gray, green, red, yellow
import inspect

A = TypeVar("A")

#: retry 环需要捕获的瞬时失败: API/网络层错误, 响应解析与结构校验错误, 以及适配器中"响应不含期望结构"的断言
#: NOTE: 依赖适配器中的 assert 语句, python -O 模式下 assert 被剥离, 该类故障将退化为不可重试异常 (见 doc/en/DecoratorParams.md D2)
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (openai.APIError, msgspec.DecodeError, AssertionError)


class Result(msgspec.Struct, Generic[A]):
    result: A


P = ParamSpec("P")
R = TypeVar("R")


@sync_compatible
async def _simulate_and_validate(function_name: str, simulator: FunctionSimulator, result_type: type, arguments_json: str) -> Any:
    ''' 模拟函数执行并按 Result[R] 校验响应 (重试环的最小单元)

    NOTE: 必须是模块级函数且依赖走显式参数 —— easy_sync 的 sync 转写会重编译函数体, 闭包变量在其中不可见
    '''
    resp_str = await simulator.query_model(arguments_json)

    if DEBUG:
        print(f"{resp_str =}")
        print(green(f"[fn {function_name}] response extracted."))

    returned_result = msgspec.json.decode(resp_str, type = result_type)

    if DEBUG:
        print(f"{returned_result =}")
        print(green(f"[fn {function_name}] response validated."))

    return returned_result.result


@sync_compatible
async def _call_with_retry(function_name: str, sig: inspect.Signature, simulator: FunctionSimulator, result_type: type, args: tuple[Any, ...], kwargs: dict[str, Any], retry: int) -> Any:
    ''' 绑定参数 -> 模拟 -> 校验, 瞬时失败时最多额外重试 retry 次; 耗尽后原样抛出最后一个异常. sync/async 两种 wrapper 共用的主逻辑

    NOTE: 必须是模块级函数且依赖走显式参数 —— easy_sync 的 sync 转写会重编译函数体, 闭包变量在其中不可见
    '''
    real_arg = sig.bind(*args, **kwargs)
    real_arg_str = msgspec.json.encode(real_arg.arguments).decode('utf-8')

    if DEBUG:
        print(f"{real_arg_str =}")

    for attempt in range(retry + 1):
        try:
            return await _simulate_and_validate(function_name, simulator, result_type, real_arg_str)
        except RETRYABLE_EXCEPTIONS as e:
            if attempt == retry:  # 最后一次尝试, 失败则原样抛出
                if DEBUG:
                    print(red(f"[fn {function_name}] all {retry + 1} attempts exhausted, giving up with {e!r}"))
                raise
            if DEBUG:
                print(yellow(f"[fn {function_name}] attempt {attempt + 1}/{retry + 1} failed with {e!r}, retrying ..."))

    raise AssertionError("unreachable")  # 运行时防御: 循环必经 return/raise 退出; 防御负 retry 等契约外输入导致空循环静默返回 None


@overload
def ai_powered(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    ...


@overload
def ai_powered(fn: Callable[P, R]) -> Callable[P, R]:
    ...


@overload
def ai_powered(fn: Callable[P, R], *, retry: int = 0) -> Callable[P, R]:
    ''' 混合形式: 显式传入函数与参数 (程序化装饰场景) '''
    ...


@overload
def ai_powered(*, retry: int = 0) -> Callable[[Callable[P, R]], Callable[P, R]]:
    ''' 参数形式: R 不加 Awaitable 特判 —— async 函数的类型本身就是 (...) -> Coroutine[..., R], 无约束 R 已覆盖 '''
    ...


def ai_powered(
    fn: Callable[P, Awaitable[R]] | Callable[P, R] | None = None,
    *,
    retry: int = 0,
) -> Any:
    ''' Provide an AI powered implementation of a function '''

    if fn is None:  # used as `@ai_powered(...)`
        return partial(ai_powered, retry = retry)

    if not isinstance(retry, int) or isinstance(retry, bool) or retry < 0:
        raise ValueError(f"retry must be a non-negative integer, got {retry!r}")

    function_name = fn.__name__
    sig = inspect.signature(fn)
    # Construct full signature including function name
    full_signature = f"def {function_name}{sig}"
    docstring = inspect.getdoc(fn)

    if DEBUG:
        print(f"{sig =}")
        print(f"{docstring =}")

        for param in sig.parameters.values():
            print(f"{param.name}: {param.annotation}")

        print(f"{sig.return_annotation =}")

    parameters_schema = {param.name: msgspec.json.schema(param.annotation) for param in sig.parameters.values()}
    return_type = sig.return_annotation
    raw_return_schema = msgspec.json.schema(return_type)
    return_schema = deref(raw_return_schema)
    result_type = Result[return_type]

    if DEBUG:
        for param_name, schema in parameters_schema.items():
            print(f"{param_name} (json schema): {schema}")
        print(f"return (json schema): {return_schema}")

    connection = LlmConnection(api_key = OPENAI_API_KEY, base_url = OPENAI_BASE_URL)
    model_config = complete_model_config(OPENAI_BASE_URL, OPENAI_MODEL_NAME, OPENAI_MODEL_FEATURES)
    model_name = model_config.model_name
    model_features: set[ModelFeature] = model_config.supported_features
    model_options: dict[str, Any] = model_config.suggested_options

    sys_prompt = SYSTEM_PROMPT.format(
        signature = full_signature,
        docstring = docstring or "no doc, guess intention from function name",
        parameters_schema = json.dumps(parameters_schema),
    ) + (
        "" if ModelFeature.tools in model_features else SYSTEM_PROMPT_RETURN_SCHEMA.format(return_schema = json.dumps(return_schema),) +
        SYSTEM_PROMPT_JSON_SYNTAX
    )

    if DEBUG:
        print(f"{sys_prompt =}")
        print(f"{return_schema =}")

    fn_simulator = FunctionSimulatorSelector(
        function_name, full_signature, docstring, parameters_schema, return_schema, connection, model_name, model_features, model_options
    )

    if DEBUG:
        print("fn_simulator =", gray(f"{fn_simulator}"))
        print(green(f"[fn {function_name}] AI is powering up."))

    @wraps(fn)
    def wrapper_fn(*args: P.args, **kwargs: P.kwargs) -> R:
        # NOTE: .wait() 取结果 —— sync 语义
        return _call_with_retry(function_name, sig, fn_simulator, result_type, args, kwargs, retry).wait()

    @wraps(fn)
    def wrapper_fn_async(*args: P.args, **kwargs: P.kwargs) -> R:
        # NOTE: 直接返回 Waitable (其本身是 Awaitable): await 时走 _call_with_retry 的原生协程体, .wait() 时走其 sync 转写体.
        # 本 wrapper 自身不做 sync_compatible 转写, 从而避开"闭包变量在转写体中不可见"的陷阱 (sig/fn_simulator 等均为闭包变量)
        return _call_with_retry(function_name, sig, fn_simulator, result_type, args, kwargs, retry)  #type: ignore

    if asyncio.iscoroutinefunction(fn):
        return wrapper_fn_async
    else:
        return wrapper_fn
