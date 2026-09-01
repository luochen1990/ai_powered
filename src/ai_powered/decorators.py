from functools import partial, wraps
from typing import Any, Awaitable, Callable, Generic, overload
from easy_sync import sync_compatible
from typing_extensions import ParamSpec, TypeVar
import msgspec
import openai

from ai_powered.llm.connection import LlmGatewayError
from ai_powered.llm.definitions import FunctionSimulator, FunctionSpec, ModelFeature
from ai_powered.llm.adapter_selector import FunctionSimulatorSelector
from ai_powered.llm.resolver import resolve_model
from ai_powered.agent import AgentRunError, FunctionAgent
from ai_powered.constants import DEBUG
from ai_powered.colors import gray, green, red, yellow
import inspect

A = TypeVar("A")

#: retry 环需要捕获的瞬时失败: API/网络层错误, LLM 网关 200+body.error 透传的上游瞬时故障 (LlmGatewayError),
#: agent run 的采样噪声类故障 (AgentRunError: 多轮预算耗尽/调用未知工具), 响应解析与结构校验错误,
#: 以及适配器中"响应不含期望结构"的断言
#: NOTE: 依赖适配器中的 assert 语句, python -O 模式下 assert 被剥离, 该类故障将退化为不可重试异常 (见 doc/en/DecoratorParams.md D2)
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (openai.APIError, LlmGatewayError, AgentRunError, msgspec.DecodeError, AssertionError)


class Result(msgspec.Struct, Generic[A]):
    result: A


P = ParamSpec("P")
R = TypeVar("R")


class _AgentEngine:
    ''' 适配器: FunctionAgent.run → query_model 形状, 使 agent 与单次模拟器复用同一 retry/校验管线

    NOTE: 必须是模块级类且 agent 走显式参数 —— easy_sync 的 sync 转写会重编译函数体, 闭包变量在其中不可见
    '''

    def __init__(self, agent: FunctionAgent) -> None:
        self.agent = agent

    @sync_compatible
    async def query_model(self, arguments_json: str) -> str:
        return await self.agent.run(arguments_json)


@sync_compatible
async def _simulate_and_validate(function_name: str, engine: FunctionSimulator | _AgentEngine, result_type: type, arguments_json: str) -> Any:
    ''' 模拟函数执行并按 Result[R] 校验响应 (重试环的最小单元)

    NOTE: 必须是模块级函数且依赖走显式参数 —— easy_sync 的 sync 转写会重编译函数体, 闭包变量在其中不可见
    '''
    resp_str = await engine.query_model(arguments_json)

    if DEBUG:
        print(f"{resp_str =}")
        print(green(f"[fn {function_name}] response extracted."))

    returned_result = msgspec.json.decode(resp_str, type = result_type)

    if DEBUG:
        print(f"{returned_result =}")
        print(green(f"[fn {function_name}] response validated."))

    return returned_result.result


@sync_compatible
async def _call_with_retry(function_name: str, sig: inspect.Signature, simulator: FunctionSimulator | _AgentEngine, result_type: type, args: tuple[Any, ...], kwargs: dict[str, Any], retry: int) -> Any:
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
def ai_powered(fn: Callable[P, R], *, model: str | None = None, retry: int = 0, agent: type[FunctionAgent] | None = None) -> Callable[P, R]:
    ''' 混合形式: 显式传入函数与参数 (程序化装饰场景) '''
    ...


@overload
def ai_powered(*, model: str | None = None, retry: int = 0, agent: type[FunctionAgent] | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    ''' 参数形式: R 不加 Awaitable 特判 —— async 函数的类型本身就是 (...) -> Coroutine[..., R], 无约束 R 已覆盖 '''
    ...


def ai_powered(
    fn: Callable[P, Awaitable[R]] | Callable[P, R] | None = None,
    *,
    model: str | None = None,
    retry: int = 0,
    agent: type[FunctionAgent] | None = None,
) -> Any:
    ''' Provide an AI powered implementation of a function

    model: "provider/model" 前缀名走注册表 (如 "deepseek/deepseek-chat"); None/裸名走环境变量路径 (现状)
    agent: FunctionAgent 子类 (仅 class, 非 instance) —— 替换默认的单次 LLM 模拟器为自定义引擎 (如多轮推理);
           每个被装饰函数在装饰期构造一次, retry 重试的是整个 agent run
    '''

    if fn is None:  # used as `@ai_powered(...)`
        return partial(ai_powered, model = model, retry = retry, agent = agent)

    if not isinstance(retry, int) or isinstance(retry, bool) or retry < 0:
        raise ValueError(f"retry must be a non-negative integer, got {retry!r}")

    if agent is not None and (not isinstance(agent, type) or not issubclass(agent, FunctionAgent)):
        raise ValueError(f"agent must be a FunctionAgent subclass (class, not an instance), got {agent!r}")

    if agent is not None:
        #NOTE: sync 转写体经 .wait() 调用 agent.run —— 裸 async def 的协程没有 .wait, 纯 sync def 返回值也没有;
        #两者都只在首次调用时爆晦涩 AttributeError, 故装饰期验证 run 呈 sync_compatible 包装形态
        #(结构判别: 合法包装的 iscoroutinefunction 为 False, 解开 __wrapped__ 后为 True;
        # 与合法 wrapper 结构同构的手工包装不可区分 —— easy-sync wrapper 无标记属性所限, 误放后果为 sync 路径首次调用 AttributeError)
        unwrapped_run = getattr(agent.run, "__wrapped__", agent.run)
        if not (inspect.iscoroutinefunction(unwrapped_run) and not inspect.iscoroutinefunction(agent.run)):
            raise ValueError(f"{agent.__name__}.run must be a @sync_compatible async def (easy-sync constraint)")

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

    # NOTE: parameters_schema 有意保持 raw (msgspec 原始输出, 不经 deref) —— 它只进 prompt 文本 (base_system_prompt),
    # 不进结构化请求参数, 与 return_schema (raw 存储 + 消费点能力门控) 是有意的对称性差异 (issue 18 零漂移决策)
    parameters_schema = {param.name: msgspec.json.schema(param.annotation) for param in sig.parameters.values()}
    return_type = sig.return_annotation
    return_schema = msgspec.json.schema(return_type)  # raw schema: 展开或透传由消费点按 schema_refs 能力门控 (issue 18)
    result_type = Result[return_type]

    if DEBUG:
        for param_name, schema in parameters_schema.items():
            print(f"{param_name} (json schema): {schema}")
        print(f"return (json schema): {return_schema}")

    resolved = resolve_model(model)
    connection = resolved.connection
    model_config = resolved.model_config
    model_name = model_config.model_name
    model_features: set[ModelFeature] = model_config.supported_features
    model_options: dict[str, Any] = model_config.suggested_options

    spec = FunctionSpec(function_name, full_signature, docstring, parameters_schema, return_schema)

    if agent is not None:
        missing_features = agent.required_features - model_features
        if missing_features:
            missing_names = sorted(f.name for f in missing_features)
            raise ValueError(f"{agent.__name__} requires model features {missing_names} not supported by model {model_name!r}")
        fn_simulator: FunctionSimulator | _AgentEngine = _AgentEngine(agent(spec, model_config, connection))
    else:
        fn_simulator = FunctionSimulatorSelector(spec, connection, model_name, model_features, model_options)

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

    if inspect.iscoroutinefunction(fn):
        return wrapper_fn_async
    else:
        return wrapper_fn
