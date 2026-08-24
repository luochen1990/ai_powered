'''
职责: @ai_powered 装饰器 retry 参数的离线单元测试.

通过 monkeypatch `ai_powered.decorators.LlmConnection` 注入脚本化的 FakeLlmConnection,
不触网、结果确定, 覆盖重试语义的全部关键分支 (与 test/examples/ 的真实 LLM 测试互补).

重试契约 (见 doc/en/DecoratorParams.md 决策 D2):
  - retry=N 表示额外重试 N 次 (总尝试 N+1 次), retry=0 与旧行为等价
  - 可重试失败: openai.APIError (网络/API 层), LlmGatewayError (网关 200+body.error
    透传上游瞬时故障, 见 test_gateway_guard.py), msgspec.DecodeError (含 ValidationError,
    JSON 语法/结构校验失败), AssertionError (适配器中"响应不含期望结构")
  - 不可重试异常 (如裸 RuntimeError / ValueError) 直接抛出
  - 耗尽后原样抛出最后一个异常
'''
import httpx
import openai
import pytest
from easy_sync import Waitable, sync_compatible
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from typing import Optional, cast

import ai_powered.decorators as decorators
from ai_powered import ai_powered
from ai_powered.llm.connection import LlmGatewayError


def make_completion(content: Optional[str]) -> ChatCompletion:
    return ChatCompletion(
        id = "fake-completion", created = 0, model = "fake-model", object = "chat.completion",
        choices = [Choice(finish_reason = "stop", index = 0, message = ChatCompletionMessage(role = "assistant", content = content))],
    )


class FakeLlmConnection:
    ''' 脚本化连接: 依次消费脚本元素, Exception 实例则抛出, str 则作为 content 返回; 脚本耗尽后重复最后一个元素 '''

    def __init__(self, script: list[Exception | str]):
        self.script = list(script)
        self.call_count = 0

    @sync_compatible
    async def chat_completions(self, **kwargs) -> ChatCompletion:
        ''' 与 LlmConnection.chat_completions 同为 sync_compatible, 兼容 easy_sync 的 sync 转写 '''
        step = self.script[min(self.call_count, len(self.script) - 1)]
        self.call_count += 1
        if isinstance(step, Exception):
            raise step
        return make_completion(step)


@pytest.fixture
def fake_llm(monkeypatch):
    ''' 安装脚本化连接; 同时 pin 住装饰期读取的模型配置 (走 StructuredOutput 路径, 裸 JSON 响应), 使测试与宿主环境变量解耦 '''
    def setup(script: list[Exception | str]) -> FakeLlmConnection:
        fake = FakeLlmConnection(script)
        monkeypatch.setattr(decorators, "LlmConnection", lambda **kwargs: fake)
        monkeypatch.setattr(decorators, "OPENAI_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setattr(decorators, "OPENAI_MODEL_NAME", None)
        monkeypatch.setattr(decorators, "OPENAI_MODEL_FEATURES", None)
        return fake
    return setup


def conn_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request = httpx.Request("POST", "http://fake.test"))


def test_retry_recovers_from_transient_network_error(fake_llm):
    fake = fake_llm([conn_error(), '{"result": 42}'])

    @ai_powered(retry = 2)
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    assert answer(6) == 42
    assert fake.call_count == 2


def test_retry_exhausted_raises_last_error(fake_llm):
    fake = fake_llm([conn_error()])

    @ai_powered(retry = 2)
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    with pytest.raises(openai.APIError):
        answer(6)
    assert fake.call_count == 3


def test_exhausted_raises_the_last_exception_not_the_first(fake_llm):
    ''' 耗尽时抛出的必须是最后一个异常 (与 "额外 N 次重试" 语义配套的顺序性质); 全部脚本异常须可重试, 否则会提前穿透 '''
    fake = fake_llm([AssertionError("first failure"), AssertionError("last failure")])

    @ai_powered(retry = 1)
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    with pytest.raises(AssertionError, match = "last failure"):
        answer(6)
    assert fake.call_count == 2


def test_retry_recovers_from_json_syntax_error(fake_llm):
    ''' 非法 JSON (DecodeError 而非 ValidationError 子类路径) 同样可重试 '''
    fake = fake_llm(['not json at all', '{"result": 42}'])

    @ai_powered(retry = 1)
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    assert answer(6) == 42
    assert fake.call_count == 2


def test_retry_recovers_from_gateway_error(fake_llm):
    ''' LLM 网关 200+body.error 透传的上游瞬时故障 (OpenRouter 503 overloaded 等) 属于可重试失败 '''
    fake = fake_llm([LlmGatewayError("LLM gateway returned error in 200 response: 503 overloaded"), '{"result": 42}'])

    @ai_powered(retry = 1)
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    assert answer(6) == 42
    assert fake.call_count == 2


def test_plain_runtime_error_not_retried(fake_llm):
    ''' 裸 RuntimeError 不重试 —— 只有专用的 LlmGatewayError 纳入可重试集合, 避免误重试编程错误 (如嵌套事件循环误用) '''
    fake = fake_llm([RuntimeError("boom"), '{"result": 42}'])

    @ai_powered(retry = 3)
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    with pytest.raises(RuntimeError, match = "boom"):
        answer(6)
    assert fake.call_count == 1


def test_retry_recovers_from_schema_validation_error(fake_llm):
    ''' LLM 返回了合法 JSON 但类型不符 (result 应为 int 却给了 str), 属于可重试的结构校验失败 '''
    fake = fake_llm(['{"result": "not-an-int"}', '{"result": 42}'])

    @ai_powered(retry = 1)
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    assert answer(6) == 42
    assert fake.call_count == 2


def test_non_retryable_error_not_retried(fake_llm):
    fake = fake_llm([ValueError("boom")])

    @ai_powered(retry = 3)
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    with pytest.raises(ValueError):
        answer(6)
    assert fake.call_count == 1


def test_bare_form_default_no_retry(fake_llm):
    ''' 裸形式 @ai_powered 不重试, 保持旧行为 '''
    fake = fake_llm([conn_error(), '{"result": 42}'])

    @ai_powered
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    with pytest.raises(openai.APIError):
        answer(6)
    assert fake.call_count == 1


def test_negative_retry_rejected_at_decoration_time(fake_llm):
    ''' retry 为负数是编程错误, 应在装饰期 (import 时) fail-fast; 安装 fake 以免依赖"校验先于 connection 构造"的代码顺序不变式 '''
    fake_llm(['{"result": 42}'])
    with pytest.raises(ValueError):
        @ai_powered(retry = -1)
        def answer(x: int) -> int:
            ''' produce the answer '''
            ...


@pytest.mark.parametrize("bad_retry", [1.5, True, "2"])
def test_non_integer_retry_rejected_at_decoration_time(fake_llm, bad_retry):
    ''' 非整数 retry (含 bool) 是编程错误, 应在装饰期 fail-fast 而非调用期 TypeError '''
    fake_llm(['{"result": 42}'])
    with pytest.raises(ValueError):
        @ai_powered(retry = bad_retry)
        def answer(x: int) -> int:
            ''' produce the answer '''
            ...


def test_empty_parentheses_form_works(fake_llm):
    ''' @ai_powered() 空参形式等价于裸形式 (fn is None 分支的边界) '''
    fake = fake_llm(['{"result": 42}'])

    @ai_powered()
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    assert answer(6) == 42
    assert fake.call_count == 1


def test_explicit_zero_retry_equivalent_to_bare(fake_llm):
    fake = fake_llm([conn_error(), '{"result": 42}'])

    @ai_powered(retry = 0)
    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    with pytest.raises(openai.APIError):
        answer(6)
    assert fake.call_count == 1


def test_mixed_form_function_and_retry(fake_llm):
    ''' 混合形式: 显式传函数 + retry (程序化装饰场景) '''
    fake = fake_llm([conn_error(), '{"result": 42}'])

    def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    powered = ai_powered(answer, retry = 1)
    assert powered(6) == 42
    assert fake.call_count == 2


@pytest.mark.asyncio
async def test_retry_with_async_function(fake_llm):
    fake = fake_llm([conn_error(), '{"result": 42}'])

    @ai_powered(retry = 1)
    async def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    assert await answer(6) == 42
    assert fake.call_count == 2


@pytest.mark.asyncio
async def test_bare_async_function_awaited(fake_llm):
    ''' 旧基线最常用路径: bare 形式 + async + await (回归守护) '''
    fake = fake_llm(['{"result": 42}'])

    @ai_powered
    async def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    assert await answer(6) == 42
    assert fake.call_count == 1


@pytest.mark.asyncio
async def test_retry_exhausted_with_async_function(fake_llm):
    fake = fake_llm([conn_error()])

    @ai_powered(retry = 1)
    async def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    with pytest.raises(openai.APIError):
        await answer(6)
    assert fake.call_count == 2


@pytest.mark.asyncio
async def test_async_non_retryable_error_not_retried(fake_llm):
    fake = fake_llm([ValueError("boom")])

    @ai_powered(retry = 3)
    async def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    with pytest.raises(ValueError):
        await answer(6)
    assert fake.call_count == 1


def test_async_function_callable_in_sync_context_via_wait(fake_llm):
    ''' 回归: README 承诺 async 装饰函数可在 sync 上下文用 .wait() 调用

    旧实现中 wrapper_fn_async 自身被 sync_compatible 转写, 转写体内闭包变量不可见,
    导致 add(1, 1).wait() 抛 NameError (存量 bug, 本批修复: wrapper 直接返回模块级
    sync_compatible 函数的 Waitable, 不再做自身转写)
    '''
    fake = fake_llm(['{"result": 42}'])

    @ai_powered
    async def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    # 类型层面 async 装饰函数是 Awaitable[int] (运行时实际是 easy_sync.Waitable), 此处 cast 表达 .wait() 用法
    assert cast(Waitable[int], answer(6)).wait() == 42
    assert fake.call_count == 1


def test_wait_with_retry_recovers_in_translated_body(fake_llm):
    ''' .wait() (sync 转写体) 路径下的重试恢复 —— 转写体是历史上出过 NameError bug 的路径, 需要测试守护 '''
    fake = fake_llm([conn_error(), '{"result": 42}'])

    @ai_powered(retry = 2)
    async def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    assert cast(Waitable[int], answer(6)).wait() == 42
    assert fake.call_count == 2


def test_wait_with_retry_exhausted_in_translated_body(fake_llm):
    ''' .wait() (sync 转写体) 路径下的重试耗尽 —— 转写体中的 bare raise 语义与协程体一致, 且抛出的应是最后一个异常 '''
    fake = fake_llm([AssertionError("first failure"), AssertionError("last failure")])

    @ai_powered(retry = 1)
    async def answer(x: int) -> int:
        ''' produce the answer '''
        ...

    with pytest.raises(AssertionError, match = "last failure"):
        cast(Waitable[int], answer(6)).wait()
    assert fake.call_count == 2
