'''
职责: FunctionAgent 协议 + agent= 参数的离线单元测试.

覆盖 doc/en/DecoratorParams.md D3 决策:
  - agent 仅接受 class (非 instance), 装饰期构造一次
  - agent 与 retry 正交: retry 重试的是整个 agent run (run 调用次数 == 总尝试次数)
  - agent 路径出口校验依然生效 (Result[R] 解码, 无效输出触发 retry / DecodeError)
  - agent 接收 (spec, model_config, connection) —— model 解析结果原样传入 agent 构造器
  - ReasoningAgent 参考实现: 多轮推理循环 / return_result 提交 / max_rounds 耗尽 / 未知工具拒绝

全程离线: FakeAgent 脚本化; ReasoningAgent 用脚本化 FakeLlmConnection 驱动.
'''
from typing import Optional, cast

import msgspec
import pytest
from easy_sync import Waitable, sync_compatible
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall, Function
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from ai_powered import FunctionAgent, ReasoningAgent, ai_powered
from ai_powered.agent import AgentRunError
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import FunctionSpec, ModelFeature
from ai_powered.llm.known_models import KnownModel
from ai_powered.tool_call import return_result_tool_param

#: ReasoningAgent 离线测试的模型配置 (声明 tools 能力, 满足其 required_features)
_REASONING_MODEL = KnownModel("fake-model", {ModelFeature.tools})


def make_content_completion(content: Optional[str]) -> ChatCompletion:
    return ChatCompletion(
        id = "fake-completion", created = 0, model = "fake-model", object = "chat.completion",
        choices = [Choice(finish_reason = "stop", index = 0, message = ChatCompletionMessage(role = "assistant", content = content))],
    )


def make_tool_call_completion(tool_name: str, arguments_json: str) -> ChatCompletion:
    ''' 模拟模型调用工具 (如 return_result) 提交结果 '''
    tc = ChatCompletionMessageFunctionToolCall(id = "call-1", function = Function(name = tool_name, arguments = arguments_json), type = "function")
    return ChatCompletion(
        id = "fake-completion", created = 0, model = "fake-model", object = "chat.completion",
        choices = [Choice(finish_reason = "tool_calls", index = 0, message = ChatCompletionMessage(role = "assistant", content = None, tool_calls = [tc]))],
    )


class ScriptedCompletionConnection:
    ''' 脚本化连接 (ReasoningAgent 离线驱动): 依次返回预构造的 ChatCompletion (支持 tool_call 形态),

    耗尽后重复末元素; spy 记录每次请求的 messages 供对话累积断言. 与 LlmConnection 同为 sync_compatible.
    '''

    def __init__(self, script: list[ChatCompletion]):
        self.script = list(script)
        self.call_count = 0
        self.seen_messages: list[list[ChatCompletionMessageParam]] = []

    @sync_compatible
    async def chat_completions(self, **kwargs) -> ChatCompletion:
        self.seen_messages.append(list(kwargs.get("messages", [])))  # 拷贝快照 —— 调用方可能继续 append 同一 list
        step = self.script[min(self.call_count, len(self.script) - 1)]
        self.call_count += 1
        return step


class FakeAgent(FunctionAgent):
    ''' 脚本化 agent: 依次消费 class 级 script 元素 (Exception 抛出 / str 作为 run 返回值), 耗尽重复末元素;

    script 为 class 级共享 —— agent 实例在装饰期 (脚本设置之前) 构造, 测试通过 class 属性注入脚本.
    '''

    construct_count = 0
    script: list[Exception | str] = []
    last_instance: Optional["FakeAgent"] = None

    def __init__(self, spec: FunctionSpec, model_config, connection) -> None:
        super().__init__(spec, model_config, connection)
        FakeAgent.construct_count += 1
        FakeAgent.last_instance = self
        self.run_count = 0
        self.seen_args: list[str] = []

    @sync_compatible
    async def run(self, arguments_json: str) -> str:
        self.seen_args.append(arguments_json)
        self.run_count += 1
        step = FakeAgent.script[min(self.run_count - 1, len(FakeAgent.script) - 1)]
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture(autouse = True)
def reset_fake_agent():
    ''' 隔离 FakeAgent 的 class 级状态, 防止用例间污染 '''
    FakeAgent.construct_count = 0
    FakeAgent.script = []
    FakeAgent.last_instance = None
    yield
    FakeAgent.construct_count = 0
    FakeAgent.script = []
    FakeAgent.last_instance = None


def test_agent_receives_spec_model_config_connection(fake_llm):
    ''' agent 构造器收到 (spec, resolved.model_config, resolved.connection) —— model 解析结果原样传入 '''
    fake = fake_llm(['unused'])
    FakeAgent.script = ['{"result": 42}']

    @ai_powered(agent = FakeAgent)
    def fn(x: int) -> int:
        ''' doc '''
        ...

    agent = FakeAgent.last_instance
    assert agent is not None
    assert agent.spec.function_name == "fn"
    assert agent.spec.signature == "def fn(x: int) -> int"
    assert agent.spec.docstring is not None and agent.spec.docstring.strip() == "doc"
    assert agent.model_config.model_name == "fake-model"  # resolve_model 返回的 model_config 原样透传
    assert agent.connection is fake  # resolve_model 返回的 connection 原样透传
    assert fn(6) == 42
    assert agent.seen_args == ['{"x":6}']  # msgspec.json.encode 紧凑分隔 (无空格)


def test_agent_constructed_once_per_decoration(fake_llm):
    ''' 装饰期构造一次, 调用 N 次只复用同一实例 (D3: constructed once per decorated function) '''
    fake_llm(['{"result": 42}'])
    FakeAgent.script = ['{"result": 42}']

    @ai_powered(agent = FakeAgent)
    def fn(x: int) -> int:
        ''' doc '''
        ...

    assert fn(1) == 42
    assert fn(2) == 42
    agent = FakeAgent.last_instance
    assert agent is not None
    assert FakeAgent.construct_count == 1
    assert agent.run_count == 2


def test_agent_exit_validation_and_retry_recovers(fake_llm):
    ''' agent 返回类型不符的结果 → 出口校验拦截 → retry 重试整个 agent run (D3 正交性) '''
    fake_llm(['unused'])
    FakeAgent.script = ['{"result": "not-an-int"}', '{"result": 42}']

    @ai_powered(agent = FakeAgent, retry = 1)
    def fn(x: int) -> int:
        ''' doc '''
        ...

    assert fn(6) == 42
    agent = FakeAgent.last_instance
    assert agent is not None
    assert agent.run_count == 2  # retry 重试的是 run, 不是 run 内部


def test_retry_exhausted_reruns_whole_agent_run(fake_llm):
    fake_llm(['unused'])
    FakeAgent.script = ['{"result": "still-not-int"}']

    @ai_powered(agent = FakeAgent, retry = 2)
    def fn(x: int) -> int:
        ''' doc '''
        ...

    with pytest.raises(msgspec.DecodeError):
        fn(6)
    agent = FakeAgent.last_instance
    assert agent is not None
    assert agent.run_count == 3  # 总尝试 = 1 + retry


def test_agent_run_error_is_retryable(fake_llm):
    ''' AgentRunError (多轮耗尽/未知工具等采样噪声) 纳入可重试集合 —— 与单次路径的 AssertionError 同类 '''
    fake_llm(['unused'])
    FakeAgent.script = [AgentRunError("max rounds exhausted"), '{"result": 42}']

    @ai_powered(agent = FakeAgent, retry = 1)
    def fn(x: int) -> int:
        ''' doc '''
        ...

    assert fn(6) == 42
    agent = FakeAgent.last_instance
    assert agent is not None
    assert agent.run_count == 2


def test_bare_async_run_rejected_at_decoration_time(fake_llm):
    ''' 裸 async def run (漏 @sync_compatible) 装饰期拦截 —— sync 路径会在首次调用时爆晦涩 AttributeError '''
    fake_llm(['unused'])

    class BareAsyncAgent(FunctionAgent):
        async def run(self, arguments_json: str) -> str:  #type: ignore[override]  # 故意违反协议
            return '{"result": 42}'

    with pytest.raises(ValueError, match = "sync_compatible"):
        @ai_powered(agent = BareAsyncAgent)
        def fn(x: int) -> int:
            ''' doc '''
            ...


def test_agent_required_features_mismatch_rejected(fake_llm):
    ''' agent 声明的 required_features 超出模型能力 → 装饰期 fail-fast (而非运行中长诊断链) '''
    fake_llm(['unused'])  # fake_llm 的 config 只含 {structured_outputs} 不含 tools, 与 ReasoningAgent 的依赖冲突

    with pytest.raises(ValueError, match = "requires model features"):
        @ai_powered(agent = ReasoningAgent)
        def fn(x: int) -> int:
            ''' doc '''
            ...


def test_agent_non_class_rejected_at_decoration_time(fake_llm):
    ''' D3: 仅接受 class —— 传 instance (或任意非 class 对象) 装饰期 fail-fast '''
    fake_llm(['unused'])

    with pytest.raises(ValueError, match = "class"):
        @ai_powered(agent = object())  #type: ignore[arg-type]
        def fn1(x: int) -> int:
            ''' doc '''
            ...

    with pytest.raises(ValueError, match = "class"):
        @ai_powered(agent = "not-a-class")  #type: ignore[arg-type]
        def fn2(x: int) -> int:
            ''' doc '''
            ...


@pytest.mark.asyncio
async def test_agent_with_async_function(fake_llm):
    fake_llm(['unused'])
    FakeAgent.script = ['{"result": 42}']

    @ai_powered(agent = FakeAgent)
    async def afn(x: int) -> int:
        ''' doc '''
        ...

    assert await afn(6) == 42


def test_agent_async_wait_in_sync_context(fake_llm):
    ''' agent 路径的 .wait() (sync 转写体) —— _AgentEngine.query_model 转写要求 agent.run 为 sync_compatible '''
    fake_llm(['unused'])
    FakeAgent.script = ['{"result": 42}']

    @ai_powered(agent = FakeAgent)
    async def afn(x: int) -> int:
        ''' doc '''
        ...

    assert cast(Waitable[int], afn(6)).wait() == 42


def _spec() -> FunctionSpec:
    return FunctionSpec("fn", "def fn(x: int) -> int", "doc", {"x": {"type": "integer"}}, {"type": "integer"})


def test_reasoning_agent_submits_after_musing():
    ''' 参考实现: 先输出一轮纯文本推理, 再调用 return_result 提交 → 两轮循环, 中间推理累积进对话 '''
    conn = ScriptedCompletionConnection([make_content_completion("let me think: 6 * 7 ..."), make_tool_call_completion("return_result", '{"result": 42}')])
    agent = ReasoningAgent(_spec(), _REASONING_MODEL, cast(LlmConnection, conn))

    assert agent.run('{"x": 6}').wait() == '{"result": 42}'
    assert conn.call_count == 2
    # 对话累积: 第 2 轮请求应含 (system, user, assistant 推理) 三条
    assert [m["role"] for m in conn.seen_messages[1]] == ["system", "user", "assistant"]


def test_reasoning_agent_empty_round_nudges_and_recovers():
    ''' 空 content 且无工具调用的轮次: 追问 (维持 role 交替) 而非终止/死循环, 下一轮提交成功 '''
    conn = ScriptedCompletionConnection([make_content_completion(None), make_content_completion(None), make_tool_call_completion("return_result", '{"result": 7}')])
    agent = ReasoningAgent(_spec(), _REASONING_MODEL, cast(LlmConnection, conn))

    assert agent.run('{"x": 6}').wait() == '{"result": 7}'
    assert conn.call_count == 3
    # 第 2 轮请求 = (system, user, assistant空占位, user追问); 第 3 轮再加一组追问
    assert [m["role"] for m in conn.seen_messages[1]] == ["system", "user", "assistant", "user"]
    assert [m["role"] for m in conn.seen_messages[2]] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_reasoning_agent_parallel_tool_calls_pick_return_result():
    ''' 并行工具调用 [其它工具, return_result]: 取 return_result, 忽略其它 '''
    other = ChatCompletionMessageFunctionToolCall(id = "call-0", function = Function(name = "do_something_else", arguments = "{}"), type = "function")
    wanted = ChatCompletionMessageFunctionToolCall(id = "call-1", function = Function(name = "return_result", arguments = '{"result": 42}'), type = "function")
    resp = ChatCompletion(
        id = "fake", created = 0, model = "m", object = "chat.completion",
        choices = [Choice(finish_reason = "tool_calls", index = 0, message = ChatCompletionMessage(role = "assistant", content = None, tool_calls = [other, wanted]))],
    )
    conn = ScriptedCompletionConnection([resp])
    agent = ReasoningAgent(_spec(), _REASONING_MODEL, cast(LlmConnection, conn))

    assert agent.run('{"x": 6}').wait() == '{"result": 42}'


def test_reasoning_agent_max_rounds_exhausted():
    ''' 一直只输出文本不提交 → max_rounds 轮后 AgentRunError (可被 retry 捕获) '''
    conn = ScriptedCompletionConnection([make_content_completion("thinking ...")])  # 耗尽重复末元素

    class EagerAgent(ReasoningAgent):
        max_rounds = 3

    agent = EagerAgent(_spec(), _REASONING_MODEL, cast(LlmConnection, conn))

    with pytest.raises(AgentRunError, match = "3 rounds"):
        agent.run('{}').wait()
    assert conn.call_count == 3


def test_reasoning_agent_rejects_unexpected_tool():
    conn = ScriptedCompletionConnection([make_tool_call_completion("do_something_else", '{}')])
    agent = ReasoningAgent(_spec(), _REASONING_MODEL, cast(LlmConnection, conn))

    with pytest.raises(AgentRunError, match = "do_something_else"):
        agent.run('{}').wait()


def test_return_result_tool_param_shape():
    ''' 提交契约 SSOT: return_result 工具参数 schema 即 {"result": 返回值schema} (模拟器与 agent 共用) '''
    param = return_result_tool_param({"type": "integer"})
    assert param == {
        "type": "function",
        "function": {
            "name": "return_result",
            "parameters": {"type": "object", "properties": {"result": {"type": "integer"}}, "required": ["result"]},
        },
    }
