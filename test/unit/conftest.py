'''
职责: test/unit 共享测试基建 —— 脚本化 FakeLlmConnection / ScriptedCompletionConnection 与 completion 工厂.

FakeLlmConnection 依次消费脚本元素 (Exception 实例则抛出, str 则作为 content 返回),
脚本耗尽后重复最后一个元素; @sync_compatible 与真实 LlmConnection 对齐 (easy_sync 转写要求).

ScriptedCompletionConnection 依次返回预构造 ChatCompletion (支持 tool_call 形态), 耗尽重复末元素;
spy 记录每轮请求的 messages 供对话累积断言 —— agent 类 (ReasoningAgent/GenericAgent) 的离线驱动器.

(经 pyright strict + pytest prepend 解析实证, 兄弟测试文件可安全 `from conftest import ...`)
'''
from typing import Any, Optional, cast

import pytest
from easy_sync import sync_compatible
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall, Function
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

import ai_powered.decorators as decorators
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import FunctionSpec, ModelFeature
from ai_powered.llm.known_models import KnownModel
from ai_powered.llm.resolver import ResolvedModel

#: agent 离线测试的模型配置 (声明 tools 能力, 满足 ReasoningAgent/GenericAgent 的 required_features)
FAKE_TOOLS_MODEL = KnownModel("fake-model", {ModelFeature.tools})


def make_spec() -> FunctionSpec:
    ''' agent 测试通用的最小 FunctionSpec '''
    return FunctionSpec("fn", "def fn(x: int) -> int", "doc", {"x": {"type": "integer"}}, {"type": "integer"})


def make_content_completion(content: Optional[str]) -> ChatCompletion:
    return ChatCompletion(
        id = "fake-completion", created = 0, model = "fake-model", object = "chat.completion",
        choices = [Choice(finish_reason = "stop", index = 0, message = ChatCompletionMessage(role = "assistant", content = content))],
    )


def make_tool_call_completion(tool_name: str, arguments_json: str) -> ChatCompletion:
    ''' 模拟模型调用工具 (return_result / python 等) 提交结果 '''
    tc = ChatCompletionMessageFunctionToolCall(id = "call-1", function = Function(name = tool_name, arguments = arguments_json), type = "function")
    return ChatCompletion(
        id = "fake-completion", created = 0, model = "fake-model", object = "chat.completion",
        choices = [Choice(finish_reason = "tool_calls", index = 0, message = ChatCompletionMessage(role = "assistant", content = None, tool_calls = [tc]))],
    )


class ScriptedCompletionConnection:
    ''' 脚本化连接: 依次返回预构造 ChatCompletion, 耗尽重复末元素; spy 记录每轮请求的 messages '''

    def __init__(self, script: list[ChatCompletion]):
        self.script = list(script)
        self.call_count = 0
        self.seen_messages: list[list[ChatCompletionMessageParam]] = []
        self.seen_tools: list[list[str]] = []  # 每轮请求的 tool 名列表 (工具面断言)
        self.seen_tool_params: list[list[ChatCompletionToolParam]] = []  # 每轮请求的完整 tool 参数 (schema 门控断言, issue 18)

    @sync_compatible
    async def chat_completions(self, **kwargs: Any) -> ChatCompletion:
        self.seen_messages.append(list(kwargs.get("messages", [])))  # 拷贝快照 —— 调用方可能继续 append 同一 list
        self.seen_tools.append([t["function"]["name"] for t in kwargs.get("tools", [])])  #type: ignore[index]
        self.seen_tool_params.append(list(kwargs.get("tools", [])))  #type: ignore[arg-type]
        step = self.script[min(self.call_count, len(self.script) - 1)]
        self.call_count += 1
        return step


class FakeLlmConnection:
    ''' 脚本化连接: 依次消费脚本元素, Exception 实例则抛出, str 则作为 content 返回; 脚本耗尽后重复最后一个元素 '''

    def __init__(self, script: list[Exception | str]):
        self.script = list(script)
        self.call_count = 0
        self.seen_models: list[str | None] = []  # 记录 resolve_model 收到的 model 参数 (spy 用)

    @sync_compatible
    async def chat_completions(self, **kwargs: Any) -> ChatCompletion:
        ''' 与 LlmConnection.chat_completions 同为 sync_compatible, 兼容 easy_sync 的 sync 转写 '''
        step = self.script[min(self.call_count, len(self.script) - 1)]
        self.call_count += 1
        if isinstance(step, Exception):
            raise step
        return make_content_completion(step)


@pytest.fixture
def fake_llm(monkeypatch):
    ''' 安装脚本化连接 (替换 resolve_model 的返回, pin 到 StructuredOutput 路径 / 裸 JSON 响应), 使测试与宿主环境变量解耦;
    fake.seen_models 记录装饰期收到的 model 参数 (供 spy 断言) '''
    def setup(script: list[Exception | str]) -> FakeLlmConnection:
        fake = FakeLlmConnection(script)
        fake_config = KnownModel("fake-model", {ModelFeature.structured_outputs})

        def fake_resolve(model: str | None) -> ResolvedModel:
            fake.seen_models.append(model)
            return ResolvedModel(cast(LlmConnection, fake), fake_config)

        monkeypatch.setattr(decorators, "resolve_model", fake_resolve)
        return fake
    return setup
