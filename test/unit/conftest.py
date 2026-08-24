'''
职责: test/unit 共享测试基建 —— 脚本化 FakeLlmConnection 与 fake_llm fixture.

FakeLlmConnection 依次消费脚本元素 (Exception 实例则抛出, str 则作为 content 返回),
脚本耗尽后重复最后一个元素; @sync_compatible 与真实 LlmConnection 对齐 (easy_sync 转写要求).
'''
from typing import Any, Optional, cast

import pytest
from easy_sync import sync_compatible
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

import ai_powered.decorators as decorators
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import ModelFeature
from ai_powered.llm.known_models import KnownModel
from ai_powered.llm.resolver import ResolvedModel


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
        self.seen_models: list[str | None] = []  # 记录 resolve_model 收到的 model 参数 (spy 用)

    @sync_compatible
    async def chat_completions(self, **kwargs: Any) -> ChatCompletion:
        ''' 与 LlmConnection.chat_completions 同为 sync_compatible, 兼容 easy_sync 的 sync 转写 '''
        step = self.script[min(self.call_count, len(self.script) - 1)]
        self.call_count += 1
        if isinstance(step, Exception):
            raise step
        return make_completion(step)


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
