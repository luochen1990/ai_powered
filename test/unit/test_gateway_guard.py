'''
职责: LlmConnection 网关防御 (_guard_gateway_error) 的离线单元测试.

钉住 guard ↔ retry 的契约拼接点: OpenRouter 等网关以 HTTP 200 + body.error 透传上游
瞬时故障时, guard 将其转为专用的 LlmGatewayError (可被 @ai_powered(retry=N) 重试),
而非让调用方在 choices[0] 处爆 opaque TypeError.
'''
from typing import Any, Optional

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from ai_powered.llm.connection import LlmGatewayError, _guard_gateway_error


def make_completion(content: Optional[str], **extra: Any) -> ChatCompletion:
    return ChatCompletion(
        id = "fake-completion", created = 0, model = "fake-model", object = "chat.completion",
        choices = [Choice(finish_reason = "stop", index = 0, message = ChatCompletionMessage(role = "assistant", content = content))],
        **extra,
    )


def make_gateway_response(error: Any) -> ChatCompletion:
    ''' 模拟网关 200+error 形态: choices 为 None, error 在 body 顶层

    用 model_construct 直构: 构造器会因 choices 必填 List[Choice] 而拒绝 None,
    而 construct 是 openai SDK 解析响应的真实路径 (生产中能出现 choices=None 本身
    即证明 SDK 走无校验构造), error 等未知字段自动落入 model_extra.
    '''
    return ChatCompletion.model_construct(
        id = "fake-completion", created = 0, model = "fake-model", object = "chat.completion",
        choices = None, error = error,
    )


def test_guard_passes_normal_response_through():
    resp = make_completion('{"result": 42}')
    assert _guard_gateway_error(resp) is resp


def test_guard_raises_gateway_error_with_upstream_detail():
    resp = make_gateway_response({"code": 503, "message": "overloaded"})
    with pytest.raises(LlmGatewayError, match = "503"):
        _guard_gateway_error(resp)


def test_guard_error_message_without_error_detail():
    ''' choices 为 None 但 body 无 error 键的畸形响应: 消息含 None 但仍抛 LlmGatewayError 而非 TypeError '''
    resp = ChatCompletion.model_construct(
        id = "fake-completion", created = 0, model = "fake-model", object = "chat.completion",
        choices = None,
    )
    with pytest.raises(LlmGatewayError, match = "None"):
        _guard_gateway_error(resp)


def test_gateway_error_is_runtime_error_subclass():
    ''' RuntimeError 基类契约: 下游按裸 RuntimeError 捕获 (master 上曾经的形态) 依然命中 '''
    assert issubclass(LlmGatewayError, RuntimeError)
