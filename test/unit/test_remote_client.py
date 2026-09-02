'''
职责: AgentServiceClient 的离线单元测试 —— 以 httpx.MockTransport 脚本化 agent-service 的
注册/invoke 两端点, 覆盖错误映射全表 (completed/failed/404 自愈/422/网络异常), 内容寻址缓存,
sync/async 双形态与超时构造透传. 全部离线, 无真实网络 IO.

FakeAgentService 模式: 两端点各自消费脚本队列 (Exception 则在 handler 内抛出, Response 则返回),
耗尽重复末元素; spy 暴露注册/调用计数与 invoke 实际命中的 agent_id 序列 —— 缓存与自愈路由的断言锚点.
'''
from typing import Any

import httpx
import pytest

from ai_powered.agent import AgentRunError
from ai_powered.remote.client import AgentServiceClient, RemoteContractError

#: 代表性 AgentDefinition (client 对其形状透传不解释, wire 契约见 client.py 模块头;
#: 形态对齐真实注册体 —— 字段带 type 包装, 输出顶层为 product (服务端硬约束))
DEFN: dict[str, Any] = {
    "name": "greet",
    "system_prompt": "You are a concise greeter.",
    "model": "glm-4.7-flash",
    "input_spec": {"kind": "product", "fields": {"name": {"type": {"kind": "string"}}}},
    "output_spec": {"kind": "product", "fields": {"result": {"type": {"kind": "string"}}}},
}

#: 缓存区分度用的第二个 defn (异 name 即异内容寻址 key)
DEFN_2: dict[str, Any] = {**DEFN, "name": "greet-v2"}


def _error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _register_ok(agent_id: str) -> httpx.Response:
    return httpx.Response(200, json = {"agent_id": agent_id})


def _invoke_completed(result: Any) -> httpx.Response:
    return httpx.Response(200, json = {"status": "completed", "engine": "fake-engine", "elapsed_ms": 3, "result": result, "error": None})


def _invoke_failed(message: str) -> httpx.Response:
    return httpx.Response(200, json = {"status": "failed", "engine": "fake-engine", "elapsed_ms": 3, "result": None, "error": message})


def _agent_not_found(agent_id: str) -> httpx.Response:
    return httpx.Response(404, json = _error_body("agent_not_found", f"no agent registered with id {agent_id!r}"))


class FakeAgentService:
    ''' 脚本化 agent-service: 注册 (POST /v1/agents) 与调用 (POST /v1/agents/{id}/invoke) 两端点
    各自消费脚本队列, 耗尽重复末元素; 未预期的请求路径直接 AssertionError (fail-fast).

    单 MockTransport 实例同时注入被测 client 的 sync/async 双 httpx 客户端
    (MockTransport 同时实现两 handle 协议, 见 client.py 模块头 "transport seam").
    '''

    register_calls: int
    invoke_calls: int
    #: 每次 invoke 实际命中的 agent_id 序列 (URL path 剥离), 自愈路由断言用
    seen_invoke_agent_ids: list[str]

    def __init__(self, *, register_script: list[httpx.Response | Exception] | None = None, invoke_script: list[httpx.Response | Exception] | None = None) -> None:
        self.register_script: list[httpx.Response | Exception] = register_script if register_script is not None else [_register_ok("agent-001")]
        self.invoke_script: list[httpx.Response | Exception] = invoke_script if invoke_script is not None else [_invoke_completed("ok")]
        self.register_calls = 0
        self.invoke_calls = 0
        self.seen_invoke_agent_ids = []

    @staticmethod
    def _next(script: list[httpx.Response | Exception], nth: int) -> httpx.Response | Exception:
        return script[min(nth, len(script) - 1)]

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/v1/agents":
            step = self._next(self.register_script, self.register_calls)
            self.register_calls += 1
        elif request.method == "POST" and path.startswith("/v1/agents/") and path.endswith("/invoke"):
            step = self._next(self.invoke_script, self.invoke_calls)
            self.seen_invoke_agent_ids.append(path.removeprefix("/v1/agents/").removesuffix("/invoke"))
            self.invoke_calls += 1
        else:
            raise AssertionError(f"unexpected request: {request.method} {path}")
        if isinstance(step, Exception):
            raise step
        return step

    def make_client(self, **kwargs: Any) -> AgentServiceClient:
        transport = httpx.MockTransport(self.handler)
        return AgentServiceClient("http://agent-service.test", transport = transport, **kwargs)


def test_register_returns_agent_id() -> None:
    svc = FakeAgentService(register_script = [_register_ok("agent-42")])
    client = svc.make_client()
    assert client.register_agent(DEFN).wait() == "agent-42"


def test_register_422_raises_contract_error_with_server_message() -> None:
    svc = FakeAgentService(register_script = [httpx.Response(422, json = _error_body("invalid_spec", "output_spec: unknown kind 'yaml'"))])
    client = svc.make_client()
    with pytest.raises(RemoteContractError, match = "unknown kind 'yaml'") as excinfo:
        client.register_agent(DEFN).wait()
    assert not isinstance(excinfo.value, AgentRunError)  # 422 是编程错误, 不得落在可重试异常族


def test_invoke_completed_returns_result_field() -> None:
    svc = FakeAgentService(invoke_script = [_invoke_completed({"answer": 7})])
    client = svc.make_client()
    agent_id = client.register_agent(DEFN).wait()
    assert client.invoke_agent(agent_id, {"name": "world"}).wait() == {"answer": 7}


def test_invoke_failed_envelope_raises_run_error() -> None:
    svc = FakeAgentService(invoke_script = [_invoke_failed("sandbox crashed")])
    client = svc.make_client()
    with pytest.raises(AgentRunError, match = "sandbox crashed"):
        client.invoke(DEFN, {"name": "world"}).wait()


def test_invoke_404_self_heals_by_reregistering() -> None:
    svc = FakeAgentService(
        register_script = [_register_ok("agent-1"), _register_ok("agent-2")],
        invoke_script = [_agent_not_found("agent-1"), _invoke_completed("healed")],
    )
    client = svc.make_client()
    assert client.invoke(DEFN, {"name": "world"}).wait() == "healed"
    assert svc.register_calls == 2  # 自愈触发了重新注册
    assert svc.seen_invoke_agent_ids == ["agent-1", "agent-2"]  # 重试路由到重注册返回的新 id


def test_invoke_404_twice_raises_run_error() -> None:
    svc = FakeAgentService(invoke_script = [_agent_not_found("agent-1"), _agent_not_found("agent-1")])
    client = svc.make_client()
    with pytest.raises(AgentRunError, match = "not found"):
        client.invoke(DEFN, {"name": "world"}).wait()
    assert svc.register_calls == 2  # 自愈已尝试过一次重注册, 仍 404 才放弃


def test_invoke_422_raises_contract_error() -> None:
    svc = FakeAgentService(invoke_script = [httpx.Response(422, json = _error_body("invalid_request", "input missing required field: name"))])
    client = svc.make_client()
    with pytest.raises(RemoteContractError, match = "input missing required field"):
        client.invoke(DEFN, {}).wait()


def test_network_error_raises_run_error() -> None:
    svc = FakeAgentService(invoke_script = [httpx.ConnectError("connection refused")])
    client = svc.make_client()
    with pytest.raises(AgentRunError, match = "connection refused"):
        client.invoke(DEFN, {"name": "world"}).wait()


def test_content_addressed_cache_skips_second_registration() -> None:
    svc = FakeAgentService(invoke_script = [_invoke_completed(1), _invoke_completed(2), _invoke_completed(3)])
    client = svc.make_client()
    assert client.invoke(DEFN, {"name": "a"}).wait() == 1
    assert client.invoke(DEFN, {"name": "b"}).wait() == 2
    assert svc.register_calls == 1  # 同 defn 复用缓存, 第二次 invoke 不再注册
    assert client.invoke(DEFN_2, {"name": "c"}).wait() == 3
    assert svc.register_calls == 2  # 异 defn 是独立缓存项


def test_sync_form_wait_smoke() -> None:
    ''' sync 形态冒烟: 三方法均经 .wait() 同步取值 (上游 easy_sync 转写体的调用形态) '''
    svc = FakeAgentService(invoke_script = [_invoke_completed("via-invoke-agent"), _invoke_completed("via-invoke")])
    client = svc.make_client()
    agent_id = client.register_agent(DEFN).wait()
    assert client.invoke_agent(agent_id, {"name": "w"}).wait() == "via-invoke-agent"
    assert client.invoke(DEFN, {"name": "w"}).wait() == "via-invoke"


@pytest.mark.asyncio
async def test_async_form_smoke() -> None:
    ''' async 形态冒烟: 三方法均可直接 await 取值 (Waitable 即 Awaitable) '''
    svc = FakeAgentService(invoke_script = [_invoke_completed("awaited")])
    client = svc.make_client()
    agent_id = await client.register_agent(DEFN)
    assert await client.invoke_agent(agent_id, {"name": "w"}) == "awaited"
    assert await client.invoke(DEFN, {"name": "w"}) == "awaited"


def test_timeout_is_passed_through_to_both_http_clients() -> None:
    svc = FakeAgentService()
    client = svc.make_client(timeout = 0.5)
    assert client.sync_http.timeout.connect == 0.5
    assert client.async_http.timeout.connect == 0.5


def test_register_non_json_200_body() -> None:
    ''' L-4 分支: 200 但 body 非 JSON -> RemoteContractError (服务端违约, 不可 retry) '''
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text = "<html>oops</html>"))
    client = AgentServiceClient("http://test", transport = transport)
    with pytest.raises(RemoteContractError, match = "non-JSON"):
        client.register_agent(DEFN).wait()


def test_register_envelope_without_agent_id() -> None:
    ''' L-4 分支: 200 JSON 但信封无 agent_id -> RemoteContractError '''
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json = {"unexpected": 1}))
    client = AgentServiceClient("http://test", transport = transport)
    with pytest.raises(RemoteContractError, match = "agent_id"):
        client.register_agent(DEFN).wait()


def test_invoke_non_json_200_body() -> None:
    ''' L-4 分支: invoke 200 但 body 非 JSON -> RemoteContractError '''
    calls = {"n": 0}
    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json = {"agent_id": "id-1"})
        return httpx.Response(200, text = "not json")
    client = AgentServiceClient("http://test", transport = httpx.MockTransport(handler))
    with pytest.raises(RemoteContractError, match = "non-JSON"):
        client.invoke(DEFN, {}).wait()


def test_invoke_unknown_envelope_status() -> None:
    ''' L-4 分支: status 未知值 -> malformed envelope -> RemoteContractError '''
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json = {"agent_id": "id-1"})
        return httpx.Response(200, json = {"status": "pending", "engine": "x", "elapsed_ms": 1, "result": None, "error": None})
    client = AgentServiceClient("http://test", transport = httpx.MockTransport(handler))
    with pytest.raises(RemoteContractError, match = "malformed"):
        client.invoke(DEFN, {}).wait()


def test_invoke_completed_without_result_key() -> None:
    ''' L-1 回归锚: completed 但信封缺 result 键 (区别于 result: null) -> RemoteContractError '''
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json = {"agent_id": "id-1"})
        return httpx.Response(200, json = {"status": "completed", "engine": "x", "elapsed_ms": 1})
    client = AgentServiceClient("http://test", transport = httpx.MockTransport(handler))
    with pytest.raises(RemoteContractError, match = "result"):
        client.invoke(DEFN, {}).wait()


def test_5xx_maps_to_agent_run_error() -> None:
    ''' L-4 分支: 5xx -> AgentRunError (瞬时, 可 retry) '''
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json = {"error": {"code": "unavailable", "message": "overloaded"}})
    client = AgentServiceClient("http://test", transport = httpx.MockTransport(handler))
    with pytest.raises(AgentRunError, match = "503"):
        client.invoke_agent("some-id", {}).wait()


def test_invoke_agent_single_shot_404_no_healing() -> None:
    ''' L-4 分支: invoke_agent 单发 404 不自愈 (无 defn 可用) -> 直接 AgentRunError, 注册端点零流量 '''
    calls = {"register": 0}
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/agents":
            calls["register"] += 1
            return httpx.Response(200, json = {"agent_id": "id-1"})
        return httpx.Response(404, json = {"error": {"code": "agent_not_found", "message": "未注册"}})
    client = AgentServiceClient("http://test", transport = httpx.MockTransport(handler))
    with pytest.raises(AgentRunError, match = "not found"):
        client.invoke_agent("missing-id", {}).wait()
    assert calls["register"] == 0


def test_close_is_waitable() -> None:
    ''' M-4: close 双形态存在且 sync 路径可调用 (不炸即为过 —— 显式资源回收入口) '''
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json = {"agent_id": "id-1"}))
    client = AgentServiceClient("http://test", transport = transport)
    client.close().wait()


def test_agent_id_url_quoted() -> None:
    ''' L-3 回归锚: agent_id 含路径特殊字符时被 quote (safe=""), 不静默路由错
    (断言用 raw_path —— httpx 的 url.path 属性返回解码形态, 编码信息只在 raw_path) '''
    seen: list[bytes] = []
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.raw_path)
        if request.url.path.endswith("/invoke"):
            return httpx.Response(200, json = {"status": "completed", "engine": "x", "elapsed_ms": 1, "result": None})
        return httpx.Response(200, json = {"agent_id": "weird/id"})
    client = AgentServiceClient("http://test", transport = httpx.MockTransport(handler))
    client.invoke_agent("weird/id", {}).wait()
    assert seen == [b"/v1/agents/weird%2Fid/invoke"]
