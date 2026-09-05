'''
职责: AgentServiceClient 的离线单元测试 —— 以 httpx.MockTransport 脚本化 agent-service 的
注册/提交/任务查询三端点, 覆盖错误映射全表 (202→轮询至 done/failed/cancelled/预算耗尽/
旧信封 200 兼容/404 自愈/422/网络异常), 内容寻址缓存, sync/async 双形态与超时构造透传.
全部离线, 无真实网络 IO.

FakeAgentService 模式: 三端点各自消费脚本队列 (Exception 则在 handler 内抛出, Response 则返回),
耗尽重复末元素; spy 暴露注册/提交/查询计数与 invoke 实际命中的 agent_id 序列 —— 缓存与自愈
路由的断言锚点. ad-hoc 用例 (自带 transport 的形状违约类) 经 _offline_client/_submit_then_poll_transport
脚手架构造; 全文件轮询间隔一律压到毫秒级 (poll_interval=0.01), 测试耗时与生产默认解耦.
'''
import time
from typing import Any

import httpx
import pytest

from ai_powered.agent import AgentRunError
from ai_powered.remote.client import AgentServiceClient, RemoteContractError, RemoteTaskCancelled, RemoteTaskFailed, RemoteTaskTimeout

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

#: 脚本默认使用的 task_id (提交/查询两端点共用)
_TASK_ID = "task-001"


def _error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _register_ok(agent_id: str) -> httpx.Response:
    return httpx.Response(200, json = {"agent_id": agent_id})


def _submit_accepted(task_id: str = _TASK_ID) -> httpx.Response:
    ''' 新契约的提交受理响应 (202) '''
    return httpx.Response(202, json = {"task_id": task_id, "status": "queued"})


def _task_status(status: str) -> httpx.Response:
    ''' 非终态任务体 (queued/running/paused) '''
    return httpx.Response(200, json = {"task_id": _TASK_ID, "status": status, "rounds": 1, "tokens_used": 10})


def _task_done(result: Any) -> httpx.Response:
    return httpx.Response(200, json = {"task_id": _TASK_ID, "status": "done", "rounds": 2, "tokens_used": 20, "result": result, "engine": "fake-engine", "elapsed_ms": 5, "usage": None})


def _task_failed(message: str, *, upstream_code: str | None = None, next_flush_time: str | None = None) -> httpx.Response:
    body: dict[str, Any] = {"task_id": _TASK_ID, "status": "failed", "rounds": 1, "tokens_used": 10, "error": message, "engine": "fake-engine", "elapsed_ms": 5}
    if upstream_code is not None:
        body["upstream_code"] = upstream_code
    if next_flush_time is not None:
        body["next_flush_time"] = next_flush_time
    return httpx.Response(200, json = body)


def _task_cancelled(message: str) -> httpx.Response:
    return httpx.Response(200, json = {"task_id": _TASK_ID, "status": "cancelled", "rounds": 0, "tokens_used": 0, "error": message})


def _invoke_completed(result: Any) -> httpx.Response:
    ''' 旧契约 (过渡兼容) 的同步信封 '''
    return httpx.Response(200, json = {"status": "completed", "engine": "fake-engine", "elapsed_ms": 3, "result": result, "error": None})


def _invoke_failed(message: str, *, upstream_code: str | None = None, next_flush_time: str | None = None) -> httpx.Response:
    body: dict[str, Any] = {"status": "failed", "engine": "fake-engine", "elapsed_ms": 3, "result": None, "error": message}
    if upstream_code is not None:
        body["upstream_code"] = upstream_code
    if next_flush_time is not None:
        body["next_flush_time"] = next_flush_time
    return httpx.Response(200, json = body)


def _agent_not_found(agent_id: str) -> httpx.Response:
    return httpx.Response(404, json = _error_body("agent_not_found", f"no agent registered with id {agent_id!r}"))


def _task_not_found(task_id: str) -> httpx.Response:
    return httpx.Response(404, json = _error_body("task_not_found", f"no task with id {task_id!r}"))


def _offline_client(transport: httpx.MockTransport, **kwargs: Any) -> AgentServiceClient:
    ''' 自带 transport 的 ad-hoc 用例构造器: 默认注入毫秒级轮询间隔 (测试节律与生产
    默认 0.5s 解耦 —— 与 FakeAgentService.make_client 同一 SSOT 惯例, 显式传入时以传入为准) '''
    kwargs.setdefault("poll_interval", 0.01)
    return AgentServiceClient("http://test", transport = transport, **kwargs)


def _submit_then_poll_transport(task_response: httpx.Response) -> httpx.MockTransport:
    ''' 202 受理后任务查询恒回 task_response 的最小脚手架 (任务体违约用例共用:
    注册与提交两分支对全部用例恒定, 用例体只保留差异行 —— 违约任务体本身) '''
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/agents":
            return _register_ok("id-1")
        if request.url.path.endswith("/invoke"):
            return _submit_accepted("t1")
        return task_response
    return httpx.MockTransport(handler)


def _slow_task_transport() -> httpx.MockTransport:
    ''' 任务查询慢响应脚手架 (钳制超时转译用例的 sync/async 两钉子共用):
    消耗掉全部剩余预算后才抛 ReadTimeout '''
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json = {"agent_id": "id-1"})
        if request.url.path.endswith("/invoke"):
            return httpx.Response(202, json = {"task_id": "t1", "status": "queued"})
        time.sleep(0.12)  # 模拟慢响应: 消耗掉全部剩余预算后才失败
        raise httpx.ReadTimeout("read timed out")
    return httpx.MockTransport(handler)


class FakeAgentService:
    ''' 脚本化 agent-service: 注册 (POST /v1/agents) / 提交 (POST /v1/agents/{id}/invoke) /
    任务查询 (GET /v1/tasks/{task_id}) 三端点各自消费脚本队列, 耗尽重复末元素;
    未预期的请求路径直接 AssertionError (fail-fast).

    单 MockTransport 实例同时注入被测 client 的 sync/async 双 httpx 客户端
    (MockTransport 同时实现两 handle 协议, 见 client.py 模块头 "transport seam").
    '''

    register_calls: int
    invoke_calls: int
    task_calls: int
    #: 每次 invoke 实际命中的 agent_id 序列 (URL path 剥离), 自愈路由断言用
    seen_invoke_agent_ids: list[str]
    #: 任务查询实际命中的 task_id 序列, 轮询路由断言用
    seen_task_ids: list[str]

    def __init__(self, *, register_script: list[httpx.Response | Exception] | None = None, invoke_script: list[httpx.Response | Exception] | None = None, task_script: list[httpx.Response | Exception] | None = None) -> None:
        self.register_script: list[httpx.Response | Exception] = register_script if register_script is not None else [_register_ok("agent-001")]
        self.invoke_script: list[httpx.Response | Exception] = invoke_script if invoke_script is not None else [_submit_accepted()]
        self.task_script: list[httpx.Response | Exception] = task_script if task_script is not None else [_task_done("ok")]
        self.register_calls = 0
        self.invoke_calls = 0
        self.task_calls = 0
        self.seen_invoke_agent_ids = []
        self.seen_task_ids = []

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
        elif request.method == "GET" and path.startswith("/v1/tasks/"):
            step = self._next(self.task_script, self.task_calls)
            self.seen_task_ids.append(path.removeprefix("/v1/tasks/"))
            self.task_calls += 1
        else:
            raise AssertionError(f"unexpected request: {request.method} {path}")
        if isinstance(step, Exception):
            raise step
        return step

    def make_client(self, **kwargs: Any) -> AgentServiceClient:
        return _offline_client(httpx.MockTransport(self.handler), **kwargs)


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


def test_submit_then_poll_through_nonterminal_statuses_to_done() -> None:
    ''' 主链路: 202 受理 -> queued/running/paused 非终态轮询 -> done 返回 result;
    全部轮询命中同一 task_id '''
    svc = FakeAgentService(invoke_script = [_submit_accepted()], task_script = [_task_status("queued"), _task_status("running"), _task_status("paused"), _task_done({"answer": 7})])
    client = svc.make_client()
    agent_id = client.register_agent(DEFN).wait()
    assert client.invoke_agent(agent_id, {"name": "world"}).wait() == {"answer": 7}
    assert svc.task_calls == 4
    assert svc.seen_task_ids == [_TASK_ID] * 4


def test_poll_failed_terminal_raises_with_upstream_fields() -> None:
    ''' failed 终态: RemoteTaskFailed (可 retry 族) + upstream_code/next_flush_time 结构化属性 '''
    svc = FakeAgentService(task_script = [_task_failed("rate limited", upstream_code = "1302", next_flush_time = "2026-09-06T00:00:00Z")])
    client = svc.make_client()
    with pytest.raises(RemoteTaskFailed, match = "rate limited") as excinfo:
        client.invoke(DEFN, {"name": "world"}).wait()
    assert excinfo.value.upstream_code == "1302"
    assert excinfo.value.next_flush_time == "2026-09-06T00:00:00Z"
    assert isinstance(excinfo.value, AgentRunError)  # 可 retry 语义延续旧信封时代


def test_poll_failed_terminal_without_upstream_fields() -> None:
    ''' failed 终态无上游字段时属性为 None (缺省显式, 不伪造) '''
    svc = FakeAgentService(task_script = [_task_failed("boom")])
    client = svc.make_client()
    with pytest.raises(RemoteTaskFailed) as excinfo:
        client.invoke(DEFN, {"name": "world"}).wait()
    assert excinfo.value.upstream_code is None
    assert excinfo.value.next_flush_time is None


def test_poll_cancelled_terminal_raises_distinct_error() -> None:
    ''' cancelled 终态: 独立异常类型 (不挂可重试族 —— 取消是显式决策), 消息含服务端 error '''
    svc = FakeAgentService(task_script = [_task_cancelled("cancelled by operator")])
    client = svc.make_client()
    with pytest.raises(RemoteTaskCancelled, match = "cancelled by operator") as excinfo:
        client.invoke(DEFN, {"name": "world"}).wait()
    assert not isinstance(excinfo.value, AgentRunError)


def test_poll_budget_exhaustion_raises_timeout() -> None:
    ''' 预算耗尽仍非终态 -> RemoteTaskTimeout (可 retry 族); 任务 id 与预算进消息 '''
    svc = FakeAgentService(task_script = [_task_status("running")])
    client = svc.make_client(timeout = 0.1)
    with pytest.raises(RemoteTaskTimeout, match = _TASK_ID) as excinfo:
        client.invoke(DEFN, {"name": "world"}).wait()
    assert isinstance(excinfo.value, AgentRunError)
    assert svc.task_calls >= 2  # 轮询确实发生了多次 (非首查即弃)


def test_legacy_200_completed_returns_result_without_polling() -> None:
    ''' 过渡兼容: 未升级服务端回 200 旧信封 -> 直接解析, 任务查询端点零流量 '''
    svc = FakeAgentService(invoke_script = [_invoke_completed({"answer": 7})])
    client = svc.make_client()
    assert client.invoke(DEFN, {"name": "world"}).wait() == {"answer": 7}
    assert svc.task_calls == 0


def test_legacy_200_failed_carries_upstream_fields() -> None:
    ''' 过渡兼容: 旧信封 failed 同样经 RemoteTaskFailed 透出上游字段 (新旧信封字段同构) '''
    svc = FakeAgentService(invoke_script = [_invoke_failed("quota exhausted", upstream_code = "1308", next_flush_time = "2026-09-06T01:00:00Z")])
    client = svc.make_client()
    with pytest.raises(RemoteTaskFailed, match = "quota exhausted") as excinfo:
        client.invoke(DEFN, {"name": "world"}).wait()
    assert excinfo.value.upstream_code == "1308"
    assert excinfo.value.next_flush_time == "2026-09-06T01:00:00Z"


def test_invoke_404_self_heals_by_reregistering() -> None:
    svc = FakeAgentService(
        register_script = [_register_ok("agent-1"), _register_ok("agent-2")],
        invoke_script = [_agent_not_found("agent-1"), _submit_accepted()],
        task_script = [_task_done("healed")],
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


def test_poll_network_error_raises_run_error() -> None:
    ''' 轮询期网络故障 (提交已成功) -> AgentRunError (瞬时口径, 重试属调用方决策);
    快失败发生在预算耗尽前, 不转译为 RemoteTaskTimeout '''
    svc = FakeAgentService(task_script = [httpx.ConnectError("poll connection reset")])
    client = svc.make_client()
    with pytest.raises(AgentRunError, match = "poll connection reset") as excinfo:
        client.invoke(DEFN, {"name": "world"}).wait()
    assert not isinstance(excinfo.value, RemoteTaskTimeout)


def test_poll_read_timeout_at_deadline_translated_to_task_timeout() -> None:
    ''' 预算尾段的钳制读超时转译为 RemoteTaskTimeout: 调用方以异常类型区分
    "预算太小 (调大 invoke_timeout)" 与 "网络抖动 (直接 retry)" 时信号不失真 '''
    client = _offline_client(_slow_task_transport(), timeout = 0.1)
    with pytest.raises(RemoteTaskTimeout, match = "t1"):
        client.invoke(DEFN, {}).wait()


def test_task_query_404_raises_run_error() -> None:
    ''' 任务记录缺失 (服务端重启丢任务/非法 id) -> AgentRunError, 不触发注册自愈 '''
    svc = FakeAgentService(task_script = [_task_not_found(_TASK_ID)])
    client = svc.make_client()
    with pytest.raises(AgentRunError, match = "task_not_found"):
        client.invoke(DEFN, {"name": "world"}).wait()
    assert svc.register_calls == 1  # 仅初始注册, 无自愈重注册


def test_content_addressed_cache_skips_second_registration() -> None:
    svc = FakeAgentService()
    client = svc.make_client()
    assert client.invoke(DEFN, {"name": "a"}).wait() == "ok"
    assert client.invoke(DEFN, {"name": "b"}).wait() == "ok"
    assert svc.register_calls == 1  # 同 defn 复用缓存, 第二次 invoke 不再注册
    assert client.invoke(DEFN_2, {"name": "c"}).wait() == "ok"
    assert svc.register_calls == 2  # 异 defn 是独立缓存项


def test_sync_form_wait_smoke() -> None:
    ''' sync 形态冒烟: 三方法均经 .wait() 同步取值 (上游 easy_sync 转写体的调用形态) '''
    svc = FakeAgentService()
    client = svc.make_client()
    agent_id = client.register_agent(DEFN).wait()
    assert client.invoke_agent(agent_id, {"name": "w"}).wait() == "ok"
    assert client.invoke(DEFN, {"name": "w"}).wait() == "ok"


@pytest.mark.asyncio
async def test_async_form_smoke() -> None:
    ''' async 形态冒烟: 三方法均可直接 await 取值 (Waitable 即 Awaitable) '''
    svc = FakeAgentService()
    client = svc.make_client()
    agent_id = await client.register_agent(DEFN)
    assert await client.invoke_agent(agent_id, {"name": "w"}) == "ok"
    assert await client.invoke(DEFN, {"name": "w"}) == "ok"


@pytest.mark.asyncio
async def test_async_poll_through_nonterminal_to_done() -> None:
    ''' async 侧多轮轮询 (手写 sync/async 双实现的同构性钉子): .wait() 形态只走 sync
    实现, async_poll_terminal 的多轮迭代 + asyncio.sleep 钳制唯有本用例执行 '''
    svc = FakeAgentService(task_script = [_task_status("queued"), _task_status("running"), _task_done("async-ok")])
    client = svc.make_client()
    assert await client.invoke(DEFN, {"name": "w"}) == "async-ok"
    assert svc.task_calls == 3


@pytest.mark.asyncio
async def test_async_poll_budget_exhaustion() -> None:
    ''' async 侧预算耗尽分支 (同构性钉子的另一半) '''
    svc = FakeAgentService(task_script = [_task_status("running")])
    client = svc.make_client(timeout = 0.1)
    with pytest.raises(RemoteTaskTimeout, match = _TASK_ID):
        await client.invoke(DEFN, {"name": "w"})


@pytest.mark.asyncio
async def test_async_poll_read_timeout_at_deadline_translated_to_task_timeout() -> None:
    ''' async 侧的钳制超时转译分支 (sync/async 手写双实现的同构性钉子) '''
    client = _offline_client(_slow_task_transport(), timeout = 0.1)
    with pytest.raises(RemoteTaskTimeout, match = "t1"):
        await client.invoke(DEFN, {})


def test_timeout_is_passed_through_to_both_http_clients() -> None:
    svc = FakeAgentService()
    client = svc.make_client(timeout = 0.5)
    assert client.sync_http.timeout.connect == 0.5
    assert client.async_http.timeout.connect == 0.5


def test_poll_interval_must_be_positive() -> None:
    ''' 轮询间隔构造期 fail-fast: 非正数会把轮询循环变成忙转/倒转, 属编程错误 '''
    svc = FakeAgentService()
    with pytest.raises(ValueError, match = "poll_interval"):
        svc.make_client(poll_interval = 0.0)


def test_poll_interval_paces_requests() -> None:
    ''' 轮询节律: 固定间隔真实生效 (两次 sleep 下界断言 —— 上界受调度抖动影响不设) '''
    svc = FakeAgentService(task_script = [_task_status("queued"), _task_status("queued"), _task_done("paced")])
    client = svc.make_client(poll_interval = 0.05)
    started = time.monotonic()
    assert client.invoke(DEFN, {"name": "w"}).wait() == "paced"
    assert time.monotonic() - started >= 0.1  # 两次非终态 -> 两次 0.05s 间隔


def test_submit_non_json_202_body() -> None:
    ''' L-4 分支: 202 但 body 非 JSON -> RemoteContractError (服务端违约, 不可 retry) '''
    transport = httpx.MockTransport(lambda request: httpx.Response(202, text = "<html>oops</html>"))
    client = _offline_client(transport)
    with pytest.raises(RemoteContractError, match = "non-JSON"):
        client.invoke_agent("some-id", {}).wait()


def test_submit_202_without_task_id() -> None:
    ''' L-4 分支: 202 JSON 但信封无 task_id -> RemoteContractError '''
    transport = httpx.MockTransport(lambda request: httpx.Response(202, json = {"unexpected": 1}))
    client = _offline_client(transport)
    with pytest.raises(RemoteContractError, match = "task_id"):
        client.invoke_agent("some-id", {}).wait()


def test_register_non_json_200_body() -> None:
    ''' L-4 分支: 200 但 body 非 JSON -> RemoteContractError (服务端违约, 不可 retry) '''
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text = "<html>oops</html>"))
    client = _offline_client(transport)
    with pytest.raises(RemoteContractError, match = "non-JSON"):
        client.register_agent(DEFN).wait()


def test_register_envelope_without_agent_id() -> None:
    ''' L-4 分支: 200 JSON 但信封无 agent_id -> RemoteContractError '''
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json = {"unexpected": 1}))
    client = _offline_client(transport)
    with pytest.raises(RemoteContractError, match = "agent_id"):
        client.register_agent(DEFN).wait()


def test_invoke_unknown_envelope_status() -> None:
    ''' L-4 分支: 旧信封 status 未知值 -> malformed envelope -> RemoteContractError '''
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json = {"agent_id": "id-1"})
        return httpx.Response(200, json = {"status": "pending", "engine": "x", "elapsed_ms": 1, "result": None, "error": None})
    client = _offline_client(httpx.MockTransport(handler))
    with pytest.raises(RemoteContractError, match = "malformed"):
        client.invoke(DEFN, {}).wait()


def test_legacy_200_non_json_body() -> None:
    ''' L-4 分支: 兼容路径 200 但 body 非 JSON -> RemoteContractError (过渡代码最易腐烂, 直锚) '''
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json = {"agent_id": "id-1"})
        return httpx.Response(200, text = "not json")
    client = _offline_client(httpx.MockTransport(handler))
    with pytest.raises(RemoteContractError, match = "non-JSON"):
        client.invoke(DEFN, {}).wait()


def test_legacy_200_completed_without_result_key() -> None:
    ''' L-1 回归锚: 兼容信封 completed 但缺 result 键 (区别于 result: null) -> RemoteContractError '''
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json = {"agent_id": "id-1"})
        return httpx.Response(200, json = {"status": "completed", "engine": "x", "elapsed_ms": 1})
    client = _offline_client(httpx.MockTransport(handler))
    with pytest.raises(RemoteContractError, match = "result"):
        client.invoke(DEFN, {}).wait()


def test_task_body_not_a_dict() -> None:
    ''' L-4 分支: 任务体顶层非 dict (如 JSON 数组) -> RemoteContractError '''
    client = _offline_client(_submit_then_poll_transport(httpx.Response(200, json = [1, 2, 3])))
    with pytest.raises(RemoteContractError, match = "malformed task envelope"):
        client.invoke(DEFN, {}).wait()


def test_task_done_without_result_key() -> None:
    ''' L-1 回归锚: done 但任务体缺 result 键 (区别于 result: null) -> RemoteContractError '''
    client = _offline_client(_submit_then_poll_transport(httpx.Response(200, json = {"task_id": "t1", "status": "done", "rounds": 1, "tokens_used": 1})))
    with pytest.raises(RemoteContractError, match = "result"):
        client.invoke(DEFN, {}).wait()


def test_task_unknown_status() -> None:
    ''' L-4 分支: 任务体 status 值域外 -> RemoteContractError '''
    client = _offline_client(_submit_then_poll_transport(httpx.Response(200, json = {"task_id": "t1", "status": "suspended", "rounds": 0, "tokens_used": 0})))
    with pytest.raises(RemoteContractError, match = "unknown task status"):
        client.invoke(DEFN, {}).wait()


def test_5xx_maps_to_agent_run_error() -> None:
    ''' L-4 分支: 5xx -> AgentRunError (瞬时, 可 retry) '''
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json = {"error": {"code": "unavailable", "message": "overloaded"}})
    client = _offline_client(httpx.MockTransport(handler))
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
    client = _offline_client(httpx.MockTransport(handler))
    with pytest.raises(AgentRunError, match = "not found"):
        client.invoke_agent("missing-id", {}).wait()
    assert calls["register"] == 0


def test_close_is_waitable() -> None:
    ''' M-4: close 双形态存在且 sync 路径可调用 (不炸即为过 —— 显式资源回收入口) '''
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json = {"agent_id": "id-1"}))
    client = _offline_client(transport)
    client.close().wait()


def test_agent_id_url_quoted() -> None:
    ''' L-3 回归锚: agent_id 含路径特殊字符时被 quote (safe=""), 不静默路由错
    (断言用 raw_path —— httpx 的 url.path 属性返回解码形态, 编码信息只在 raw_path) '''
    seen: list[bytes] = []
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.raw_path)
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json = {"agent_id": "weird/id"})
        if request.url.path.endswith("/invoke"):
            return httpx.Response(202, json = {"task_id": "t1", "status": "queued"})
        return httpx.Response(200, json = {"task_id": "t1", "status": "done", "rounds": 1, "tokens_used": 1, "result": None})
    client = _offline_client(httpx.MockTransport(handler))
    client.invoke_agent("weird/id", {}).wait()
    assert seen == [b"/v1/agents/weird%2Fid/invoke", b"/v1/tasks/t1"]


def test_task_id_url_quoted() -> None:
    ''' 轮询路径的 quoting 锚点: task_id 含路径特殊字符时 GET 路径同样被 quote;
    同时精确断言整个请求序列 (无意外额外请求) '''
    seen: list[bytes] = []
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.raw_path)
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json = {"agent_id": "id-1"})
        if request.url.path.endswith("/invoke"):
            return httpx.Response(202, json = {"task_id": "weird/task", "status": "queued"})
        return httpx.Response(200, json = {"task_id": "weird/task", "status": "done", "rounds": 1, "tokens_used": 1, "result": None})
    client = _offline_client(httpx.MockTransport(handler))
    client.invoke_agent("id-1", {}).wait()
    assert seen == [b"/v1/agents/id-1/invoke", b"/v1/tasks/weird%2Ftask"]
