'''
职责: RemoteAgent 的离线单元测试 —— 覆盖 defn 组装正确性 / 出口协议对齐 (Result[R] 形态) /
装饰器全链路 (sync 与 async) / 注册缓存 / retry 集成 / invoke 404 自愈链路 / cancelled
不经 retry 直达调用方 / 装饰期 fail-fast.

全部离线: 经 _create_client 工厂注入 httpx.MockTransport 模拟 agent-service 的 wire 契约
(注册幂等 + 202 提交 + 任务轮询终态), 与 test_remote_client.py 的差异: 此处验证
装饰器→agent→client 的完整链路组装, wire 层细节在彼处.
'''
import json
from typing import Any, ClassVar

import httpx
import pytest

from ai_powered import ai_powered
from ai_powered.llm.resolver import resolve_model
from ai_powered.remote import SERVICE_URL_ENV, RemoteAgent, RemoteTaskCancelled
from ai_powered.remote.client import AgentServiceClient

_FAKE_URL = "http://fake-agent-service:8300"

#: fake 服务端使用的固定标识 (提交/查询两端点共用)
_AGENT_ID = "test-agent-id"
_TASK_ID = "task-001"


class FakeAgentService:
    '''脚本化 agent-service: spy 记录请求 + 可编程的提交/任务响应序列 (404 自愈/retry 场景).

    与 test_remote_client.py 的 FakeAgentService 差异: 脚本耗尽抛 IndexError (fail-fast) 而非
    重复末元素 —— agent 层用例依赖精确序列 (retry/自愈的路由断言不容末元素混入).'''

    def __init__(self, invoke_script: list[httpx.Response] | None = None, task_script: list[httpx.Response] | None = None) -> None:
        self.requests: list[tuple[str, Any]] = []  # (path, json body)
        self._invoke_script = list(invoke_script) if invoke_script is not None else None
        self._task_script = list(task_script) if task_script is not None else None

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        self.requests.append((request.url.path, body))
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json = {"agent_id": _AGENT_ID})
        if request.url.path == f"/v1/agents/{_AGENT_ID}/invoke":
            if self._invoke_script is not None:
                return self._invoke_script.pop(0)
            return self._submit_accepted()
        if request.url.path == f"/v1/tasks/{_TASK_ID}":
            if self._task_script is not None:
                return self._task_script.pop(0)
            return self._task_done({"result": 42})
        raise AssertionError(f"unexpected request: {request.url.path}")

    @staticmethod
    def _submit_accepted() -> httpx.Response:
        return httpx.Response(202, json = {"task_id": _TASK_ID, "status": "queued"})

    @staticmethod
    def _task_done(result: Any) -> httpx.Response:
        return httpx.Response(200, json = {"task_id": _TASK_ID, "status": "done", "rounds": 2, "tokens_used": 20, "result": result, "engine": "fake", "elapsed_ms": 1, "usage": None})

    @staticmethod
    def _task_failed(message: str) -> httpx.Response:
        return httpx.Response(200, json = {"task_id": _TASK_ID, "status": "failed", "rounds": 1, "tokens_used": 10, "error": message, "engine": "fake", "elapsed_ms": 1})

    @staticmethod
    def _task_cancelled(message: str) -> httpx.Response:
        return httpx.Response(200, json = {"task_id": _TASK_ID, "status": "cancelled", "rounds": 0, "tokens_used": 0, "error": message})

    @property
    def register_count(self) -> int:
        return sum(1 for path, _ in self.requests if path == "/v1/agents")

    @property
    def invoke_count(self) -> int:
        return sum(1 for path, _ in self.requests if path.endswith("/invoke"))

    @property
    def registered_defn(self) -> dict[str, Any]:
        for path, body in reversed(self.requests):
            if path == "/v1/agents":
                assert isinstance(body, dict)
                return body
        raise AssertionError("no registration was made")


def make_agent_cls(transport: httpx.MockTransport) -> type[RemoteAgent]:
    '''测试注入 seam: 覆盖 _create_client 以替换真实网络 (对齐 test_generic_agent 的 Fake sandbox 惯例);
    轮询间隔压到毫秒级, 测试耗时与生产默认 0.5s 解耦'''
    class _TestAgent(RemoteAgent):
        def _create_client(self, url: str) -> AgentServiceClient:
            return AgentServiceClient(url, transport = transport, poll_interval = 0.01)
    return _TestAgent


@pytest.fixture
def service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SERVICE_URL_ENV, _FAKE_URL)


def test_definition_shape(service_env: None) -> None:
    ''' defn 组装: 签名/docstring 进 prompt, 参数/返回类型编译为 TypeDesc, model 名透传 '''
    from ai_powered.constants import base_system_prompt
    from ai_powered.llm.definitions import FunctionSpec

    fake = FakeAgentService()
    agent_cls = make_agent_cls(fake.transport)
    resolved = resolve_model("fake-model")
    spec = FunctionSpec(
        "translate", "def translate(text: str) -> str", "Translate the text to English.",
        {"text": {"type": "string"}}, {"type": "string"},
        parameter_annotations = {"text": str}, return_annotation = str,
    )

    agent = agent_cls(spec, resolved.model_config, resolved.connection)  # 构造期纯本地编译, 不走网络
    defn = agent._defn  # noqa: SLF001

    assert defn["name"] == "translate"
    assert defn["model"] == "fake-model"
    assert defn["input_spec"] == {"kind": "product", "fields": {"text": {"type": {"kind": "string"}}}}
    assert defn["output_spec"] == {"kind": "product", "fields": {"result": {"type": {"kind": "string"}}}}
    assert base_system_prompt(spec) in defn["system_prompt"]
    assert "sandbox" in defn["system_prompt"]


def test_full_chain_sync(service_env: None) -> None:
    ''' 装饰器全链路 (sync): 参数打包 -> 注册 -> 202 提交 -> 轮询 done -> Result[R] 出口校验 -> 返回值 '''
    fake = FakeAgentService()
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(model = "fake-model", agent = agent_cls)
    def add(a: int, b: int) -> int:
        ''' Add two numbers. '''
        ...

    assert add(1, 2) == 42
    assert fake.register_count == 1
    assert fake.invoke_count == 1
    assert fake.registered_defn["input_spec"] == {
        "kind": "product",
        "fields": {"a": {"type": {"kind": "int"}}, "b": {"type": {"kind": "int"}}},
    }
    invoke_body = next(body for path, body in fake.requests if path.endswith("/invoke"))
    assert invoke_body == {"input": {"a": 1, "b": 2}}  # M1a 信封: 调度参数缺省不发键


@pytest.mark.asyncio
async def test_full_chain_async(service_env: None) -> None:
    ''' 装饰器全链路 (async): await 形态与 sync 共用同一 Waitable 管线 '''
    fake = FakeAgentService()
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(model = "fake-model", agent = agent_cls)
    async def add(a: int, b: int) -> int:
        ''' Add two numbers. '''
        ...

    assert await add(1, 2) == 42


def test_register_cached_across_calls(service_env: None) -> None:
    ''' defn 内容寻址缓存: 多次调用只注册一次 (幂等正确性由服务端保证, 缓存省往返) '''
    fake = FakeAgentService()
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(model = "fake-model", agent = agent_cls)
    def add(a: int, b: int) -> int:
        ''' Add two numbers. '''
        ...

    assert add(1, 2) == 42
    assert add(3, 4) == 42
    assert fake.register_count == 1
    assert fake.invoke_count == 2


def test_retry_on_failed_task(service_env: None) -> None:
    ''' failed 终态 -> RemoteTaskFailed (AgentRunError 族) -> @ai_powered(retry=N) 接管:
    重试单元是整个 agent run (重新提交新任务), 服务端瞬时失败可重试 '''
    fake = FakeAgentService(
        invoke_script = [FakeAgentService._submit_accepted(), FakeAgentService._submit_accepted()],
        task_script = [FakeAgentService._task_failed("boom"), FakeAgentService._task_done({"result": 7})],
    )
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(model = "fake-model", agent = agent_cls, retry = 1)
    def add(a: int, b: int) -> int:
        ''' Add two numbers. '''
        ...

    assert add(1, 2) == 7
    assert fake.invoke_count == 2


def test_cancelled_task_propagates_without_retry(service_env: None) -> None:
    ''' cancelled 终态 -> RemoteTaskCancelled 不在可重试族, retry=N 也不接管 ——
    取消是显式决策, 异常直达调用方 (自动重试会静默重提交新任务) '''
    fake = FakeAgentService(task_script = [FakeAgentService._task_cancelled("killed by operator")])
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(model = "fake-model", agent = agent_cls, retry = 2)
    def add(a: int, b: int) -> int:
        ''' Add two numbers. '''
        ...

    with pytest.raises(RemoteTaskCancelled, match = "killed by operator"):
        add(1, 2)
    assert fake.invoke_count == 1  # 未发生重试


def test_invoke_404_heals_via_reregistration(service_env: None) -> None:
    ''' 服务端重启 (注册表丢失) -> 404 自愈: 弃缓存重注册 -> 重试成功 '''
    fake = FakeAgentService(invoke_script = [
        httpx.Response(404, json = {"error": {"code": "agent_not_found", "message": "未注册"}}),
        FakeAgentService._submit_accepted(),
    ])
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(model = "fake-model", agent = agent_cls)
    def add(a: int, b: int) -> int:
        ''' Add two numbers. '''
        ...

    assert add(1, 2) == 42
    assert fake.register_count == 2  # 初始注册 + 自愈重注册
    assert fake.invoke_count == 2


def test_failfast_unsupported_param_type(service_env: None) -> None:
    ''' 远程通道不支持的类型 (dict) 装饰期即拒绝, 不留到运行期 '''
    agent_cls = make_agent_cls(FakeAgentService().transport)

    with pytest.raises(ValueError, match = "dict"):
        @ai_powered(model = "fake-model", agent = agent_cls)
        def probe(mapping: dict[str, int]) -> int:
            ''' Probe. '''
            ...


def test_failfast_missing_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SERVICE_URL_ENV, raising = False)

    with pytest.raises(ValueError, match = "endpoint"):
        @ai_powered(model = "fake-model", agent = RemoteAgent)
        def probe(x: int) -> int:
            ''' Probe. '''
            ...


def test_model_falls_back_to_resolver_default(service_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    ''' 未指定 model 参数: 注册模型名回落 resolver 的平台默认兜底 (恒非空), 无缺失分支 '''
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising = False)
    fake = FakeAgentService()
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(agent = agent_cls)
    def add(a: int, b: int) -> int:
        ''' Add two numbers. '''
        ...

    assert add(1, 2) == 42
    assert isinstance(fake.registered_defn["model"], str) and fake.registered_defn["model"]


def test_default_model_class_attribute_overrides(service_env: None) -> None:
    ''' default_model 类属性优先于装饰器解析 (继承定制: 固定该 agent 族的注册模型) '''
    fake = FakeAgentService()

    class PinnedModelAgent(RemoteAgent):
        default_model: ClassVar[str | None] = "pinned-model"
        def _create_client(self, url: str) -> AgentServiceClient:
            return AgentServiceClient(url, transport = fake.transport, poll_interval = 0.01)

    @ai_powered(model = "fake-model", agent = PinnedModelAgent)
    def add(a: int, b: int) -> int:
        ''' Add two numbers. '''
        ...

    assert add(1, 2) == 42
    assert fake.registered_defn["model"] == "pinned-model"


def test_service_url_class_attribute_override(monkeypatch: pytest.MonkeyPatch) -> None:
    ''' service_url 类属性优先于环境变量 (继承定制端点的场景) '''
    monkeypatch.delenv(SERVICE_URL_ENV, raising = False)
    fake = FakeAgentService()

    class PinnedAgent(RemoteAgent):
        service_url: ClassVar[str | None] = _FAKE_URL
        def _create_client(self, url: str) -> AgentServiceClient:
            assert url == _FAKE_URL
            return AgentServiceClient(url, transport = fake.transport, poll_interval = 0.01)

    @ai_powered(model = "fake-model", agent = PinnedAgent)
    def add(a: int, b: int) -> int:
        ''' Add two numbers. '''
        ...

    assert add(1, 2) == 42


def test_output_default_fields_required_and_dump_compatible(service_env: None) -> None:
    ''' H-1 回归锚: 返回类型的缺省字段在 output_spec 中必选 (无 maybe) —— 服务端 model_dump 恒含全字段,
    maybe 形态下 LLM 省略缺省字段会产生显式 null, 击穿客户端 Result[R] 解码; 必选编译后 dump 形态恒可解码 '''
    from dataclasses import dataclass

    @dataclass
    class UserInfo:
        name: str
        age: int = 0

    fake = FakeAgentService()
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(model = "fake-model", agent = agent_cls)
    def extract(raw_text: str) -> UserInfo:
        ''' Extract user info. '''
        ...

    # 服务端 canonical dump 形态: 恒含全部字段 (必选编译下 LLM 恒造值, 无 null 陷阱)
    fake._task_script = [FakeAgentService._task_done({"result": {"name": "bob", "age": 5}})]
    assert extract("bob is 5") == UserInfo(name = "bob", age = 5)
    # output_spec 的 age 必选 (不带 maybe 包装)
    assert fake.registered_defn["output_spec"]["fields"]["result"]["type"]["fields"]["age"] == {"type": {"kind": "int"}}


def test_default_parameters_omissible_on_input(service_env: None) -> None:
    ''' 输入侧缺省语境: 带 default 的参数在 input_spec 包 maybe (调用方省略时 bind 不含该参数,
    服务端输入校验对 maybe 允许省略); 显式传参时 body 照常携带 '''
    fake = FakeAgentService(task_script = [
        FakeAgentService._task_done({"result": "hello!"}),
        FakeAgentService._task_done({"result": "hello!"}),
    ])
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(model = "fake-model", agent = agent_cls)
    def greet(name: str, punctuation: str = "!") -> str:
        ''' Greet someone. '''
        ...

    assert greet("bob") == "hello!"  # 省略 punctuation: body 仅含 name, 服务端 maybe 容忍
    assert fake.registered_defn["input_spec"]["fields"] == {
        "name": {"type": {"kind": "string"}},
        "punctuation": {"type": {"kind": "maybe", "element": {"kind": "string"}}},
    }
    first_body = [body for path, body in fake.requests if path.endswith("/invoke")][0]
    assert first_body == {"input": {"name": "bob"}}  # M1a 信封, 调度参数缺省不发键

    assert greet("alice", "?") == "hello!"  # 显式传参: body 携带全字段
    second_body = [body for path, body in fake.requests if path.endswith("/invoke")][1]
    assert second_body == {"input": {"name": "alice", "punctuation": "?"}}


def test_failfast_reserved_parameter_name(service_env: None) -> None:
    ''' M-1: 形参名命中服务端保留面 (下划线/model_ 前缀) 装饰期拒绝, 不留到注册期 422 '''
    agent_cls = make_agent_cls(FakeAgentService().transport)

    with pytest.raises(ValueError, match = "model_config"):
        @ai_powered(model = "fake-model", agent = agent_cls)
        def probe_one(model_config: str) -> int:
            ''' Probe. '''
            ...

    with pytest.raises(ValueError, match = "_unused"):
        @ai_powered(model = "fake-model", agent = agent_cls)
        def probe_two(_unused: int) -> int:
            ''' Probe. '''
            ...


def test_default_param_with_default_fields_combination(service_env: None) -> None:
    ''' 双语境组合锚 (review ④): 参数带 default 且参数类型为含缺省字段的容器 —— 顶层 maybe (参数级)
    与嵌套缺省字段 maybe (defaults_omissible=True) 同时出现, 服务端 maybe 语义双层允许省略 '''
    from dataclasses import dataclass

    @dataclass
    class Config:
        label: str
        retries: int = 0

    fake = FakeAgentService()
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(model = "fake-model", agent = agent_cls)
    def run_with(config: Config = Config("d")) -> int:
        ''' Run with config. '''
        ...

    assert run_with() == 42  # 参数整体省略: body 为 {}, 顶层 maybe 容忍
    assert fake.registered_defn["input_spec"]["fields"]["config"]["type"] == {
        "kind": "maybe",
        "element": {
            "kind": "product",
            "fields": {
                "label": {"type": {"kind": "string"}},
                "retries": {"type": {"kind": "maybe", "element": {"kind": "int"}}},
            },
        },
    }


def test_tools_class_attribute_passed_through(service_env: None) -> None:
    ''' tools 定制面: 继承子类的 ClassVar.tools 透传进注册体 (宿主侧 MCP 白名单引用);
    默认 RemoteAgent 不带 tools 键 (= 服务端仅沙箱六件套) '''
    fake = FakeAgentService(task_script = [FakeAgentService._task_done({"result": "report"})])

    class ToolAgent(RemoteAgent):
        tools: ClassVar[list[str]] = ["web-search", "akshare"]
        def _create_client(self, url: str) -> AgentServiceClient:
            return AgentServiceClient(url, transport = fake.transport, poll_interval = 0.01)

    @ai_powered(model = "fake-model", agent = ToolAgent)
    def research(topic: str) -> str:
        ''' Research a topic. '''
        ...

    assert research("kw") == "report"
    assert fake.registered_defn["tools"] == ["web-search", "akshare"]


def test_default_agent_has_no_tools_key(service_env: None) -> None:
    ''' 默认路径: 注册体不含 tools 键 (空列表省略 —— 服务端 canonical 剥默认值, 发空与不发同 id) '''
    fake = FakeAgentService()
    agent_cls = make_agent_cls(fake.transport)

    @ai_powered(model = "fake-model", agent = agent_cls)
    def plain(x: int) -> int:
        ''' Plain. '''
        ...

    assert plain(1) == 42
    assert "tools" not in fake.registered_defn
