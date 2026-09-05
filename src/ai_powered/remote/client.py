'''
职责: agent-service 动态 agent 的 HTTP 客户端 —— 注册 (POST /v1/agents) / 提交
(POST /v1/agents/{agent_id}/invoke) / 任务轮询 (GET /v1/tasks/{task_id}),
defn -> agent_id 内容寻址缓存, invoke 404 自愈, 以及 HTTP 结果 -> 本库异常体系的
集中错误映射. (子包模块划分定位见 remote/__init__.py)

wire 契约 (服务端: agent-service 仓, 异步化契约见其 issue #30 spec §10):
- 注册 (跨契约切换不变): body = AgentDefinition JSON dict (name/system_prompt/model/
  input_spec/output_spec, tools/definitions 可省); 200 -> {"agent_id": str}; 幂等
  (服务端内容寻址, 同 defn 必同 id)
- 提交: body = 按注册 input_spec 校验的输入 JSON object (请求体形态跨契约切换不变,
  响应码即版本判别器; 服务端规划中的 {input, priority?, budget_tokens?} 信封落地时
  再行同步); 新契约 202 -> {"task_id": str, "status": "queued"}; 旧契约 (过渡兼容,
  切换窗口期的未升级服务端) 200 -> 同步信封, 直接解析结果
- 任务查询: GET /v1/tasks/{task_id} -> {"task_id", "status", "rounds", "tokens_used"}
  - 非终态 status: queued | running | paused (rounds/tokens_used 为观测字段, 客户端不消费)
  - 终态 done: 附 result (输出 spec 规范 JSON) / engine / elapsed_ms / usage
  - 终态 failed: 附 error / engine / elapsed_ms / upstream_code? / next_flush_time?
  - 终态 cancelled: 附 error
- 错误体 (4xx): {"error": {"code": str, "message": str}}; invoke 404 的
  code=agent_not_found 特指注册表丢失 —— 服务端注册表为内存态, 重启即清, 是客户端
  自愈信号而非终态错误

错误映射表 (核心职责; 集中实现在 _map_register_response / _map_invoke_response /
_map_task_response / _map_task_body 四纯函数, sync/async 共享):
| HTTP 结果                             | 行为                                                 | 语义                 |
| invoke 202 + task_id                  | 转入轮询循环                                          | —                    |
| 轮询至终态 done                        | 返回 result 字段值                                    | —                    |
| 轮询至终态 failed                      | RemoteTaskFailed (携带 upstream_code/next_flush_time) | 瞬时失败, 可 retry   |
| 轮询至终态 cancelled                   | RemoteTaskCancelled (独立类型)                        | 显式取消, 不可 retry |
| 轮询预算耗尽仍非终态                    | RemoteTaskTimeout                                    | 瞬时, 可 retry       |
| invoke 200 + status=completed (旧信封) | 兼容分支: 直接返回信封 result 字段值                   | —                    |
| invoke 200 + status=failed (旧信封)    | RemoteTaskFailed (字段从旧信封同位提取)                | 瞬时失败, 可 retry   |
| invoke 404                             | invoke 门面先自愈 (见下); 仍 404 -> AgentRunError      | 注册丢失, 可 retry   |
| 422                                   | RemoteContractError(ValueError)                       | 契约违规, 不可 retry |
| 任务查询 404 / 5xx / 快速网络失败        | AgentRunError                                         | 瞬时, 可 retry       |
| 任务查询失败且已耗尽预算 (钳制超时)       | RemoteTaskTimeout                                    | 瞬时, 可 retry       |
| 200/202/任务体 形状违约                 | RemoteContractError                                  | 服务端违约, 不可 retry |

异常族 (与既有体系的关系): AgentRunError 复用 ai_powered.agent 的定义 ——
RemoteTaskFailed/RemoteTaskTimeout 是其子类, 天然进入 @ai_powered(retry=N) 的
RETRYABLE_EXCEPTIONS (failed 的可重试语义延续旧同步信封时代); RemoteTaskCancelled
特意不挂 AgentRunError 族 —— 取消是调用方/运维的显式决策, 自动重试会静默重提交一个
新任务, 让它落在可重试集合之外直达调用方; upstream_code/next_flush_time 的字段名
沿用 agent-service 的 upstream_error.py (上游 LLM 网关结构化错误码/配额恢复时间,
宽容提取, 无则 None —— 信息字段不为硬失败); RemoteContractError 继承 ValueError,
与本地 spec 校验错误同为编程错误.

404 自愈机制: invoke 门面遇 _AgentNotFound 时丢弃内容寻址缓存 -> 重新注册 (幂等
保证重建通常得回同一 id, 开销仅一次往返) -> 重试一次 invoke; 仍 404 才抛
AgentRunError. invoke_agent 单发方法无 defn 可用, 不自愈, 404 直接转译
AgentRunError (需要自愈的调用方应走 invoke 门面). 任务查询 404 不在自愈之列 ——
提交已受理过, 任务记录缺失属另一种故障 (直接 AgentRunError, 重试属调用方决策).

轮询节律 (设计决策: 固定间隔, 默认 0.5s, 不做指数退避): 任务查询是服务端 SQLite
主键点查 O(1), 内网单在跑任务 ~2 QPS 的负载可忽略; 指数退避会以结果交付延迟 (最坏
一个间隔) 换取不必要的请求节省 —— 典型任务 20-35s 量级, 固定 0.5s 把轮询引入的
结果延迟上界钉在 0.5s. 预算: 提交+轮询共享构造参数 timeout (锚点在提交前, 注册往返
不占预算 —— 对齐旧契约下每请求独立预算的语义); 每次任务查询的单请求超时钳制在剩余
预算内 (预算是硬上界, 不被单次挂起请求突破), 耗尽预算才失败的网络错误转译为
RemoteTaskTimeout (预算内快失败仍为 AgentRunError —— 供调用方按类型区分
"预算太小" 与 "网络抖动").

双形态约束 (模式参照 llm/connection.py 的 LlmConnection): 上游 FunctionAgent.run 是
@sync_compatible async def, 其 sync 转写体要求被 await 的对象是 Waitable —— 故三方法均返回 Waitable:
构造期建 httpx sync/async 双客户端, async 实现经 sync_compatible(sync_fn=同步实现) 包装后存实例属性.
提交+轮询序列在 sync/async 两侧各一实现 (轮询 sleep 分别为 time.sleep / asyncio.sleep,
是唯一差异), 全部 wire 解析集中在纯函数共享.

生命周期约定: 实例预期长生命周期 (每被装饰函数一个), 线程安全不做要求; transport 构造参数是测试注入
seam —— 需同时实现 sync/async handle 协议 (httpx.MockTransport 即满足), 单实例注入双客户端.
'''
import asyncio
import json
import time
import urllib.parse
from typing import Any, Callable, Mapping, cast

import httpx
from easy_sync import Waitable, sync_compatible

from ..agent import AgentRunError

#: 注册端点路径 (相对 base_url)
_REGISTER_PATH = "/v1/agents"

#: 任务查询端点路径前缀 (相对 base_url)
_TASKS_PATH = "/v1/tasks"

#: 提交+轮询总预算的缺省值 (SSOT: RemoteAgent.invoke_timeout 与本类的构造缺省共用 —— 服务端沙箱任务典型 20-35s 量级)
DEFAULT_INVOKE_TIMEOUT = 120.0

#: 轮询间隔缺省值 (固定间隔, 决策理由见模块头 "轮询节律")
DEFAULT_POLL_INTERVAL = 0.5

#: 任务状态体的非终态值域 (终态 done/failed/cancelled 的分发见 _map_task_body)
_NON_TERMINAL_STATUSES = frozenset({"queued", "running", "paused"})


class RemoteContractError(ValueError):
    ''' 远程契约违规 (不可 retry 的编程错误): 请求被服务端 422 拒绝, 或 2xx 响应信封形状违约 '''


class RemoteTaskFailed(AgentRunError):
    ''' 任务 failed 终态 (瞬时失败, 可 retry —— 语义延续旧同步信封的 failed 分支):
    upstream_code/next_flush_time 为上游 LLM 网关的结构化事实 (字段名沿用
    agent-service 的 upstream_error.py), 无则 None '''

    upstream_code: str | None
    next_flush_time: str | None

    def __init__(self, message: str, *, upstream_code: str | None = None, next_flush_time: str | None = None) -> None:
        super().__init__(message)
        self.upstream_code = upstream_code
        self.next_flush_time = next_flush_time


class RemoteTaskCancelled(Exception):
    ''' 任务 cancelled 终态 (不可 retry —— 特意不挂 AgentRunError 族, 且与 RuntimeError
    解耦: except RuntimeError 不应同时捕住语义相反的 "可重试失败" 与 "显式取消"):
    取消是调用方/运维的显式决策, 自动重试会静默重提交一个新任务, 异常应直达调用方 '''


class RemoteTaskTimeout(AgentRunError):
    ''' 提交+轮询总预算耗尽仍未见终态 (可 retry): 注意任务可能仍在服务端运行
    (孤儿任务), retry 提交的是新任务而非续查旧任务 '''


class _AgentNotFound(AgentRunError):
    ''' 内部自愈信号: invoke 404. 仅在 invoke 门面内作 catch 用, 对外路径一律转译回基类, 不泄出 '''


def _invoke_path(agent_id: str) -> str:
    ''' agent_id 做路径段 quoting —— 当前 id 是服务端 uuid (安全), 防御未来 id 方案/中间网关引入特殊字符 '''
    return f"{_REGISTER_PATH}/{urllib.parse.quote(agent_id, safe = '')}/invoke"


def _task_path(task_id: str) -> str:
    ''' task_id 做路径段 quoting (同 _invoke_path 的防御性惯例) '''
    return f"{_TASKS_PATH}/{urllib.parse.quote(task_id, safe = '')}"


def _cache_key(defn: Mapping[str, Any]) -> str:
    ''' defn 的内容寻址缓存 key —— 客户端不需要实现服务端 canonical 算法: 缓存只省一次注册往返,
    正确性由注册幂等 (内容寻址) 兜底, key 只需进程内稳定且与 defn 一一对应 '''
    return json.dumps(defn, sort_keys = True, ensure_ascii = False)


def _error_detail(response: httpx.Response) -> str:
    ''' 集中式错误信封解析: {"error": {"code": str, "message": str}} -> "code: message" 人读串;
    body 非 JSON 或形状不符时降级为原文片段, 服务端违约仍可诊断 '''
    try:
        body: Any = response.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        code, message = error.get("code"), error.get("message")
        if isinstance(code, str) and isinstance(message, str):
            return f"{code}: {message}"
    return repr(response.text[:200])


def _json_body(response: httpx.Response) -> Any:
    ''' 2xx 响应的 JSON 解析 (仅在已判 2xx 后调用): 非 JSON body 属服务端违约 '''
    try:
        return response.json()
    except ValueError as e:
        raise RemoteContractError(f"non-JSON body in HTTP {response.status_code} response: {response.text[:200]!r}") from e


def _map_register_response(response: httpx.Response) -> str:
    ''' 注册响应 -> agent_id (纯函数, sync/async 双路径共享; 映射表见模块头) '''
    if response.status_code == 200:
        body = _json_body(response)
        agent_id = body.get("agent_id") if isinstance(body, dict) else None
        if isinstance(agent_id, str):
            return agent_id
        raise RemoteContractError(f"register response envelope has no agent_id: {body!r}")
    if response.status_code == 422:
        raise RemoteContractError(f"agent registration rejected: {_error_detail(response)}")
    raise AgentRunError(f"agent-service register failed with HTTP {response.status_code}: {_error_detail(response)}")


def _optional_str(body: dict[str, Any], key: str) -> str | None:
    ''' 信息性字段 (upstream_code/next_flush_time) 的宽容提取: str 则取, 缺席/异型则 None
    (与服务端 best-effort 透传的态度一致 —— 信息字段不为硬失败) '''
    value = body.get(key)
    return value if isinstance(value, str) else None


def _error_text(body: dict[str, Any]) -> str:
    ''' error 字段的宽容提取 (failed/cancelled 终态共用同一口径): str 则取,
    缺席/异型则整个 body repr (人读兜底 —— 服务端契约 error 恒为 str, 此为防御) '''
    error = body.get("error")
    return error if isinstance(error, str) else repr(body)


def _task_failed_from(body: dict[str, Any]) -> RemoteTaskFailed:
    ''' failed 终态体 (新旧信封字段同构) -> RemoteTaskFailed 的纯构造 (轮询终态与旧信封兼容分支共用) '''
    return RemoteTaskFailed(
        f"agent run failed: {_error_text(body)}",
        upstream_code = _optional_str(body, "upstream_code"),
        next_flush_time = _optional_str(body, "next_flush_time"),
    )


def _map_invoke_response(response: httpx.Response) -> tuple[bool, Any]:
    ''' invoke 响应 -> (已提交异步任务?, 载荷) (纯函数, sync/async 双路径共享; 映射表见模块头):
    202 -> (True, task_id); 200 旧信封 completed -> (False, result 字段值) —— 响应码即版本判别器 '''
    if response.status_code == 202:
        body = _json_body(response)
        task_id = body.get("task_id") if isinstance(body, dict) else None
        if isinstance(task_id, str) and task_id:
            # status 字段不校验: 202 即提交受理, 任务状态以查询端点为准 (SSOT)
            return True, task_id
        raise RemoteContractError(f"submit response envelope has no task_id: {body!r}")
    if response.status_code == 200:  # 过渡兼容: 未升级服务端的旧同步信封
        body = _json_body(response)
        if isinstance(body, dict):
            status = body.get("status")
            if status == "completed":
                if "result" not in body:  # "result": null 与键缺失区分 —— 后者是信封形状违约
                    raise RemoteContractError(f"invoke envelope has no result key: {body!r}")
                return False, body["result"]
            if status == "failed":
                raise _task_failed_from(body)
        raise RemoteContractError(f"malformed invoke envelope: {body!r}")
    if response.status_code == 404:
        raise _AgentNotFound(f"agent not found: {_error_detail(response)}")
    if response.status_code == 422:
        raise RemoteContractError(f"invoke rejected: {_error_detail(response)}")
    raise AgentRunError(f"agent-service invoke failed with HTTP {response.status_code}: {_error_detail(response)}")


def _map_task_response(response: httpx.Response) -> Any:
    ''' 任务查询响应 -> 状态体 JSON (纯函数): 200 -> body; 非 200 (含 404 任务记录缺失)
    -> AgentRunError (瞬时口径 —— 服务端重启窗口/网关抖动, 重试属调用方决策) '''
    if response.status_code == 200:
        return _json_body(response)
    raise AgentRunError(f"agent-service task query failed with HTTP {response.status_code}: {_error_detail(response)}")


def _poll_timeout_error(task_id: str, budget: float) -> RemoteTaskTimeout:
    ''' 预算耗尽异常构造 (循环头检查与钳制超时转译共用同一消息口径) '''
    return RemoteTaskTimeout(f"task {task_id} did not reach a terminal state within {budget}s poll budget (it may still be running server-side)")


def _map_task_body(body: Any) -> tuple[bool, Any]:
    ''' 任务状态体 -> (终态?, 载荷) (纯函数, sync/async 共享): 非终态 -> (False, status);
    done -> (True, result); failed/cancelled 在此抛对应异常; 形状违约 (非 dict / 无
    status / 值域外 / done 缺 result 键) -> RemoteContractError '''
    if not isinstance(body, dict):
        raise RemoteContractError(f"malformed task envelope: {body!r}")
    status = body.get("status")
    if status == "done":
        if "result" not in body:  # "result": null 与键缺失区分 —— 后者是信封形状违约
            raise RemoteContractError(f"done task envelope has no result key: {body!r}")
        return True, body["result"]
    if status == "failed":
        raise _task_failed_from(body)
    if status == "cancelled":
        raise RemoteTaskCancelled(f"agent task cancelled: {_error_text(body)}")
    if status in _NON_TERMINAL_STATUSES:
        return False, status
    raise RemoteContractError(f"unknown task status in envelope: {body!r}")


def _request_failed(path: str, e: Exception) -> AgentRunError:
    ''' 网络层失败的消息构造 SSOT (四个 requester 的 except 体共用). NOTE: 只共享消息,
    requester 本体不合并 —— httpx 的显式 timeout 传参与缺省 (USE_CLIENT_DEFAULT)
    语义不同, 统一传参会静默改变 _post_* 的预算行为 '''
    return AgentRunError(f"agent-service request failed ({path}): {e}")


class AgentServiceClient:
    ''' agent-service 的长生命周期客户端 (每被装饰函数一个实例): 注册幂等缓存 +
    提交-轮询门面 (202 -> GET /v1/tasks/{task_id} 轮询至终态; 未升级服务端的 200
    旧信封直接取结果) + invoke 404 自愈; 三方法双形态 (Waitable), 详见模块头 '''

    sync_http: httpx.Client
    async_http: httpx.AsyncClient
    #: defn 内容寻址缓存 (cache key -> agent_id); invoke 404 自愈时作废
    _agent_ids: dict[str, str]
    #: 双形态实现载体 (构造期由 sync_compatible 包装生成, 见模块头 "双形态约束")
    _register_agent: Callable[[Mapping[str, Any]], Waitable[str]]
    _invoke_agent: Callable[[str, Mapping[str, Any]], Waitable[Any]]
    _invoke: Callable[[Mapping[str, Any], Mapping[str, Any]], Waitable[Any]]

    def __init__(self, base_url: str, *, timeout: float = DEFAULT_INVOKE_TIMEOUT, poll_interval: float = DEFAULT_POLL_INTERVAL, transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None) -> None:
        ''' timeout: 提交+轮询总预算 (兼容路径下即单次 invoke 请求预算, 两语义在旧服务端上重合);
        poll_interval: 任务轮询间隔 (固定间隔, 决策见模块头), 须为正数 (构造期 fail-fast) '''
        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be positive, got {poll_interval!r}")
        self.sync_http = httpx.Client(base_url = base_url, timeout = timeout, transport = cast(httpx.BaseTransport | None, transport))
        self.async_http = httpx.AsyncClient(base_url = base_url, timeout = timeout, transport = cast(httpx.AsyncBaseTransport | None, transport))
        self._agent_ids = {}

        async def async_register(defn: Mapping[str, Any]) -> str:
            return self._remember(defn, _map_register_response(await self._post_async(_REGISTER_PATH, defn)))

        def sync_register(defn: Mapping[str, Any]) -> str:
            return self._remember(defn, _map_register_response(self._post_sync(_REGISTER_PATH, defn)))

        async def async_poll_terminal(task_id: str, deadline: float) -> Any:
            ''' 轮询至终态 (async 侧): 首查立即发出 (快速任务零等待), 之后按固定间隔;
            sleep 时长钳制在剩余预算内 (不睡过界); 单请求超时 = 剩余预算 '''
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _poll_timeout_error(task_id, timeout)
                try:
                    response = await self._get_async(_task_path(task_id), timeout = remaining)
                except AgentRunError as e:
                    # 单请求超时已被钳制到剩余预算 —— 耗尽预算才失败的网络错误转译为语义精确的
                    # RemoteTaskTimeout (预算尾段服务端响应变慢的窗口并不罕见); 预算未尽的快失败原样放行
                    if time.monotonic() >= deadline:
                        raise _poll_timeout_error(task_id, timeout) from e
                    raise
                terminal, payload = _map_task_body(_map_task_response(response))
                if terminal:
                    return payload
                await asyncio.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

        def sync_poll_terminal(task_id: str, deadline: float) -> Any:
            ''' 轮询至终态 (sync 侧, 与 async_poll_terminal 同构 —— sleep 形态是唯一差异) '''
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _poll_timeout_error(task_id, timeout)
                try:
                    response = self._get_sync(_task_path(task_id), timeout = remaining)
                except AgentRunError as e:
                    if time.monotonic() >= deadline:
                        raise _poll_timeout_error(task_id, timeout) from e
                    raise
                terminal, payload = _map_task_body(_map_task_response(response))
                if terminal:
                    return payload
                time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

        async def async_run_invocation(agent_id: str, input: Mapping[str, Any]) -> Any:
            ''' 提交 + 按响应版本分流 (202 -> 轮询至终态; 200 旧信封 -> 直取结果) —— 两门面的共用序列 '''
            deadline = time.monotonic() + timeout  # 预算锚点: 提交前 (注册往返不占预算; heal 重试重新起算)
            submitted, payload = _map_invoke_response(await self._post_async(_invoke_path(agent_id), input))
            return await async_poll_terminal(payload, deadline) if submitted else payload

        def sync_run_invocation(agent_id: str, input: Mapping[str, Any]) -> Any:
            deadline = time.monotonic() + timeout
            submitted, payload = _map_invoke_response(self._post_sync(_invoke_path(agent_id), input))
            return sync_poll_terminal(payload, deadline) if submitted else payload

        async def async_invoke_agent(agent_id: str, input: Mapping[str, Any]) -> Any:
            try:
                return await async_run_invocation(agent_id, input)
            except _AgentNotFound as e:
                raise AgentRunError(str(e)) from e

        def sync_invoke_agent(agent_id: str, input: Mapping[str, Any]) -> Any:
            try:
                return sync_run_invocation(agent_id, input)
            except _AgentNotFound as e:
                raise AgentRunError(str(e)) from e

        async def async_invoke(defn: Mapping[str, Any], input: Mapping[str, Any]) -> Any:
            key = _cache_key(defn)
            agent_id = self._agent_ids.get(key)
            if agent_id is None:
                agent_id = await async_register(defn)
            try:
                return await async_run_invocation(agent_id, input)
            except _AgentNotFound:
                self._agent_ids.pop(key, None)  # 注册表丢失 => 陈旧缓存作废 (先弃后建, 重注册失败也不留脏项)
                agent_id = await async_register(defn)  # 幂等重注册
                try:
                    return await async_run_invocation(agent_id, input)
                except _AgentNotFound as second:
                    raise AgentRunError(f"agent still not found after re-registration: {second}") from second

        def sync_invoke(defn: Mapping[str, Any], input: Mapping[str, Any]) -> Any:
            key = _cache_key(defn)
            agent_id = self._agent_ids.get(key)
            if agent_id is None:
                agent_id = sync_register(defn)
            try:
                return sync_run_invocation(agent_id, input)
            except _AgentNotFound:
                self._agent_ids.pop(key, None)
                agent_id = sync_register(defn)
                try:
                    return sync_run_invocation(agent_id, input)
                except _AgentNotFound as second:
                    raise AgentRunError(f"agent still not found after re-registration: {second}") from second

        self._register_agent = sync_compatible(sync_fn = sync_register)(async_register)
        self._invoke_agent = sync_compatible(sync_fn = sync_invoke_agent)(async_invoke_agent)
        self._invoke = sync_compatible(sync_fn = sync_invoke)(async_invoke)

        async def async_close() -> None:
            self.sync_http.close()
            await self.async_http.aclose()

        def sync_close() -> None:
            # 只关 sync 侧: async client 的 aclose 需事件循环, 且 sync 转写体使用它时其自身无连接 (惰性建立)
            self.sync_http.close()

        self._close = sync_compatible(sync_fn = sync_close)(async_close)

    def close(self) -> Waitable[None]:
        ''' 显式资源回收 (双形态): async 路径关双客户端, sync 路径关 sync 客户端.

        生命周期约定: 与 LlmConnection 同款现状 —— 常规用法下实例随进程存活 (每被装饰函数一个),
        不强制显式关闭; 需要确定性回收的场景 (如测试套件开 -W error 防 ResourceWarning) 才调用.
        '''
        return self._close()

    def register_agent(self, defn: Mapping[str, Any]) -> Waitable[str]:
        ''' 注册 (或幂等重注册) 一个动态 agent, 返回其 agent_id '''
        return self._register_agent(defn)

    def invoke_agent(self, agent_id: str, input: Mapping[str, Any]) -> Waitable[Any]:
        ''' 单发调用已注册 agent (提交+轮询至终态) —— 无 defn 可自愈, 404 直接
        AgentRunError; 需自愈请走 invoke 门面 '''
        return self._invoke_agent(agent_id, input)

    def invoke(self, defn: Mapping[str, Any], input: Mapping[str, Any]) -> Waitable[Any]:
        ''' 组合门面 (上游 RemoteAgent.run 的主入口): 缓存命中免注册往返, miss 则注册;
        提交 (202) + 轮询至终态返回 result 字段值 (未升级服务端的 200 旧信封直接取结果);
        invoke 404 自愈 (弃缓存 -> 重注册 -> 重试一次) '''
        return self._invoke(defn, input)

    def _remember(self, defn: Mapping[str, Any], agent_id: str) -> str:
        ''' 注册成功的公共后置: 写入内容寻址缓存 (register_agent 公共路径与 invoke 门面的 miss/heal 路径共用) '''
        self._agent_ids[_cache_key(defn)] = agent_id
        return agent_id

    async def _post_async(self, path: str, body: Mapping[str, Any]) -> httpx.Response:
        ''' 网络层异常 (连接/超时等 httpx.HTTPError 族) 集中映射为 AgentRunError (瞬时, 可 retry);
        dict(body) 为集中预处理 —— 映射逻辑此后只面对 JSON-ready dict '''
        try:
            return await self.async_http.post(path, json = dict(body))
        except httpx.HTTPError as e:
            raise _request_failed(path, e) from e

    def _post_sync(self, path: str, body: Mapping[str, Any]) -> httpx.Response:
        try:
            return self.sync_http.post(path, json = dict(body))
        except httpx.HTTPError as e:
            raise _request_failed(path, e) from e

    async def _get_async(self, path: str, *, timeout: float) -> httpx.Response:
        ''' 任务查询的 GET 形态 (网络层异常映射同 _post_*; timeout 为单请求钳制值 —— 调用方传入剩余预算) '''
        try:
            return await self.async_http.get(path, timeout = timeout)
        except httpx.HTTPError as e:
            raise _request_failed(path, e) from e

    def _get_sync(self, path: str, *, timeout: float) -> httpx.Response:
        try:
            return self.sync_http.get(path, timeout = timeout)
        except httpx.HTTPError as e:
            raise _request_failed(path, e) from e
