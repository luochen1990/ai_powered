'''
职责: agent-service 动态 agent 的 HTTP 客户端 —— 注册 (POST /v1/agents) / 调用
(POST /v1/agents/{agent_id}/invoke), defn -> agent_id 内容寻址缓存, invoke 404 自愈,
以及 HTTP 结果 -> 本库异常体系的集中错误映射. (子包模块划分定位见 remote/__init__.py)

wire 契约 (服务端: agent-service 仓):
- 注册: body = AgentDefinition JSON dict (name/system_prompt/model/input_spec/output_spec,
  tools/definitions 可省); 200 -> {"agent_id": str}; 幂等 (服务端内容寻址, 同 defn 必同 id)
- 调用: body = 按注册 input_spec 校验的输入 JSON object; 200 -> 信封
  {"status": "completed"|"failed", "engine": str, "elapsed_ms": int, "result": 任意JSON|null, "error": str|null}
- 错误体 (4xx): {"error": {"code": str, "message": str}}; invoke 404 的 code=agent_not_found 特指
  注册表丢失 —— 服务端注册表为内存态, 重启即清, 是客户端自愈信号而非终态错误

错误映射表 (核心职责; 集中实现在 _map_register_response / _map_invoke_response 两纯函数, sync/async 共享):
| HTTP 结果                       | 行为                                       | 语义                 |
|---------------------------------|--------------------------------------------|----------------------|
| invoke 200 + status=completed   | 返回信封 result 字段值                      | —                    |
| invoke 200 + status=failed      | AgentRunError (信封 error 入消息)           | 瞬时失败, 可 retry   |
| invoke 404                      | invoke 门面先自愈 (见下); 仍 404 -> AgentRunError | 注册丢失, 可 retry |
| 422                             | RemoteContractError(ValueError)             | 契约违规, 不可 retry |
| 网络错误 / 超时 / 5xx             | AgentRunError                              | 瞬时, 可 retry       |
| 200 但信封形状违约               | RemoteContractError                        | 服务端违约, 不可 retry |

AgentRunError 复用 ai_powered.agent 的既有定义 (@ai_powered(retry=N) 的 RETRYABLE_EXCEPTIONS
已含它); RemoteContractError 继承 ValueError, 与本地 spec 校验错误同为编程错误, 天然落在可重试集合之外.

404 自愈机制: invoke 门面遇 _AgentNotFound 时丢弃内容寻址缓存 -> 重新注册 (幂等保证重建通常得回
同一 id, 开销仅一次往返) -> 重试一次 invoke; 仍 404 才抛 AgentRunError. invoke_agent 单发方法无 defn
可用, 不自愈, 404 直接转译 AgentRunError (需要自愈的调用方应走 invoke 门面).

双形态约束 (模式参照 llm/connection.py 的 LlmConnection): 上游 FunctionAgent.run 是
@sync_compatible async def, 其 sync 转写体要求被 await 的对象是 Waitable —— 故三方法均返回 Waitable:
构造期建 httpx sync/async 双客户端, async 实现经 sync_compatible(sync_fn=同步实现) 包装后存实例属性.

生命周期约定: 实例预期长生命周期 (每被装饰函数一个), 线程安全不做要求; transport 构造参数是测试注入
seam —— 需同时实现 sync/async handle 协议 (httpx.MockTransport 即满足), 单实例注入双客户端.
'''
import json
import urllib.parse
from typing import Any, Callable, Mapping, cast

import httpx
from easy_sync import Waitable, sync_compatible

from ..agent import AgentRunError

#: 注册端点路径 (相对 base_url)
_REGISTER_PATH = "/v1/agents"

#: 任务级超时的缺省值 (SSOT: RemoteAgent.invoke_timeout 与本类的构造缺省共用 —— 服务端沙箱任务典型 20-35s 量级)
DEFAULT_INVOKE_TIMEOUT = 120.0


class RemoteContractError(ValueError):
    ''' 远程契约违规 (不可 retry 的编程错误): 请求被服务端 422 拒绝, 或 200 响应信封形状违约 '''


class _AgentNotFound(AgentRunError):
    ''' 内部自愈信号: invoke 404. 仅在 invoke 门面内作 catch 用, 对外路径一律转译回基类, 不泄出 '''


def _invoke_path(agent_id: str) -> str:
    ''' agent_id 做路径段 quoting —— 当前 id 是服务端 uuid (安全), 防御未来 id 方案/中间网关引入特殊字符 '''
    return f"{_REGISTER_PATH}/{urllib.parse.quote(agent_id, safe = '')}/invoke"


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
    ''' 200 响应的 JSON 解析 (仅在已判 200 后调用): 非 JSON body 属服务端违约 '''
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


def _map_invoke_response(response: httpx.Response) -> Any:
    ''' invoke 响应 -> 信封 result 字段值 (纯函数, sync/async 双路径共享; 映射表见模块头) '''
    if response.status_code == 200:
        body = _json_body(response)
        if isinstance(body, dict):
            status = body.get("status")
            if status == "completed":
                if "result" not in body:  # "result": null 与键缺失区分 —— 后者是信封形状违约
                    raise RemoteContractError(f"invoke envelope has no result key: {body!r}")
                return body["result"]
            if status == "failed":
                raise AgentRunError(f"agent run failed: {body.get('error')}")
        raise RemoteContractError(f"malformed invoke envelope: {body!r}")
    if response.status_code == 404:
        raise _AgentNotFound(f"agent not found: {_error_detail(response)}")
    if response.status_code == 422:
        raise RemoteContractError(f"invoke rejected: {_error_detail(response)}")
    raise AgentRunError(f"agent-service invoke failed with HTTP {response.status_code}: {_error_detail(response)}")


class AgentServiceClient:
    ''' agent-service 的长生命周期客户端 (每被装饰函数一个实例):
    注册幂等缓存 + invoke 404 自愈门面; 三方法双形态 (Waitable), 详见模块头 '''

    sync_http: httpx.Client
    async_http: httpx.AsyncClient
    #: defn 内容寻址缓存 (cache key -> agent_id); invoke 404 自愈时作废
    _agent_ids: dict[str, str]
    #: 双形态实现载体 (构造期由 sync_compatible 包装生成, 见模块头 "双形态约束")
    _register_agent: Callable[[Mapping[str, Any]], Waitable[str]]
    _invoke_agent: Callable[[str, Mapping[str, Any]], Waitable[Any]]
    _invoke: Callable[[Mapping[str, Any], Mapping[str, Any]], Waitable[Any]]

    def __init__(self, base_url: str, *, timeout: float = DEFAULT_INVOKE_TIMEOUT, transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None) -> None:
        self.sync_http = httpx.Client(base_url = base_url, timeout = timeout, transport = cast(httpx.BaseTransport | None, transport))
        self.async_http = httpx.AsyncClient(base_url = base_url, timeout = timeout, transport = cast(httpx.AsyncBaseTransport | None, transport))
        self._agent_ids = {}

        async def async_register(defn: Mapping[str, Any]) -> str:
            return self._remember(defn, _map_register_response(await self._post_async(_REGISTER_PATH, defn)))

        def sync_register(defn: Mapping[str, Any]) -> str:
            return self._remember(defn, _map_register_response(self._post_sync(_REGISTER_PATH, defn)))

        async def async_invoke_agent(agent_id: str, input: Mapping[str, Any]) -> Any:
            try:
                return _map_invoke_response(await self._post_async(_invoke_path(agent_id), input))
            except _AgentNotFound as e:
                raise AgentRunError(str(e)) from e

        def sync_invoke_agent(agent_id: str, input: Mapping[str, Any]) -> Any:
            try:
                return _map_invoke_response(self._post_sync(_invoke_path(agent_id), input))
            except _AgentNotFound as e:
                raise AgentRunError(str(e)) from e

        async def async_invoke(defn: Mapping[str, Any], input: Mapping[str, Any]) -> Any:
            key = _cache_key(defn)
            agent_id = self._agent_ids.get(key)
            if agent_id is None:
                agent_id = await async_register(defn)
            try:
                return _map_invoke_response(await self._post_async(_invoke_path(agent_id), input))
            except _AgentNotFound:
                self._agent_ids.pop(key, None)  # 注册表丢失 => 陈旧缓存作废 (先弃后建, 重注册失败也不留脏项)
                agent_id = await async_register(defn)  # 幂等重注册
                try:
                    return _map_invoke_response(await self._post_async(_invoke_path(agent_id), input))
                except _AgentNotFound as second:
                    raise AgentRunError(f"agent still not found after re-registration: {second}") from second

        def sync_invoke(defn: Mapping[str, Any], input: Mapping[str, Any]) -> Any:
            key = _cache_key(defn)
            agent_id = self._agent_ids.get(key)
            if agent_id is None:
                agent_id = sync_register(defn)
            try:
                return _map_invoke_response(self._post_sync(_invoke_path(agent_id), input))
            except _AgentNotFound:
                self._agent_ids.pop(key, None)
                agent_id = sync_register(defn)
                try:
                    return _map_invoke_response(self._post_sync(_invoke_path(agent_id), input))
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
        ''' 单发调用已注册 agent —— 无 defn 可自愈, 404 直接 AgentRunError; 需自愈请走 invoke 门面 '''
        return self._invoke_agent(agent_id, input)

    def invoke(self, defn: Mapping[str, Any], input: Mapping[str, Any]) -> Waitable[Any]:
        ''' 组合门面 (上游 RemoteAgent.run 的主入口): 缓存命中免注册往返, miss 则注册;
        invoke 404 自愈 (弃缓存 -> 重注册 -> 重试一次); 返回信封 result 字段值 '''
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
            raise AgentRunError(f"agent-service request failed ({path}): {e}") from e

    def _post_sync(self, path: str, body: Mapping[str, Any]) -> httpx.Response:
        try:
            return self.sync_http.post(path, json = dict(body))
        except httpx.HTTPError as e:
            raise AgentRunError(f"agent-service request failed ({path}): {e}") from e
