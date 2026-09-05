'''
职责: RemoteAgent —— agent-service 的远程沙箱执行引擎 (FunctionAgent 子类).

定位 (融合方案 2026-09): @ai_powered(agent = RemoteAgent) 把被装饰函数编译为服务端动态
agent —— 签名/docstring 进 system_prompt, 参数/返回类型经 type_desc.compile_type 编译为
TypeDesc ADT spec, LLM 调用与沙箱执行全部发生在服务端 (凭证服务端管控, 客户端零 key).

协议对齐 (与本地引擎的关键对应关系):
- 出口: output_spec 恒为 product{result: R} —— 服务端 invoke 信封的 result 字段即
  {"result": <R>} 形态的规范 JSON, run 原样序列化返回, 装饰器层的 Result[R] 校验管线
  与 retry 机制零改动复用 (与 ReasoningAgent/GenericAgent 的 return_result 工具殊途同归)
- 入口: input_spec 为 product{参数名: 参数 TypeDesc}, invoke body 即装饰器层打包的
  arguments_json 解码后的 dict (两侧均以参数名为 key, 无变换)
- required_features 为空: 模型通道特性 (tools/schema_refs 等) 由服务端引擎操心,
  客户端对本地解析出的 model 通道无要求 —— 仅取 model_name 作注册的模型名

生命周期: 注册惰性且幂等 —— 装饰期只做本地编译 (类型 fail-fast, 零网络 IO),
首次调用经 AgentServiceClient.invoke 完成注册; 服务端注册表内存态重启即清,
404 自愈 (弃缓存重注册) 由 client 门面内置, 见 client.py 头注释.

调用契约 (agent-service issue #30 契约异步化): invoke 经 202+task_id 提交,
client 内部轮询 GET /v1/tasks/{task_id} 至终态 —— 对装饰器/调用方完全透明
(sync/async 双外观不变); 终态 failed/cancelled/预算耗尽分别经
RemoteTaskFailed/RemoteTaskCancelled/RemoteTaskTimeout 透出 (语义见 client.py).

测试注入 seam: _create_client 工厂 (对齐 GenericAgent._create_sandbox 惯例) ——
子类覆盖以注入 httpx.MockTransport, 离线驱动全链路.
'''
import json
import os
from typing import Any, ClassVar

from easy_sync import sync_compatible

from ..agent import FunctionAgent
from ..constants import SYSTEM_PROMPT_RETURN_SCHEMA, base_system_prompt
from ..llm.connection import LlmConnection
from ..llm.definitions import FunctionSpec, ModelFeature
from ..llm.known_models import ModelConfig
from .client import DEFAULT_INVOKE_TIMEOUT, DEFAULT_POLL_INTERVAL, AgentServiceClient
from .type_desc import RESERVED_NAME_REASON, compile_type, field_type, is_reserved_field_name

#: 服务端地址的环境变量 (解析优先级: 类属性 service_url > 此环境变量)
SERVICE_URL_ENV = "AI_POWERED_AGENT_SERVICE_URL"

#: 远端沙箱使用指引 (服务端六件套工具面: Bash/Read/Write/Edit/Grep/Glob;
#: 与本地 GenericAgent 的 _GENERIC_AGENT_INSTRUCTION 精神一致, 文案按远端工具面改写 ——
#: 提交机制不同: 本地经 return_result 工具, 远端经服务端结构化输出机制, 无需模型显式提交)
_SANDBOX_INSTRUCTION = (
    "\nYou are working autonomously in a server-side sandbox: an isolated workspace with shell and file tools "
    "(Bash / Read / Write / Edit / Grep / Glob), where files persist across your tool calls. "
    "Get things done with these tools as needed — read the errors you encounter, fix and retry. "
    "Your final result is collected through the structured output mechanism when you are done.\n"
)


class RemoteAgent(FunctionAgent):
    ''' 远程沙箱执行引擎: 函数签名 -> 服务端动态 agent, 能力对等本地 GenericAgent 的服务端版

    两种用法 (与本地 GenericAgent 的双用法协议对偶 —— agent 参数接受子类, 装饰期构造一次实例):
    - 直接使用 (@ai_powered(model = "模型名", agent = RemoteAgent)): 端点读环境变量
      AI_POWERED_AGENT_SERVICE_URL (服务端管控凭证, 客户端零 key); 未指定 model 参数时
      注册模型名回落装饰器的解析结果 (平台默认兜底, 见 resolver.py)
    - 继承覆盖 ClassVar 定制 (推荐形态):
        class MyAgent(RemoteAgent):
            service_url = "https://agents.internal:8300"   # 固定端点
            default_model = "glm-5.3"                      # 固定注册模型 (优先于装饰器解析)
            invoke_timeout = 300.0                         # 任务级超时 = 提交+轮询总预算 (默认 120s —— 服务端沙箱任务典型 20-35s 量级)
            poll_interval = 1.0                            # 任务轮询间隔 (默认 0.5s, 固定间隔的决策见 client.py 头注释)
            tools = ["web-search", "akshare"]              # 宿主侧 MCP 工具白名单引用
      tools 引用名须在服务端 AGENT_SERVICE_TOOL_CATALOG 目录内 (白名单服务端管控, 注册方只引用名);
      未知名在首次调用的注册期报 RemoteContractError (不可 retry) —— 客户端无法本地校验白名单,
      属已知的 fail-fast 边界. 默认 [] = 仅服务端沙箱六件套 (Bash/Read/Write/Edit/Grep/Glob).
      priority/budget_tokens: 调度参数透传 (M1a 提交信封, 见 client.py 模块头)
      更深度的定制可覆盖 _build_definition 整体接管 defn 组装 (开放出口, 对齐 _create_client 的 seam 惯例).

    装饰期 fail-fast (零网络): 类型不被远程通道支持 (dict/裸 Union/递归等, 见
    type_desc.py 拒绝清单) / 端点缺失, 均抛 ValueError 族.
    '''

    service_url: ClassVar[str | None] = None
    default_model: ClassVar[str | None] = None
    invoke_timeout: ClassVar[float] = DEFAULT_INVOKE_TIMEOUT
    poll_interval: ClassVar[float] = DEFAULT_POLL_INTERVAL
    tools: ClassVar[list[str]] = []
    required_features: ClassVar[set[ModelFeature]] = set()
    #: 调度参数 (M1a 信封透传, agent-service issue #30 spec §10): priority 大者优先,
    #: budget_tokens 为任务 token 预算 (服务端预授+结算不可超); agent 级常量 (同一
    #: agent 族的全部调用同参), 按调用差异化属调用方自行组 client 的范畴
    priority: ClassVar[int] = 0
    budget_tokens: ClassVar[int | None] = None

    _defn: dict[str, Any]
    _client: AgentServiceClient

    def __init__(self, spec: FunctionSpec, model_config: ModelConfig, connection: LlmConnection) -> None:
        super().__init__(spec, model_config, connection)
        url = type(self).service_url or os.environ.get(SERVICE_URL_ENV)
        if not url:
            raise ValueError(
                f"{type(self).__name__} requires an agent-service endpoint: "
                f"set {SERVICE_URL_ENV} or override the service_url class attribute"
            )
        model = type(self).default_model or model_config.model_name
        #: 解析优先级: 类属性 default_model (继承定制, 固定该 agent 族的注册模型) > 装饰器解析的
        #: model_config.model_name (model 参数 / 平台默认兜底 —— resolver 恒给出非空名, 无缺失分支)
        self._defn = self._build_definition(model)
        self._client = self._create_client(url)

    def _create_client(self, url: str) -> AgentServiceClient:
        ''' client 工厂 (独立方法: 测试子类可覆盖以注入 transport, 对齐 GenericAgent._create_sandbox 惯例) '''
        return AgentServiceClient(url, timeout = type(self).invoke_timeout, poll_interval = type(self).poll_interval)

    def _build_definition(self, model: str) -> dict[str, Any]:
        ''' FunctionSpec -> AgentDefinition dict (装饰期纯本地编译, 网络交互延迟到首次调用)

        类型不支持与形参名不合法在此 fail-fast (UnsupportedTypeError, ValueError 族).
        input/output spec 的缺省字段语境化: 参数带 default -> maybe (调用方省略时 bind 不含该参数,
        服务端输入校验对 maybe 允许省略); 返回类型缺省字段必选 (服务端 dump 恒全字段, maybe 的 null
        会击穿客户端 Result[R] 解码 —— 见 type_desc.py 模块头 "缺省字段的双语境编译").
        '''
        spec = self.spec
        if spec.parameter_annotations is None or spec.return_annotation is None:
            # 防御: 装饰器路径恒填充注解 (decorators.py); 手工构造 spec 的调用方需自查
            raise ValueError(f"{type(self).__name__} requires FunctionSpec with raw annotations (parameter_annotations / return_annotation)")
        defaults = spec.parameter_defaults or {}
        input_fields: dict[str, Any] = {}
        for name, ann in spec.parameter_annotations.items():
            if is_reserved_field_name(name):
                raise ValueError(f"{type(self).__name__}: 形参名 {name!r} {RESERVED_NAME_REASON}")
            desc = compile_type(ann, defaults_omissible = True)
            input_fields[name] = field_type(desc, omissible = name in defaults)
        defn: dict[str, Any] = {
            "name": spec.function_name,
            # NOTE (review ②权衡记录): system_prompt 内嵌的 return_schema (msgspec 输出) 对缺省字段不列
            # required, 与 output_spec 的必选编译 (H-1 决策) 语义相消 —— LLM 可能倾向省略缺省字段, 由服务端
            # structured 校验回喂环兜底 (最坏多一轮往返); 后续可统一为 TypeDesc 侧语义, MVP 接受该张力
            "system_prompt": base_system_prompt(spec) + SYSTEM_PROMPT_RETURN_SCHEMA.format(return_schema = json.dumps(spec.return_schema)) + _SANDBOX_INSTRUCTION,
            "model": model,
            "input_spec": {"kind": "product", "fields": input_fields},
            "output_spec": {"kind": "product", "fields": {"result": {"type": compile_type(spec.return_annotation)}}},
        }
        if type(self).tools:  # 空列表省略 —— 服务端 canonical 化剥默认值, 发空与不发同 id, 不发更净
            defn["tools"] = list(type(self).tools)
        return defn

    @sync_compatible
    async def run(self, arguments_json: str) -> str:
        input_obj = json.loads(arguments_json)
        if not isinstance(input_obj, dict):  # 契约外输入防御: 装饰器层 encode 的恒是 Bind.arguments (dict)
            raise ValueError(f"RemoteAgent.run expects a JSON object, got: {arguments_json[:100]!r}")
        result = await self._client.invoke(
            self._defn, input_obj,
            priority = type(self).priority, budget_tokens = type(self).budget_tokens,
        )
        return json.dumps(result, ensure_ascii = False)
