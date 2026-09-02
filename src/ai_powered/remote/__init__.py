'''
职责: ai_powered 的 agent-service 远程执行引擎 (client SDK 侧).

定位 (融合方案 2026-09): agent-service (https://git.lambda.lc/lc-studio/agent-service) 作为
本库的服务端实现 —— @ai_powered(agent = RemoteAgent) 把被装饰函数的签名/docstring 编译为
服务端动态 agent (TypeDesc ADT spec) 并经 HTTP 调用, 沙箱执行与 LLM 凭证都在服务端.

轻量性约定: 本子包不进顶层 ai_powered 的 re-export —— 需显式 `from ai_powered.remote import ...`;
零新增第三方依赖 (httpx 为现有传递依赖).

模块划分:
- type_desc.py — Python 类型注解 -> TypeDesc JSON 的纯函数编译器 (wire 契约见该文件头注释)
- client.py — AgentServiceClient: 注册/invoke/404 自愈/错误映射 (httpx 双形态)
- agent.py — RemoteAgent (FunctionAgent 子类, 对等本地 GenericAgent 的远程沙箱引擎)
'''
from .agent import RemoteAgent, SERVICE_URL_ENV
from .client import AgentServiceClient, RemoteContractError
from .type_desc import UnsupportedTypeError, compile_type

__all__ = [
    "RemoteAgent",
    "AgentServiceClient",
    "RemoteContractError",
    "UnsupportedTypeError",
    "compile_type",
    "SERVICE_URL_ENV",
]
