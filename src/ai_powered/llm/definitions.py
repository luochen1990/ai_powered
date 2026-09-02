from abc import ABC
from dataclasses import dataclass
import enum
from typing import Any, Optional

from easy_sync import sync_compatible


class ModelFeature(enum.Enum):
    '''
    Ollama Doc: https://ollama.fan/reference/openai/#supported-features
    OpenAI Doc:
    - tools/function_calling: https://platform.openai.com/docs/guides/function-calling
    - structured_outputs/strict_mode: https://platform.openai.com/docs/guides/structured-outputs
    - seed/reproducible_outputs: https://platform.openai.com/docs/advanced-usage/reproducible-outputs

    schema_refs: 模型通道可接受 schema 中的 $ref/$defs (含递归引用);
    关闭时 deref 全量展开、遇环 fail-fast 抛 RecursiveSchemaError (issue 18)
    '''
    tools = "function_calling"
    function_calling = "function_calling"
    seed = "reproducible_outputs"
    reproducible_outputs = "reproducible_outputs"
    structured_outputs = "structured_outputs"
    strict_mode = "strict_mode"
    schema_refs = "schema_refs"


ALL_FEATURES: set[ModelFeature] = {x for x in ModelFeature}


@dataclass(frozen = True)
class FunctionSpec:
    ''' 被模拟函数的规格值对象: 装饰期从签名/docstring/类型注解一次性提取

    单次 LLM 请求模拟器 (FunctionSimulator 族) 与多轮 agent (FunctionAgent) 共用同一份规格,
    是 "AI 如何扮演这个函数" 的单一事实来源.
    '''

    function_name: str
    signature: str
    docstring: Optional[str]
    parameters_schema: dict[str, Any]
    #: raw schema (msgspec 原始输出, 顶层可为 $ref 形态) —— 能力门控的展开/透传延迟到各消费点 (adapters/agent) 按 schema_refs 能力决策 (issue 18)
    return_schema: dict[str, Any]
    #: 原始类型注解 (param.name -> annotation), 供需要类型层信息的引擎消费 (如远程通道的 TypeDesc 编译);
    #: None = 构造方未提供 (旧路径兼容), JSON schema 仍是所有引擎的公共底座
    parameter_annotations: Optional[dict[str, Any]] = None
    #: 带 default 的参数 (param.name -> default value, 只含显式带缺省者) —— 调用方省略时 sig.bind
    #: 不含该参数, 远程通道据此对 input_spec 的该字段包 maybe (服务端输入校验允许省略)
    parameter_defaults: Optional[dict[str, Any]] = None
    #: 返回值原始注解, 语义同上; None = 未提供 (注意: "注解为 NoneType" 的函数不在本库语义内, 不与缺省歧义)
    return_annotation: Any = None


@dataclass
class FunctionSimulator(ABC):
    ''' just a wrapper to call model, without checking type correctness '''

    spec: FunctionSpec

    @sync_compatible
    async def query_model(self, arguments_json: str) -> str:
        ...
