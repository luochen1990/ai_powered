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


@dataclass
class FunctionSimulator(ABC):
    ''' just a wrapper to call model, without checking type correctness '''

    spec: FunctionSpec

    @sync_compatible
    async def query_model(self, arguments_json: str) -> str:
        ...
