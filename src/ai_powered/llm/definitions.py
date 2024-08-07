from abc import ABC
from dataclasses import dataclass
import enum
from typing import Any, Optional

'''
    信息确定的过程:
    首先是连接信息被确定,如BASE URL,API KEY等
    然后是模型信息被确定,如模型名称，模型参数等
    模型信息确定后,模型支持的特性随之被确定,如是否支持函数调用,是否支持JSON格式的返回值等
    再然后是函数信息被确定,如函数签名，函数文档等
    最终模拟函数执行时,需要参考函数信息和模型信息,以及连接信息,来确定函数执行的具体方式
'''

class ModelFeature (enum.Enum):
    '''
    Ollama Doc: https://ollama.fan/reference/openai/#supported-features
    OpenAI Doc:
    - tools/function_calling: https://platform.openai.com/docs/guides/function-calling
    - structured_outputs/strict_mode: https://platform.openai.com/docs/guides/structured-outputs
    - seed/reproducible_outputs: https://platform.openai.com/docs/advanced-usage/reproducible-outputs
    '''
    tools = "function_calling"
    function_calling = "function_calling"
    seed = "reproducible_outputs"
    reproducible_outputs = "reproducible_outputs"
    structured_outputs = "structured_outputs"
    strict_mode = "strict_mode"

ALL_FEATURES : set[ModelFeature] = {x for x in ModelFeature}


@dataclass
class FunctionSimulator (ABC):
    ''' just a wrapper to call model, without checking type correctness '''

    function_name: str
    signature: str
    docstring: Optional[str]
    parameters_schema: dict[str, Any]
    return_schema: dict[str, Any]

    def query_model(self, arguments_json: str) -> str:
        ...
