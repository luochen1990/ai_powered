from abc import ABC
from dataclasses import dataclass
import enum
from typing import Any, Optional

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
