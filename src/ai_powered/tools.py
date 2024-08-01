import inspect
from typing import Any, Callable, ParamSpec, TypeVar, TypedDict
import msgspec
import openai

from ai_powered.constants import DEBUG

class ChatCompletionToolParam(TypedDict):
    function: openai.types.FunctionDefinition
    type: str

P = ParamSpec("P")
R = TypeVar("R")

def schema_of_parameters_as_object(sig : inspect.Signature) -> dict[str, Any]:
    # TODO: support context references (need support by OpenAI API)
    # schema_list, ctxt = msgspec.json.schema_components([param.annotation for param in sig.parameters.values()])
    # properties = {param.name: sch for sch, param in zip(schema_list, sig.parameters.values())}
    properties = {param.name: msgspec.json.schema(param.annotation) for param in sig.parameters.values()}
    return {
        "type": "object",
        "properties": properties,
        "required": ["result"],
    }

def tool_schema(fn : Callable[P, R]) -> ChatCompletionToolParam:

    function_name = fn.__name__
    sig = inspect.signature(fn)
    docstring = inspect.getdoc(fn)

    if DEBUG:
        print(f"{sig =}")
        print(f"{docstring =}")

        for param in sig.parameters.values():
            print(f"{param.name}: {param.annotation}")

        print(f"{sig.return_annotation =}")

    parameters_schema = schema_of_parameters_as_object(sig)

    return {
        "type": "function",
        "function": openai.types.FunctionDefinition(
            name= function_name,
            description= docstring or "",
            parameters= parameters_schema,
        ),
    }
