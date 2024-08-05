import inspect
from typing import Callable, Dict, Generic
from typing_extensions import Literal, ParamSpec, Required, TypeVar, TypedDict
import msgspec

from ai_powered.colors import yellow
from ai_powered.constants import DEBUG
from ai_powered.llm_adapter.openai.param_types import ChatCompletionToolMessageParam
from ai_powered.llm_adapter.openai.types import ChatCompletionMessageToolCall
from ai_powered.schema_deref import deref

FunctionParameters = Dict[str, object]

class FunctionDefinition(TypedDict, total=False):
    name: Required[str] #identifier, max length 64
    description: str
    parameters: FunctionParameters

class ChatCompletionToolParam(TypedDict):
    function: Required[FunctionDefinition]
    type: Required[Literal["function"]]

class Function(TypedDict):
    name: Required[str]
class ChatCompletionNamedToolChoiceParam(TypedDict):
    type: Required[Literal["function"]]
    function: Required[Function]

P = ParamSpec("P")
R = TypeVar("R")

class MakeTool(msgspec.Struct, Generic[P, R]):
    fn : Callable[P, R]

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R:
        return self.fn(*args, **kwds)

    def struct_of_parameters(self) -> type[msgspec.Struct]:
        ''' Transmission object corresponding to function parameters '''
        sig = inspect.signature(self.fn)
        properties = [(param.name, param.annotation) for param in sig.parameters.values()]
        ArgObj = msgspec.defstruct("ArgObj", properties)
        return ArgObj

    def schema(self) -> ChatCompletionToolParam:
        function_name = self.fn.__name__
        sig = inspect.signature(self.fn)
        docstring = inspect.getdoc(self.fn)

        if DEBUG:
            print(f"[MakeTool.schema()] {sig =}")

        raw_schema = msgspec.json.schema(self.struct_of_parameters())
        parameters_schema = deref(raw_schema)

        if DEBUG:
            print(yellow(f"[MakeTool.schema()] {parameters_schema =}"))

        return {
            "type": "function",
            "function": {
                "name": function_name,
                "description": docstring or "",
                "parameters": parameters_schema,
            }
        }

    def call(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        assert self.fn.__name__ == tool_call.function.name
        sig = inspect.signature(self.fn)
        args_json : str = tool_call.function.arguments
        args_dict = msgspec.json.decode(args_json)

        args = sig.bind(**args_dict).args

        function_response = self.fn(*args) #type: ignore

        # 构建函数结果消息对象
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": msgspec.json.encode(function_response).decode('utf-8')
        }

    def choice(self) -> ChatCompletionNamedToolChoiceParam:
        function_name = self.fn.__name__
        return {
            "type": "function",
            "function": {
                "name": function_name
            }
        }

def make_tool(fn: Callable[P, R]) -> MakeTool[P, R]:
    ''' Create a tool available for AI from a function '''
    return MakeTool(fn)
