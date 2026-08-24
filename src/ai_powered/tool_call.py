import inspect
from typing import Any, Callable, Generic
from typing_extensions import ParamSpec, TypeVar
import msgspec

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_named_tool_choice_param import ChatCompletionNamedToolChoiceParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from ai_powered.colors import gray, green
from ai_powered.constants import DEBUG
from ai_powered.schema_deref import deref

P = ParamSpec("P")
R = TypeVar("R")


def return_result_tool_param(return_schema: dict[str, Any]) -> ChatCompletionToolParam:
    ''' 构造 "return_result" 工具的 OpenAI 参数: 参数 schema 即 {"result": 返回值schema}

    单次模拟 (ToolsFunctionSimulator) 与多轮 agent (ReasoningAgent) 共用同一提交契约 —— 模型以
    调用该工具的方式提交结构化结果, 参数即最终返回值.
    '''
    return {
        "type": "function",
        "function":
            {
                "name": "return_result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": return_schema,
                    },
                    "required": ["result"],
                }
            },
    }


class MakeTool(msgspec.Struct, Generic[P, R]):
    fn: Callable[P, R]

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
            print(gray(f"[MakeTool.schema()] {sig =}"))

        raw_schema = msgspec.json.schema(self.struct_of_parameters())
        parameters_schema = deref(raw_schema)

        if DEBUG:
            print(gray(f"[MakeTool.schema()] {parameters_schema =}"))

        return {
            "type": "function",
            "function": {
                "name": function_name,
                "description": docstring or "",
                "parameters": parameters_schema,
            }
        }

    def call(self, tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessageParam:
        if DEBUG:
            print(green(f"[MakeTool.call()] {tool_call =}"))

        assert self.fn.__name__ == tool_call.function.name

        sig = inspect.signature(self.fn)
        args_json: str = tool_call.function.arguments
        args_obj = msgspec.json.decode(args_json, type = self.struct_of_parameters())
        args_dict = msgspec.structs.asdict(args_obj)

        if DEBUG:
            print(gray(f"[MakeTool.call()] {args_dict =}"))

        args = sig.bind(**args_dict).args

        function_response = self.fn(*args)  #type: ignore

        # 构建函数结果消息对象
        return {"role": "tool", "tool_call_id": tool_call.id, "content": msgspec.json.encode(function_response).decode('utf-8')}

    def choice(self) -> ChatCompletionNamedToolChoiceParam:
        function_name = self.fn.__name__
        return {"type": "function", "function": {"name": function_name}}


def make_tool(fn: Callable[P, R]) -> MakeTool[P, R]:
    ''' Create a tool available for AI from a function '''
    return MakeTool(fn)
