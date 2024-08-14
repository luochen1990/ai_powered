from typing import Iterable
import openai
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam
from ai_powered.tool_call import ChatCompletionToolParam
from ai_powered.llm.adapters.generic_adapter import GenericFunctionSimulator
from ai_powered.utils.parse_message import extract_json_from_message

class ToolsFunctionSimulator (GenericFunctionSimulator):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    def _param_tools_maker(self) -> Iterable[ChatCompletionToolParam] | openai.NotGiven:
        return [{
            "type": "function",
            "function": {
                "name": "return_result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": self.return_schema,
                    },
                    "required": ["result"],
                }
            },
        }]

    def _param_tool_choice_maker(self) -> ChatCompletionToolChoiceOptionParam | openai.NotGiven:
        return {"type": "function", "function": {"name": "return_result"}}

    #@override
    def _response_message_parser(self, response_message: ChatCompletionMessage) -> str:
        tool_calls = response_message.tool_calls

        if tool_calls is not None:
            return tool_calls[0].function.arguments
        else: # 兼容不支持 tool_choice 的情况, 比如 ollama
            message = response_message.content
            assert message is not None

            json_str = extract_json_from_message(message)
            assert json_str is not None

            return json_str
