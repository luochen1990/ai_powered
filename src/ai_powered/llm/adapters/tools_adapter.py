from litellm import ChatCompletionToolParam, ChatCompletionToolChoiceObjectParam, ChatCompletionResponseMessage, ToolChoice
from ai_powered.llm.adapters.generic_adapter import GenericFunctionSimulator
from ai_powered.utils.parse_message import extract_json_from_message


class ToolsFunctionSimulator(GenericFunctionSimulator):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    def _param_tools_maker(self) -> list[ChatCompletionToolParam] | None:
        return [
            {
                "type": "function",
                "function":
                    {
                        "name": "return_result",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "result": self.return_schema,
                            },
                            "required": ["result"],
                        }
                    },
            }
        ]

    def _param_tool_choice_maker(self) -> ChatCompletionToolChoiceObjectParam | None:
        return {"type": "function", "function": {"name": "return_result"}}

    #@override
    def _response_message_parser(self, response_message: ChatCompletionResponseMessage) -> str:
        tool_calls = response_message.tool_calls  # type: ignore

        if tool_calls is not None:
            tc0 = tool_calls[0]
            tc0fn = tc0.function
            return tc0fn.arguments
        else:  # 兼容不支持 tool_choice 的情况, 比如 ollama
            assert "content" in response_message, f"response message content not found in {response_message =}"
            message = response_message.content  # type: ignore
            assert message is not None

            json_str = extract_json_from_message(message)
            assert json_str is not None

            return json_str
