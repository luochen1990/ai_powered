import openai
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.completion_create_params import ResponseFormat
from ai_powered.llm.adapters.generic_adapter import GenericFunctionSimulator
#from ai_powered.llm.definitions import ModelFeature

class StructuredOutputFunctionSimulator (GenericFunctionSimulator):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    def _param_response_format_maker(self) -> ResponseFormat | openai.NotGiven:
        ''' to be overrided '''
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "math_response",
                #"strict": True if ModelFeature.strict_mode in self.model_features else None, #NOTE: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": self.return_schema,
                    },
                    "required": ["result"],
                }
            }
        }

    def _response_message_parser(self, response_message: ChatCompletionMessage) -> str:
        tool_calls = response_message.tool_calls
        assert tool_calls is None

        raw_resp_str = response_message.content
        assert raw_resp_str is not None

        return raw_resp_str
