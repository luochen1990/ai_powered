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
                "schema": self.return_schema,
            }
        }

    def _response_message_parser(self, response_message: ChatCompletionMessage) -> str:
        tool_calls = response_message.tool_calls

        assert tool_calls is not None
        return tool_calls[0].function.arguments
