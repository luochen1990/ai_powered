from typing import Any
from litellm import BaseModel, ChatCompletionResponseMessage
from ai_powered.llm.adapters.generic_adapter import GenericFunctionSimulator
#from ai_powered.llm.definitions import ModelFeature

ResponseFormat = dict[str, Any] | type[BaseModel] | None


class StructuredOutputFunctionSimulator(GenericFunctionSimulator):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    def _param_response_format_maker(self) -> ResponseFormat:
        ''' to be overrided '''
        return {
            "type": "json_schema",
            "json_schema":
                {
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

    def _response_message_parser(self, response_message: ChatCompletionResponseMessage) -> str:
        assert "tool_calls" not in response_message

        assert "content" in response_message
        raw_resp_str = response_message["content"]
        assert raw_resp_str is not None

        return raw_resp_str
