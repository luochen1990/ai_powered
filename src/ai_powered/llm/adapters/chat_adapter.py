import json
import re
from litellm import ChatCompletionResponseMessage
from ai_powered.constants import DEBUG, SYSTEM_PROMPT_JSON_SYNTAX, SYSTEM_PROMPT_RETURN_SCHEMA
from ai_powered.llm.adapters.generic_adapter import GenericFunctionSimulator
from ai_powered.utils.parse_message import extract_json_from_message

_result_pattern = re.compile(r'^\s*\{\s*"result"\s*:')  # "{"result": 2}" | "{ "result" : 2 }"


class ChatFunctionSimulator(GenericFunctionSimulator):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    def _extra_system_prompt_maker(self) -> str:
        return SYSTEM_PROMPT_RETURN_SCHEMA.format(return_schema = json.dumps(self.return_schema),) + SYSTEM_PROMPT_JSON_SYNTAX

    def _response_message_parser(self, response_message: ChatCompletionResponseMessage) -> str:
        tool_calls = response_message.tool_calls  # type: ignore

        if tool_calls is not None:
            tc0 = tool_calls[0]
            tc0fn = tc0.function
            return tc0fn.arguments
        else:
            assert "content" in response_message, f"response message content not found in {response_message =}"
            raw_resp_str = response_message.content  # type: ignore
            assert raw_resp_str is not None

            if DEBUG:
                print(f"{raw_resp_str =}")

            json_str = extract_json_from_message(raw_resp_str) or raw_resp_str

            if DEBUG:
                print(f"{json_str =}")

            if re.match(_result_pattern, json_str):  # wrapped by `{"result": ...}`
                result_str = json_str
            else:  # not wrapped by `{"result": ...}`
                result_str = f'{{"result": {json_str}}}'

            if DEBUG:
                print(f"{result_str =}")

            return result_str
