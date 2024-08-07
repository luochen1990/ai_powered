import json
from typing import Callable
import re
from ai_powered.constants import DEBUG, SYSTEM_PROMPT_JSON_SYNTAX, SYSTEM_PROMPT_RETURN_SCHEMA
from ai_powered.llm.adapters.generic_adapter import GenericFunctionSimulator
from openai.types.chat.chat_completion_message import ChatCompletionMessage


class ChatFunctionSimulator (GenericFunctionSimulator):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    def _extra_system_prompt_maker(self) -> str:
        return SYSTEM_PROMPT_RETURN_SCHEMA.format(
            return_schema = json.dumps(self.return_schema),
        ) + SYSTEM_PROMPT_JSON_SYNTAX

    def _response_message_parser(self, response_message: ChatCompletionMessage) -> str:
        tool_calls = response_message.tool_calls

        if tool_calls is not None:
            return tool_calls[0].function.arguments
        else:
            raw_resp_str = response_message.content
            assert raw_resp_str is not None

            raw_resp_str_strip = raw_resp_str.strip()
            # raw_resp_str = "```json\n{"result": 2}\n```"

            is_markdown : Callable[[str], bool]= lambda s: s.startswith("```") and s.endswith("```")

            if is_markdown(raw_resp_str_strip):
                if raw_resp_str_strip.startswith("```json"):
                    unwrapped_resp_str = raw_resp_str_strip[7:-3]
                else:
                    unwrapped_resp_str = raw_resp_str_strip[3:-3]
            else:
                unwrapped_resp_str = raw_resp_str_strip

            if DEBUG:
                print(f"{unwrapped_resp_str =}")

            is_result : Callable[[str], bool] = lambda s: re.match(r'^\s*\{\s*"result":' , s) is not None

            # unwrapped_result_str = "2" | "{"result": 2}" | "{ "result": 2 }"
            if is_result(unwrapped_resp_str):
                result_str = unwrapped_resp_str
            else:
                result_str = f'{{"result": {unwrapped_resp_str}}}'

            if DEBUG:
                print(f"{raw_resp_str =}")
                print(f"{is_markdown(raw_resp_str_strip) =}")
                print(f"{is_result(unwrapped_resp_str) =}")
                print(f"{result_str =}")

            return result_str
