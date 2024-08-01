from typing import Any, Callable, Set
from dataclasses import dataclass
import re
import openai
from ai_powered.colors import green, yellow
from ai_powered.constants import DEBUG
from ai_powered.llm_adapter.definitions import FunctionSimulator, ModelFeature

def _return_schema_wrapper(return_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "result": return_schema,
        },
        "required": ["result"],
    }

@dataclass(frozen=True)
class GenericFunctionSimulator (FunctionSimulator):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    base_url: str
    api_key: str
    model_name: str
    model_features: Set[ModelFeature]
    model_options: dict[str, Any]
    system_prompt : str

    def query_model(self, user_msg: str) -> str:
        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key, **self.model_options)
        #return_schema_need_wrap : bool = self.return_schema["type"] != "object"


        messages : Any = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg}
        ]
        tools : Any = [{
            "type": "function",
            "function": {
                "name": "return_result",
                "parameters": _return_schema_wrapper(self.return_schema)
            },
        }]
        if DEBUG:
            print(yellow(f"request.messages = {messages}"))
            print(yellow(f"request.tools = {tools}"))
            print(green(f"[fn {self.function_name}] request prepared."))

        response = client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            tools = tools if "function_call" in self.model_features else openai.NOT_GIVEN,
            tool_choice = {"type": "function", "function": {"name": "return_result"}} if "function_call" in self.model_features else openai.NOT_GIVEN,
        )
        if DEBUG:
            print(yellow(f"{response =}"))
            print(green(f"[fn {self.function_name}] response received."))

        resp_msg = response.choices[0].message
        tool_calls = resp_msg.tool_calls

        if tool_calls is not None:
            return tool_calls[0].function.arguments
        else:
            raw_resp_str = resp_msg.content
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
