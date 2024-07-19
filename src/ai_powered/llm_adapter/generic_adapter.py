from dataclasses import dataclass
from typing import Any, Set
from ai_powered.colors import green
from ai_powered.constants import DEBUG
from .definitions import FunctionSimulator, ModelFeature
import openai


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

        response = client.chat.completions.create(
            model = self.model_name,
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg}
            ],
            tools = [{
                "type": "function",
                "function": {
                    "name": "return_result",
                    "parameters": self.return_schema,
                },
            }],
            tool_choice = {"type": "function", "function": {"name": "return_result"}},
        )

        if DEBUG:
            print(f"{response =}")
            print(green(f"[fn {self.function_name}] response received."))

        resp_msg = response.choices[0].message
        tool_calls = resp_msg.tool_calls

        if tool_calls is not None:
            return tool_calls[0].function.arguments
        else:
            raw_resp_str = resp_msg.content
            assert raw_resp_str is not None

            # raw_resp_str = "```json\n{"result": 2}\n```"

            if raw_resp_str.startswith("```json\n") and raw_resp_str.endswith("\n```"):
                unwrapped_resp_str = raw_resp_str[8:-4]
            else:
                unwrapped_resp_str = raw_resp_str

            # unwrapped_result_str = "2"

            if unwrapped_resp_str.startswith('{"result":') and unwrapped_resp_str.endswith("}"):
                result_str = unwrapped_resp_str
            else:
                result_str = f'{{"result": {unwrapped_resp_str}}}'

            if DEBUG:
                print(f"{result_str =}")
            return result_str
