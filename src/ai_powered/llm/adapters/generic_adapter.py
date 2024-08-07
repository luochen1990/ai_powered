from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Set
import openai
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.completion_create_params import ResponseFormat
from ai_powered.colors import green, yellow
from ai_powered.constants import DEBUG, SYSTEM_PROMPT
from ai_powered.llm.definitions import FunctionSimulator, ModelFeature
from ai_powered.tool_call import ChatCompletionToolParam

@dataclass
class GenericFunctionSimulator (FunctionSimulator):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    client: openai.OpenAI
    model_name: str
    model_features: Set[ModelFeature]
    model_options: dict[str, Any]
    system_prompt : str = field(init=False)
    _param_tools : Iterable[ChatCompletionToolParam] | openai.NotGiven = field(init=False)
    _param_tool_choice : ChatCompletionToolChoiceOptionParam | openai.NotGiven = field(init=False)
    _param_response_format : ResponseFormat | openai.NotGiven = field(init=False)

    def __post_init__(self):
        self.system_prompt = self._system_prompt_maker()
        self._param_tools = self._param_tools_maker()
        self._param_tool_choice = self._param_tool_choice_maker()
        self._param_response_format = self._param_response_format_maker()

    def _system_prompt_maker(self) -> str:
        ''' default impl is provided '''
        common_system_prompt = SYSTEM_PROMPT.format(
            signature = self.signature,
            docstring = self.docstring or "no doc, guess intention from function name",
            parameters_schema = json.dumps(self.parameters_schema),
        )
        extra_system_prompt = self._extra_system_prompt_maker()
        return common_system_prompt + extra_system_prompt

    def _extra_system_prompt_maker(self) -> str:
        ''' to be overrided '''
        return ""

    def _param_response_format_maker(self) -> ResponseFormat | openai.NotGiven:
        ''' to be overrided '''
        return openai.NOT_GIVEN

    def _param_tools_maker(self) -> Iterable[ChatCompletionToolParam] | openai.NotGiven:
        ''' to be overrided '''
        return openai.NOT_GIVEN

    def _param_tool_choice_maker(self) -> ChatCompletionToolChoiceOptionParam | openai.NotGiven:
        ''' to be overrided '''
        return openai.NOT_GIVEN

    def _chat_completion_query(self, arguments_json: str) -> ChatCompletion:
        ''' default impl is provided '''
        return self.client.chat.completions.create(
            model = self.model_name,
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": arguments_json}
            ],
            tools = self._param_tools,
            tool_choice = self._param_tool_choice,
            response_format=self._param_response_format,
        )

    def _response_message_parser(self, response_message: ChatCompletionMessage) -> str:
        ''' to be overrided '''
        ...

    def query_model(self, arguments_json: str) -> str:

        if DEBUG:
            print(yellow(f"{arguments_json =}"))
            print(yellow(f"request.tools = {self._param_tools}"))
            print(green(f"[fn {self.function_name}] request prepared."))

        response = self._chat_completion_query(arguments_json)

        if DEBUG:
            print(yellow(f"{response =}"))
            print(green(f"[fn {self.function_name}] response received."))

        response_message = response.choices[0].message
        result_str = self._response_message_parser(response_message)
        return result_str
