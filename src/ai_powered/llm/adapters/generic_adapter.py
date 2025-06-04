from abc import ABC
from dataclasses import dataclass, field
import json
from typing import Any, Set, TypeAlias
from easy_sync import sync_compatible
from litellm import BaseModel, ChatCompletionToolParam, ChatCompletionResponseMessage, ChatCompletionToolChoiceObjectParam

from ai_powered.colors import green, red, yellow
from ai_powered.constants import DEBUG, SYSTEM_PROMPT
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import FunctionSimulator, ModelFeature

ResponseFormat: TypeAlias = dict[str, Any] | type[BaseModel] | None


@dataclass
class GenericFunctionSimulator(FunctionSimulator, ABC):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    connection: LlmConnection
    model_name: str
    model_features: Set[ModelFeature]
    model_options: dict[str, Any]
    system_prompt: str = field(init = False)
    _param_tools: list[ChatCompletionToolParam] | None = field(init = False)
    _param_tool_choice: ChatCompletionToolChoiceObjectParam | None = field(init = False)
    _param_response_format: ResponseFormat = field(init = False)

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

    def _param_response_format_maker(self) -> ResponseFormat:
        ''' to be overrided '''
        return None

    def _param_tools_maker(self) -> list[ChatCompletionToolParam] | None:
        ''' to be overrided '''
        return None

    def _param_tool_choice_maker(self) -> ChatCompletionToolChoiceObjectParam | None:
        ''' to be overrided '''
        return None

    @sync_compatible
    async def _chat_completion_query(self, arguments_json: str) -> Any:
        ''' default impl is provided '''
        return await self.connection.chat_completions(
            model = self.model_name,
            messages = [{
                "role": "system",
                "content": self.system_prompt
            }, {
                "role": "user",
                "content": arguments_json
            }],
            tools = self._param_tools,
            tool_choice = self._param_tool_choice,  #type: ignore
            response_format = self._param_response_format,
        )

    def _response_message_parser(self, response_message: ChatCompletionResponseMessage) -> str:
        ''' to be overrided '''
        if DEBUG:
            print(red(f"[GenericFunctionSimulator._response_message_parser()] {self.__class__ =}, {self._response_message_parser =}"))
        raise NotImplementedError

    #@override
    @sync_compatible
    async def query_model(self, arguments_json: str) -> str:

        if DEBUG:
            print(yellow(f"{arguments_json =}"))
            print(yellow(f"request.tools = {self._param_tools}"))
            print(green(f"[fn {self.function_name}] request prepared."))

        response = await self._chat_completion_query(arguments_json)

        if DEBUG:
            print(yellow(f"{response =}"))
            print(green(f"[fn {self.function_name}] response received."))

        response_message = response.choices[0].message
        result_str = self._response_message_parser(response_message)
        return result_str
