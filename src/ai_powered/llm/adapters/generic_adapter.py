from abc import ABC
from dataclasses import dataclass, field
import json
from typing import Any, ClassVar, Iterable
from easy_sync import sync_compatible
import openai
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.completion_create_params import ResponseFormat
from ai_powered.colors import green, red, yellow
from ai_powered.constants import DEBUG, OPENAI_EXTRA_BODY, base_system_prompt
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import FunctionSimulator, ModelFeature
from ai_powered.tool_call import ChatCompletionToolParam


@dataclass
class GenericFunctionSimulator(FunctionSimulator, ABC):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    connection: LlmConnection
    model_name: str
    model_features: set[ModelFeature]
    model_options: dict[str, Any]
    system_prompt: str = field(init = False)
    _param_tools: Iterable[ChatCompletionToolParam] | openai.Omit = field(init = False)
    _param_tool_choice: ChatCompletionToolChoiceOptionParam | openai.Omit = field(init = False)
    _param_response_format: ResponseFormat | openai.Omit = field(init = False)
    _debug_print_prompt: ClassVar[bool] = True  # selector 子类关闭 (其 prompt 是选择前的猜测, 见 adapter_selector)

    def __post_init__(self):
        self.system_prompt = self._system_prompt_maker()
        self._param_tools = self._param_tools_maker()
        self._param_tool_choice = self._param_tool_choice_maker()
        self._param_response_format = self._param_response_format_maker()
        if DEBUG and self._debug_print_prompt:
            print(f"{self.system_prompt =}")  # 打印实际生效的 prompt (构造后的准确值, 而非装饰器侧的推测)

    def _system_prompt_maker(self) -> str:
        ''' default impl is provided '''
        common_system_prompt = base_system_prompt(self.spec)
        extra_system_prompt = self._extra_system_prompt_maker()
        return common_system_prompt + extra_system_prompt

    def _extra_system_prompt_maker(self) -> str:
        ''' to be overrided '''
        return ""

    def _param_response_format_maker(self) -> ResponseFormat | openai.Omit:
        ''' to be overrided '''
        return openai.omit

    def _param_tools_maker(self) -> Iterable[ChatCompletionToolParam] | openai.Omit:
        ''' to be overrided '''
        return openai.omit

    def _param_tool_choice_maker(self) -> ChatCompletionToolChoiceOptionParam | openai.Omit:
        ''' to be overrided '''
        return openai.omit

    @sync_compatible
    async def _chat_completion_query(self, arguments_json: str) -> ChatCompletion:
        ''' default impl is provided '''
        if DEBUG:
            print(f"{self._param_response_format =}")

        # NOTE: extra_body 直传 None 表示 "不发" (openai SDK 对该参数是 None 门控,
        # 与 tools 等 NotGiven 门控参数不同 —— 后者才用 openai.omit; 误传 omit 会在
        # 未设 OPENAI_EXTRA_BODY 的默认路径上以 TypeError 崩溃, 且经
        # wraps_method_arguments_type 包装的调用不受 pyright 强制检查, 类型系统拦不住).
        # 网关 200+error 防御在 LlmConnection 咽喉点 (见 connection.py).
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
            tool_choice = self._param_tool_choice,
            response_format = self._param_response_format,
            extra_body = OPENAI_EXTRA_BODY,
        )

    def _response_message_parser(self, response_message: ChatCompletionMessage) -> str:
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
            print(green(f"[fn {self.spec.function_name}] request prepared."))

        response = await self._chat_completion_query(arguments_json)

        if DEBUG:
            print(yellow(f"{response =}"))
            print(green(f"[fn {self.spec.function_name}] response received."))

        response_message = response.choices[0].message
        result_str = self._response_message_parser(response_message)
        return result_str
