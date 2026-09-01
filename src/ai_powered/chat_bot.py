from dataclasses import dataclass, field
from typing import Any, ClassVar
import openai
from easy_sync import sync_compatible
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from ai_powered.colors import gray
from ai_powered.constants import DEBUG
from ai_powered.llm.connection import LlmConnection
from ai_powered.llm.definitions import ModelFeature
from ai_powered.llm.resolver import resolve_model
from ai_powered.tool_call import MakeTool

#: 模块级默认解析 (SSOT: 与 @ai_powered 装饰器共用 resolve_model, 见 llm/resolver.py)
_default_resolved = resolve_model(None)
default_connection = _default_resolved.connection
model_config = _default_resolved.model_config


@dataclass
class ChatBot:
    ''' A baseclass to define chatbot which can use tools '''

    system_prompts: ClassVar[list[str]] = []
    tools: ClassVar[tuple[MakeTool[..., Any], ...]] = ()
    connection: ClassVar[LlmConnection] = default_connection
    conversation: list[ChatCompletionMessageParam] = field(default_factory = lambda: [])

    def __post_init__(self):
        self._tool_dict = {tool.fn.__name__: tool for tool in self.tools}
        # 工具参数 schema 的能力门控位 (issue 18): 递归参数类型按通道能力透传或 fail-fast
        refs_allowed = ModelFeature.schema_refs in model_config.supported_features
        self._tool_schemas: list[ChatCompletionToolParam] | openai.Omit = [t.schema(refs_allowed = refs_allowed) for t in self.tools] if len(self.tools) > 0 else openai.omit

    @sync_compatible
    async def chat_continue(self) -> str:
        if DEBUG:
            print(gray(f"{self.conversation =}"))

        # NOTE: system_prompts might be updated dynamically
        self._system_prompts: list[ChatCompletionMessageParam] = [{"role": "system", "content": s} for s in self.system_prompts]

        response = await self.connection.chat_completions(
            model = model_config.model_name,
            messages = [*self._system_prompts, *self.conversation],
            tools = self._tool_schemas,
            tool_choice = "auto" if len(self.tools) > 0 else openai.omit,
        )
        assistant_message = response.choices[0].message

        self.conversation.append(assistant_message.to_dict())  #type: ignore

        tool_calls = assistant_message.tool_calls
        if tool_calls is not None:
            if DEBUG:
                print(gray(f"{len(tool_calls) =}"))

            for tool_call in tool_calls:
                if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                    continue  # 仅处理 function 类型的 tool call, 忽略 custom 类型 (如 MCP 工具)
                using_tool = self._tool_dict[tool_call.function.name]
                function_message = using_tool.call(tool_call)  #type: ignore #TODO: async & parrallel
                self.conversation.append(function_message)

            return await self.chat_continue()
        else:
            message_content = assistant_message.content
            assert message_content is not None
            return message_content

    @sync_compatible
    async def chat(self, message: str) -> str:
        self.conversation.append({"role": "user", "content": message})
        return await self.chat_continue()
