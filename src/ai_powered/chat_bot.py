from dataclasses import dataclass, field
from typing import Any, ClassVar
import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from ai_powered.colors import gray
from ai_powered.constants import DEBUG, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_NAME, OPENAI_MODEL_FEATURES
from ai_powered.llm.known_models import complete_model_config
from ai_powered.tool_call import ChatCompletionToolParam, MakeTool

default_client = openai.OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
model_config = complete_model_config(OPENAI_BASE_URL, OPENAI_MODEL_NAME, OPENAI_MODEL_FEATURES)

@dataclass
class ChatBot:
    ''' A baseclass to define chatbot which can use tools '''

    system_prompt : ClassVar[str] = "" # if not empty, it will prepend to the conversation
    tools: ClassVar[tuple[MakeTool[..., Any], ...]] = ()
    client: ClassVar[openai.OpenAI] = default_client
    conversation : list[ChatCompletionMessageParam] = field(default_factory=lambda:[])

    def __post_init__(self):
        self._system_prompt : list[ChatCompletionMessageParam] = [{"role": "system", "content": s} for s in [self.system_prompt] if len(s) > 0]
        self._tool_dict = {tool.fn.__name__: tool for tool in self.tools}
        self._tool_schemas : list[ChatCompletionToolParam] | openai.NotGiven = [ t.schema() for t in self.tools ] if len(self.tools) > 0 else openai.NOT_GIVEN

    def chat_continue(self) -> str:
        if DEBUG:
            print(gray(f"{self.conversation =}"))

        response = self.client.chat.completions.create(
            model = model_config.model_name,
            messages = [*self._system_prompt, *self.conversation],
            tools = self._tool_schemas,
        )
        assistant_message = response.choices[0].message

        self.conversation.append(assistant_message.to_dict()) #type: ignore

        tool_calls = assistant_message.tool_calls
        if tool_calls is not None:
            if DEBUG:
                print(gray(f"{len(tool_calls) =}"))

            for tool_call in tool_calls:
                using_tool = self._tool_dict[tool_call.function.name]
                tool_call.function.name
                function_message = using_tool.call(tool_call) #type: ignore #TODO: async & parrallel
                self.conversation.append(function_message)

            return self.chat_continue()
        else:
            message_content = assistant_message.content
            assert message_content is not None
            return message_content

    def chat(self, message: str) -> str:
        self.conversation.append({"role": "user", "content": message})
        return self.chat_continue()
