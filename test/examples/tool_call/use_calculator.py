from msgspec import Struct, field
import openai

from ai_powered.colors import gray, green
from ai_powered.constants import DEBUG, OPENAI_BASE_URL, OPENAI_MODEL_NAME
from ai_powered.llm_adapter.known_models import complete_model_config
from ai_powered.llm_adapter.openai.param_types import ChatCompletionMessageParam
from ai_powered.tool_call import MakeTool
from ai_powered.utils.safe_eval import safe_eval

model_config = complete_model_config(OPENAI_BASE_URL, OPENAI_MODEL_NAME)

def calculator(python_expression: str) -> str:
    ''' calculate the result of the math expression in python syntax and built-in functions '''
    print(f"{python_expression =}")
    calc_result = safe_eval(python_expression)
    print(f"{calc_result =}")
    rst = f"{calc_result}"
    return rst

calculator_tool = MakeTool(calculator)

sys_prompt = '''
Please answer the user's questions. If any calculations are required, use the calculator available in the tool. It supports complex Python expressions. When using it, make sure to convert the user's mathematical expression to a valid Python expression. Do not use any undefined functions; if the user's expression includes function calls, convert them to Python's built-in functions or syntax.
'''

client = openai.OpenAI(**model_config.suggested_options)

class ChatBotWithGoodMathSkill (Struct):
    conversation : list[ChatCompletionMessageParam] = field(default_factory=lambda:[
        {"role": "system", "content": sys_prompt},
    ])

    def chat_continue(self) -> str:
        if DEBUG:
            print(gray(f"{self.conversation =}"))

        response = client.chat.completions.create(
            model = model_config.model_name,
            messages = self.conversation,
            tools = [ calculator_tool.schema() ],
        )
        assistant_message = response.choices[0].message

        self.conversation.append(assistant_message.to_dict()) #type: ignore

        tool_calls = assistant_message.tool_calls
        if tool_calls is not None:
            if DEBUG:
                print(gray(f"{len(tool_calls) =}"))
            #function_message_list : list[ChatCompletionFunctionMessageParam] = []
            for tool_call in tool_calls:
                function_message = calculator_tool.call(tool_call) #type: ignore #TODO: async & parrallel
                #function_message_list.append(function_message)
                self.conversation.append(function_message)

            return self.chat_continue()
        else:
            message_content = assistant_message.content
            assert message_content is not None
            return message_content

    def chat(self, message: str) -> str:
        self.conversation.append({"role": "user", "content": message})
        return self.chat_continue()

def test_use_calculator():
    bot = ChatBotWithGoodMathSkill()
    print(green(bot.chat('hello, please tell me the result of 2^10 + 3^4')))
    print(green(bot.chat('and what is above result divided by 2?')))
    print(gray(f"{bot.conversation}"))
