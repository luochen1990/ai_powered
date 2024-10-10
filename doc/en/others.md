Other API Provided
==================

### `@make_tool` Decorator

This decorator converts ordinary Python functions into tools that can be used by the [function calling](https://platform.openai.com/docs/guides/function-calling) feature of LLM.

```python
from ai_powered import make_tool
import openai

@make_tool
def calculator(python_expression: str) -> str:
    ''' Evaluate a Python expression (only support built-in functions), which can be used to solve mathematical problems. '''
    return safe_eval(python_expression)

client = openai.OpenAI()
response = client.chat.completions.create(
    model = "gpt-4o-mini",
    messages = self.conversation,
    tools = [ calculator.schema() ],
    tool_choice = calculator.choice()
)
```

### `ChatBot` Class

This class implements an AI chatbot. You can inherit from it to create your own ChatBot subclass, specifying the system prompts and tools to use. It will help you handle the complex process of tool invocation.

```python
class MyChatBot (ChatBot):
    system_prompt = '''
    Please answer the user's questions. If any calculations are required, use the calculator available in the tool. It supports complex Python expressions. When using it, make sure to convert the user's mathematical expression to a valid Python expression. Do not use any undefined functions; if the user's expression includes function calls, convert them to Python's built-in functions or syntax.
    '''
    tools = (calculator,)

if __name__ == "__main__":
    bot = MyChatBot()
    print(bot.chat('hello, please tell me the result of 2^10 + 3^4'))
    print(bot.chat('and what is above result divided by 2?'))
    print(f"{bot.conversation =}")
```

More examples can be found [here](/test/examples/chat_bot/)
