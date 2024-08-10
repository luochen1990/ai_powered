from ai_powered import ChatBot, make_tool
from ai_powered.colors import gray, green
from ai_powered.utils.safe_eval import safe_eval

@make_tool
def calculator(python_expression: str) -> str:
    ''' calculate the result of the math expression in python syntax and built-in functions '''
    print(f"{python_expression =}")
    calc_result = safe_eval(python_expression)
    print(f"{calc_result =}")
    rst = f"{calc_result}"
    return rst

class MyChatBot (ChatBot):
    system_prompt = '''
    Please answer the user's questions. If any calculations are required, use the calculator available in the tool. It supports complex Python expressions. When using it, make sure to convert the user's mathematical expression to a valid Python expression. Do not use any undefined functions; if the user's expression includes function calls, convert them to Python's built-in functions or syntax.
    '''
    tools = (calculator,)

def test_use_calculator():
    bot = MyChatBot()
    print(green(bot.chat('hello, please tell me the result of 2^10 + 3^4').wait()))
    print(green(bot.chat('and what is above result divided by 2?').wait()))
    print(gray(f"{bot.conversation}"))
