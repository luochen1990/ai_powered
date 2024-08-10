from ai_powered import ChatBot
from ai_powered.colors import gray, green
from ai_powered.tools.google_search import google_search

class MyChatBot (ChatBot):
    system_prompt = '''
    Please answer the user's questions. If any calculations are required, use the calculator available in the tool. It supports complex Python expressions. When using it, make sure to convert the user's mathematical expression to a valid Python expression. Do not use any undefined functions; if the user's expression includes function calls, convert them to Python's built-in functions or syntax.
    '''
    tools = (google_search,)

def test_use_google_search():
    bot = MyChatBot()
    print(green(bot.chat("what's USD price in CNY today?").wait()))
    print(gray(f"{bot.conversation}"))
