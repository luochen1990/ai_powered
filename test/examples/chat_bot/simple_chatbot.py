from ai_powered import ChatBot
from ai_powered.colors import gray, green

def test_simple_chatbot():
    bot = ChatBot()
    print(green(bot.chat('hello, please tell me the result of 2^10 + 3^4')))
    print(green(bot.chat('and what is above result divided by 2?')))
    print(gray(f"{bot.conversation}"))
