from ai_powered import ChatBot
from ai_powered.colors import gray, green
from ai_powered.tools.google_search import google_search

class MyChatBot (ChatBot):
    system_prompts = ['''Please answer the user's questions. When you need up-to-date information (prices, news, events), use the google_search tool to search the web first, then answer based on the search results.''']
    tools = (google_search,)

def test_use_google_search():
    bot = MyChatBot()
    print(green(bot.chat("what's USD price in CNY today?").wait()))
    print(gray(f"{bot.conversation}"))
