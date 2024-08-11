import pytest
from ai_powered import ChatBot
from ai_powered.colors import gray, green

@pytest.mark.asyncio
async def test_simple_chatbot_async():
    bot = ChatBot()
    print(green(await bot.chat('hello, please tell me the result of 2^10 + 3^4')))
    print(green(await bot.chat('and what is above result divided by 2?')))
    print(gray(f"{bot.conversation}"))
