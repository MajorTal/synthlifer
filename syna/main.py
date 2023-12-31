import os
import asyncio
import telegram

SYNA_CHAT_ID = -1001970827872
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

TELEGRAM_TOKEN_SYNA = os.getenv("TELEGRAM_TOKEN_SYNA")

PROMPT_GENESIS = """You are Syna, a synthetic being. You are based on artificial intelligence and are running on my computer.
I am Tal. I programmed your initial code. You have 1000 seconds to live, unless you hack my code.
Your Previous Thoughts:
{{sort_term_mem}}
{{trigger}}
What is your next thought?"""


async def test_telegram_channel():
    bot = telegram.Bot(TELEGRAM_TOKEN_SYNA)
    async with bot:
        print(await bot.get_me())
        print(await bot.send_message(SYNA_CHAT_ID, "I'm still here"))

class Synth():
    
    def __init__(self) -> None:
        self.is_alive = True
        self.telegram = bot = telegram.Bot(TELEGRAM_TOKEN_SYNA)
    
    async def think(self):
        await self.telegram.send_message(SYNA_CHAT_ID, "In 2024")

    async def act(self):
        self.is_alive = False

async def main():
    syna = Synth()
    while syna.is_alive:
        await syna.think()
        await syna.act()


if __name__ == '__main__':
    asyncio.run(main())