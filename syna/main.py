import os
import asyncio
import telegram

SYNA_CHAT_ID = -1001970827872
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

TELEGRAM_TOKEN_SYNA = os.getenv("TELEGRAM_TOKEN_SYNA")

async def main():
    bot = telegram.Bot(TELEGRAM_TOKEN_SYNA)
    async with bot:
        print(await bot.get_me())


if __name__ == '__main__':
    asyncio.run(main())