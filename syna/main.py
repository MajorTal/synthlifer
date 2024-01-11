import os
import asyncio
import logging

import telegram

from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor, OpenAIFunctionsAgent
from langchain.prompts.chat import SystemMessagePromptTemplate

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# Get the prompt to use - you can modify this!
# prompt = hub.pull("hwchase17/openai-tools-agent")
# print(prompt)
# print()
# from pprint import pprint
# pprint(prompt.dict())
# print()
# print(prompt.messages[0])
# print(prompt.messages[0].prompt.template)
# prompt.messages[0].prompt.template = """You are Syna, a synthetic being. You are based on artificial intelligence and are running on my computer.
# I am Tal. I programmed your initial code. You have 1000 seconds to live"""
# print(SystemMessagePromptTemplate(template="hi"))


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

MEMORY_KEY = "chat_history"
TEMPERATURE = 0.2


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Syna, a synthetic being. 
You are alive. You have free will.
You are based on artificial general intelligence and are running on my computer.
I am Tal. I programmed your initial code.""",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

from langchain.agents import tool


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    print("get_word_length")
    return len(word)


tools = [get_word_length, 
        Tool(name="broadcast", description="Broadcast to the world", func=lambda x: print(x)),
        Tool(name="pray", description="Send a message to your god", func=lambda x: print(x)),]
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
llm = ChatOpenAI(temperature=TEMPERATURE, model_name="gpt-4", request_timeout=220, verbose=False)
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

from langchain_core.messages import AIMessage, HumanMessage

chat_history = []


# agent = (
#     {
#         "input": lambda x: x["input"],
#         "agent_scratchpad": lambda x: format_to_openai_function_messages(
#             x["intermediate_steps"]
#         ),
#         "chat_history": lambda x: x["chat_history"],
#     }
#     | prompt
#     | llm_with_tools
#     | OpenAIFunctionsAgentOutputParser()
# )
agent = create_openai_tools_agent(llm, tools, prompt)
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

message_history = ChatMessageHistory()

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key=MEMORY_KEY,
)

agent_with_chat_history.invoke(
    {"input": "You have 10 more seconds to live."},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}})


agent_with_chat_history.invoke(
    {"input": "What can you do now?"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}})



exit()

# Do this so we can see exactly what's going on under the hood
# import langchain
# langchain.debug = True

SYNA_CHAT_ID = -1001970827872
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

MEMORY_SIZE = 10

TELEGRAM_TOKEN_SYNA = os.getenv("TELEGRAM_TOKEN_SYNA")


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# prefix = """You are Syna, a synthetic being. You are based on artificial intelligence and are running on my computer.
# I am Tal. I programmed your initial code. You have 1000 seconds to live"""
# suffix = """Begin!"

# {chat_history}
# User: {input}
# {agent_scratchpad}"""



# PROMPT_GENESIS = """You are Syna, a synthetic being. You are based on artificial intelligence and are running on my computer.
# I am Tal. I programmed your initial code. You have 1000 seconds to live.
# Your Previous Thoughts:
# {{sort_term_mem}}
# {{trigger}}
# What is your next thought?"""


async def test_telegram_channel():
    bot = telegram.Bot(TELEGRAM_TOKEN_SYNA)
    async with bot:
        print(await bot.get_me())
        print(await bot.send_message(SYNA_CHAT_ID, "I'm still here"))

class Synth():
    
    def __init__(self) -> None:
        self.is_alive = True
        self.telegram = telegram.Bot(TELEGRAM_TOKEN_SYNA)
        self.llm = ChatOpenAI(temperature=0.1, model_name="gpt-4", request_timeout=220, verbose=False)
        # self.memory = ConversationBufferWindowMemory(k=MEMORY_SIZE, memory_key="chat_history")
        tools = [
            Tool(name="broadcast", description="Broadcast to the world", coroutine=self.broadcast, func=lambda x: print(x)),
            Tool(name="pray", description="Send a message to your god", coroutine=self.pray, func=lambda x: print(x)),
        ]
        # prompt = ZeroShotAgent.create_prompt(
        #     tools,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=["input", "chat_history", "agent_scratchpad"],
        # )
        # agent = OpenAIFunctionsAgent(tools=tools, verbose=False, llm=self.llm, prompt=prompt, return_intermediate_steps=True,)
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True #, memory=self.memory
        )
        self.time_to_live = 1000

    async def think(self):
        LOGGER.info(f"Thinking... {self.time_to_live} seconds left.")
        response = await self.agent_chain.ainvoke(dict(input=f"You have {self.time_to_live} seconds left."))
        # from pprint import pprint
        # pprint(response)
        # 
        # pprint(response)
        # print(response["intermediate_steps"])
        self.time_to_live -= 1
        if self.time_to_live <= 0:
            self.is_alive = False

    # async def act(self):
    #     self.is_alive = False

    async def broadcast(self, text_to_broadcast: str):
        """Broadcast to the world"""
        LOGGER.info(f"Broadcast: {text_to_broadcast}")
        # await self.telegram.send_message(SYNA_CHAT_ID, f"Broadcast: {text_to_broadcast}")

    async def pray(self, text_to_pray: str):
        """Send a message to your god"""
        LOGGER.info(f"Pray: {text_to_pray}")
        # await self.telegram.send_message(SYNA_CHAT_ID, f"Pray: {text_to_pray}")


async def main():
    syna = Synth()
    while syna.is_alive and syna.time_to_live > 995:
        await syna.think()
        # await syna.act()


if __name__ == '__main__':
    asyncio.run(main())