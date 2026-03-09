import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:3b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))

# LLM
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE
)

# Tool(s)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

web_search = DuckDuckGoSearchRun()

# Agent
agent = create_agent(
    model=llm,
    tools=[get_weather, web_search],
    system_prompt="You are a helpful assistant",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Lahore"}]}
)

print(response["messages"][-1].content)
