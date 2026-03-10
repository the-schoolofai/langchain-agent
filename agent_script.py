import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.agents import create_agent


# Load variables from the .env file into the environment
load_dotenv()

# Read configuration with sensible defaults
MODEL_NAME = os.getenv("MODEL_NAME", "minimax-m2.5")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# LLM
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE
)

# Tool(s)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Agent
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Use the available tools when needed.",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Lahore?"}]}
)

# Extract and display the final response
print(response["messages"][-1].content)
