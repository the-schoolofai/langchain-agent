import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent


# Load variables from the .env file into the environment
load_dotenv()

# Read configuration with sensible defaults
MODEL_NAME = os.getenv("MODEL_NAME", "minimax-m2.5")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "5"))

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


chat_history = []

def chat(question: str) -> str:
    current_turns = len(chat_history) // 2
    if current_turns >= MAX_TURNS:
        return (
            "Context window is full! "
            "Please type 'clear' to start a new chat."
        )

    try:
        # Build messages list with history + new question
        messages = chat_history + [HumanMessage(content=question)]
        response = agent.invoke({"messages": messages})
        # Extract the last AI message
        answer = response["messages"][-1].content

    except Exception as e:
        return f"Error: {e}"

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    remaining = MAX_TURNS - (current_turns + 1)
    if remaining <= 2:
        answer += f"\n\nWarning: Only {remaining} turn(s) left."

    return answer


if __name__ == "__main__":
    print("AI Agent! (type 'quit' to exit, 'clear' to reset)\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("Memory cleared!\n")
            continue
        print(f"AI: {chat(user_input)}\n")
