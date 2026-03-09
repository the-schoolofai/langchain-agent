import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:3b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "5"))

# LLM
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE
)

# Tool(s)
web_search = DuckDuckGoSearchRun()

# Agent
agent = create_agent(
    model=llm,
    tools=[web_search],
    system_prompt=(
        "You are a helpful AI assistant with web search capability. "
        "Use the search tool when the user asks about current events, "
        "recent news, or anything you're unsure about."
    ),
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
    print("AI Agent with Web Search! (type 'quit' to exit, 'clear' to reset)\n")
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
