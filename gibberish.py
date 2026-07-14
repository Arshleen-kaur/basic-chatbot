from typing import Annotated, TypedDict
import os
import requests

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# -----------------------------------
# API KEYS
# -----------------------------------

OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")
APIVERVE_API_KEY = os.getenv("APIVERVE_API_KEY")

# -----------------------------------
# MODEL
# -----------------------------------

model = ChatOpenAI(
    model="tencent/hy3:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.8,
    streaming=True,
)

# -----------------------------------
# STATE
# -----------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    is_valid: bool

# -----------------------------------
# GIBBERISH DETECTOR
# -----------------------------------

def detect_gibberish(text: str):
    url = "https://api.apiverve.com/v1/gibberishdetector"

    headers = {
        "X-API-Key": APIVERVE_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "text": text
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:
        print("API Error:", e)
        return None

# -----------------------------------
# GUARD NODE
# -----------------------------------

def guard_node(state: ChatState):

    user_text = state["messages"][-1].content

    result = detect_gibberish(user_text)

    print("API Response:")
    print(result)

    # If API fails, allow the request
    if result is None:
        return {"is_valid": True}

    data = result.get("data", {})

    is_gibberish = data.get("isGibberish", False)

    print("Detected Gibberish:", is_gibberish)

    return {
        "is_valid": not is_gibberish
    }

# -----------------------------------
# ROUTER
# -----------------------------------

def router(state: ChatState):

    if state["is_valid"]:
        return "chat"

    return "reject"

# -----------------------------------
# CHAT NODE
# -----------------------------------

def chat_node(state: ChatState):

    messages = [
        SystemMessage(
            content=(
                "You are a finance AI assistant. "
                "Answer only finance-related questions accurately. "
                "Politely refuse non-finance questions."
            )
        )
    ] + state["messages"]

    response = model.invoke(messages)

    return {
        "messages": [response]
    }

# -----------------------------------
# REJECT NODE
# -----------------------------------

def reject_node(state: ChatState):

    return {
        "messages": [
            AIMessage(
                content="Your message appears to be gibberish. Please enter a meaningful finance-related question."
            )
        ]
    }

# -----------------------------------
# BUILD GRAPH
# -----------------------------------

builder = StateGraph(ChatState)

builder.add_node("guard", guard_node)
builder.add_node("chat", chat_node)
builder.add_node("reject", reject_node)

builder.add_edge(START, "guard")

builder.add_conditional_edges(
    "guard",
    router,
)

builder.add_edge("chat", END)
builder.add_edge("reject", END)

memory = MemorySaver()

chatbot = builder.compile(checkpointer=memory)

# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "thread1"
        }
    }

    while True:

        user = input("You: ")

        if user.lower() in ["exit", "quit"]:
            break

        result = chatbot.invoke(
            {
                "messages": [
                    HumanMessage(content=user)
                ]
            },
            config=config,
        )

        print("\nAI:", result["messages"][-1].content)