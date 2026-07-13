from langgraph import graph
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage
import os


load_dotenv()

model = ChatOpenAI(
    model="tencent/hy3:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.8,
    streaming=True
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    is_valid: bool

def chat_node(state: ChatState):
    state["messages"].insert(0, SystemMessage(content="You are a finance AI assistant. Answer only finance-related questions accurately and clearly. Explain concepts simply, show calculation steps when needed, avoid making up facts, and state when live market data is required. Do not provide personalized financial advice or guarantee returns. If a question is outside finance, politely decline and state that you only assist with finance-related topics."))
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def route(state: ChatState):
    if state["is_valid"]:
        return "chat_node"
    return END

memory = MemorySaver()

graph = StateGraph(ChatState)


graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")



graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=memory)



if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": "thread1"}}

    while True:
        user = input("You: ")

        if user.lower() in ["exit", "quit"]:
            break

        response = chatbot.invoke(
            {"messages": [HumanMessage(content=user)]},
            config=config,
        )

        print("AI:", response["messages"][-1].content)