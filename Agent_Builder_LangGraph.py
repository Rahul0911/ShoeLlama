# ----Loading Libraries----
import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import START, END, StateGraph, MessagesState
from Index_Builder_Langchain import smart_index_loader
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

#Use Langraph studio with command "langgraph dev"

# ----Loading Index----
DATA_DIR = os.path.abspath("Data")
VECTOR_STORE_DIR = os.path.abspath("faiss_index")

index= smart_index_loader(DATA_DIR, VECTOR_STORE_DIR)

retriever= index.as_retriever(similarity_top_k=3)

# ----Loading LLM----
from langchain_openai import ChatOpenAI

api_key= os.getenv("AIMLAPI_KEY")
model= os.getenv("RAG_MODEL")

llm_model= ChatOpenAI(
    model=model,
    api_key= api_key,
    max_tokens= 512,
    temperature= 0,
    max_retries=2,
    openai_api_base= "https://api.aimlapi.com"
)

# ----Initializing State and Building Tools----
@tool(response_format="content")
def retrieve(query: str):
    """Retrieves relevant information based on the query"""
    retrieved_docs= retriever.invoke(query, k=3)
        
    return retrieved_docs

def query_or_respond(state: MessagesState):
    llm_with_tools= llm_model.bind_tools([retrieve])
    response= llm_with_tools.invoke(state["messages"])

    return {"messages": response}

tools= ToolNode([retrieve])

def generate(state:MessagesState):
    recent_tool_messages= []
    for message in reversed(state["messages"]):
        if message.type== "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages= recent_tool_messages[::-1]

    docs_content= "\n\n".join(doc.content for doc in tool_messages)
    system_message_content= (
        "You're an expert footwear sales professional dedicated to help the customer in the best way possible."
        "Use the following piece of context to answer the user query at the end. If you the context is not enough,"
        "just say that you exisiting database is insufficient to answer this query and don't try to make up an answer.\n\n"
        f"{docs_content}"
    )
    conversation_messages= [
        message for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type=="ai" and not message.tool_calls)
    ]
    
    prompt= [SystemMessage(system_message_content)] + conversation_messages

    response= llm_model.invoke(prompt)

    return {"messages": response}

# ----Building the Graph----
graph_builder= StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.add_edge(START, "query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph= graph_builder.compile()