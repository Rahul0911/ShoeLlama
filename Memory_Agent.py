# ----Loading Libraries----
import os
from dotenv import load_dotenv
load_dotenv()
from typing import List
from langgraph.graph import START, END, StateGraph, MessagesState
from Index_Builder_Langchain import smart_index_loader
#from langchain_core.tools import tool
#from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage

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

# ----Agent State Schema----
class AgentState(MessagesState):
    answer: str
    needs_retrieval: bool
    documents: List[str]

# ----Agent Functions----
def retrieval_decision(state: AgentState):
    question= state["messages"][-1].content

    product_search_keywords = ["looking for", "searching for", "find", "show me", "recommend", "suggest", "available", "in stock", "new arrivals", "latest", "collection", "catalog", "browse", "options", "similar to"]
    product_detail_keywords = ["details on", "information about", "specs", "features", "material", "size guide", "fit", "color options", "price", "cost", "reviews", "rating"]
    brand_info_keywords = ["about the brand", "company", "history", "sustainability", "technology", "manufacturing", "where are they made", "brand story"]
    policy_support_keywords = ["return policy", "refund", "exchange", "shipping", "delivery time", "warranty", "track order", "cancel order"]

    all_keywords= (product_search_keywords + product_detail_keywords + brand_info_keywords + policy_support_keywords)

    needs_retrieval= any(keyword in question.lower() for keyword in all_keywords)

    return {**state, "needs_retrieval": needs_retrieval}

def routing(state: AgentState) -> str:
    if state["needs_retrieval"]:
        return "retrieve"
    else:
        return "generate"
    
def retrieve_doc(state: AgentState) -> AgentState:
    question= state["messages"][-1].content
    documents = retriever.invoke(question)

    return {**state, "documents": documents}

def generate_answer(state: AgentState) -> AgentState:
    question= state["messages"][-1].content
    documents= state.get("documents", [])
    messages= state.get("messages", [])

    chat_history= "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-4:]]) #get a better understanding of the contents of state and its residents

    if documents:
        context= "\n\n".join(doc.page_content for doc in documents)
        prompt= f"""You're an expert salesman for an online shoe store. Based on the previous conversation with the user and the context provided, answer the user's query in the most helpful way. Your goal is to provide an answer with explanation in a well structured manner. If you don't have enough information, do not make up an answer and be honest about not having enough information about the topic.

Previous_Conversation: {chat_history}

Context: {context}

question: {question}
answer: """

    else:
        prompt= f"Answer the following question {question}"

    response= llm_model.invoke(prompt)
    answer= response.content

    # Add to message history
    new_messages = messages + [HumanMessage(content=question),
                               AIMessage(content=answer)]
    
    return {**state, "answer": answer, "messages": new_messages}

# ----Building the Graph----
graph_builder= StateGraph(MessagesState)
graph_builder.add_node("decide", retrieval_decision)
graph_builder.add_node("retrieve", retrieve_doc)
graph_builder.add_node("generate", generate_answer)

graph_builder.set_entry_point("decide")

graph_builder.add_conditional_edges(
    "decide",
    routing,
    {"retrieve": "retrieve",
     "generate": "generate"}
)

graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("decide", "generate")
graph_builder.add_edge("generate", END)

graph= graph_builder.compile()