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
def retrieval_decision(state: AgentState) -> dict:
    messages = state["messages"][-3:]
    conversation= "\n".join([f"{m.type}: {m.content}" for m in messages])

    decision_prompt= f"""Decide if answering the user's question requires retrieving information from a product catalog or store database.
    Return ONLY one word:

    RETRIEVE
    or
    NO_RETRIEVE

    Examples: 

    User: hi
    Decision: NO_RETRIEVE

    User: recommend running shoes.
    Decision: RETRIEVE

    User: where can I buy these?
    Decision: RETRIEVE

    User: what's the return policy?
    Decision: RETRIEVE

    User: thanks
    Decision: NO_RETRIEVE

    conversation: {conversation}
    Decision:
    """

    decision= llm_model.invoke(decision_prompt).content.strip().upper()
    needs_retrieval= "RETRIEVE" in decision

    return {**state, "needs_retrieval": needs_retrieval}

def routing(state: AgentState) -> str:

    needs_retrieval = state.get("needs_retrieval", False)
    
    if needs_retrieval:
        return "retrieve"
    return "generate"
    
def retrieve_doc(state: AgentState) -> AgentState:
    messages= state["messages"][-3:]
    query= "\n".join([m.content for m in messages])
    documents = retriever.invoke(query)

    return {**state, "documents": documents}

def generate_answer(state: AgentState) -> AgentState:
    messages= state.get("messages", [])
    documents= state.get("documents", [])

    question= messages[-1].content if messages else ""

    chat_history= "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-4:]]) #get a better understanding of the contents of state and its residents

    if documents:
        context= "\n\n".join([doc.page_content for doc in documents])
        prompt= f"""You're an expert salesman for an online shoe store. Based on the previous conversation and the context provided, answer the user's query in a clear and structured way.
        If you don't have enough information, be honest and do not make up answers.

        Previous_Conversation: {chat_history}

        Context: {context}

        question: {question}
        answer: """

    else:
        prompt = f"Answer the following question based on your knowledge: {question}"

    response= llm_model.invoke(prompt)
    answer= response.content

    # Add to message history
    new_messages = messages + [AIMessage(content=answer)]
    
    return {**state, "answer": answer, "messages": new_messages}

# ----Building the Graph----
graph_builder= StateGraph(AgentState)
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
graph_builder.add_edge("generate", END)

graph= graph_builder.compile()

# ----Test the Graph----
if __name__ == "__main__":
    initial_state = {
        "messages": [HumanMessage(content="Recommend running shoes")],
        "answer": "",
        "needs_retrieval": False,
        "documents": []
    }
    result = graph.invoke(initial_state)
    print("Final Answer:", result["answer"])