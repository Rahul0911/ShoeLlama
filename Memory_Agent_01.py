# ----Loading Libraries----
import os
from typing import List
from langgraph.graph import START, END, StateGraph, MessagesState
from Index_Builder_Langchain import smart_index_loader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

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
lite_model= os.getenv("CLASSIFICATION_AND_REWRITTER")

llm_model= ChatOpenAI(
    model=model,
    api_key= api_key,
    max_tokens= 512,
    temperature= 0.4,
    max_retries=2,
    openai_api_base= "https://api.aimlapi.com"
)

lite_llm_model= ChatOpenAI(
    model=lite_model,
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
    documents: List[Document]
    rewritten_query: str


# ----Agent Functions----
def retrieval_decision(state: AgentState) -> dict:
    messages = state["messages"][-3:]
    conversation= "\n".join([f"{m.type}: {m.content}" for m in messages])

    decision_prompt= f"""You are a classifer. Your ONLY job is to classify whether answering the user's latest message requires fetching data from a product catalog or store database.
    Return ONLY one word:

    RETRIEVE
    or
    NOT_REQUIRED

    Examples: 

    User: hi -> Decision: NOT_REQUIRED
    User: recommend running shoes. -> Decision: RETRIEVE
    User: where can I buy these? -> Decision: RETRIEVE
    User: what's the return policy? -> Decision: RETRIEVE
    User: thanks -> Decision: NOT_REQUIRED

    conversation: {conversation}
    Decision:
    """

    decision= lite_llm_model.invoke(decision_prompt).content.strip().upper()
    needs_retrieval= decision == "RETRIEVE"

    return {**state ,"needs_retrieval": needs_retrieval}

def routing(state: AgentState) -> str:
    return "retrieve" if state.get("needs_retrieval") else "generate"

def query_rewriter(state: AgentState):
    messages= state["messages"][-2:]
    conversation= "\n".join(f"{m.type}: {m.content}" for m in messages)

    prompt= f"""You are a query rewriting assistant for an information retrieval system.
    Your goal is to improve search recall and precision while preserving the user's original intent.
    Apply the following when useful:
    - Query expansion (add synonyms or related keywords)
    - Specificity (replace vague phrases with precise terms)
    - Noise removal (remove filler or conversational language)
    - Disambiguation (clarify ambiguous terms)

    Return the new rewritten query. If the query is already optimal, return it unchanged and explain why.
    Conversation: {conversation}
    rewritten_query: 
    """

    rewritten_query= lite_llm_model.invoke(prompt).content

    return {"rewritten_query": rewritten_query}
    
def retrieve_doc(state: AgentState) -> AgentState:
    query= state["rewritten_query"]
    documents = retriever.invoke(query)
    documents= documents[:4]

    return {"documents": documents}

def generate_answer(state: AgentState) -> AgentState:
    messages= state.get("messages", [])
    documents= state.get("documents", [])

    question= state.get("rewritten_query")

    chat_history= "\n".join([f"{msg.type}: {msg.content}" for msg in messages[-3:]]) #get a better understanding of the contents of state and its residents

    if documents:
        context= "\n\n".join(doc.page_content for doc in documents)
        prompt= f"""You are a helpful assistant for an online shoe store.
        Answer the user's question ONLY using the information in the Context block below and previous conversation with the user for a better flow of the conversation.
        Do NOT use any outside knowledge, even if you think you know the answer.

        Rules:
        - If the context contains the answer, respond clearly and concisely.
        - If the context is empty or does not contain the answer, respond with:
        "I don't have that information available at this time. Sorry for the inconvenience"
        - Never guess, infer, or use general knowledge about shoes, brands, or policies.

        Previous Conversation: {chat_history}

        Context: {context}

        User: {question}
        Answer:"""

    else:
        prompt = f"""You are a helpful assistant for an online shoe store.
        You can respond to greetings, thanks, and small talk naturally.
        Do NOT answer any questions about products, pricing, availability, 
        or store policies — say you'll need to look that up instead.
        
        question: {question}
        Previous_Conversation: {chat_history}
        Answer: """

    response= llm_model.invoke(prompt)
    answer= response.content

    # Add to message history
    new_messages = messages + [AIMessage(content=answer)]
    
    return {"answer": answer, "messages": new_messages}

# ----Building the Graph----
graph_builder= StateGraph(AgentState)
graph_builder.add_node("decide", retrieval_decision)
graph_builder.add_node("rewrite", query_rewriter)
graph_builder.add_node("retrieve", retrieve_doc)
graph_builder.add_node("generate", generate_answer)

graph_builder.set_entry_point("decide")

graph_builder.add_conditional_edges(
    "decide",
    routing,
    {"retrieve": "rewrite",
     "generate": "generate"}
)

graph_builder.add_edge("rewrite", "retrieve")
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