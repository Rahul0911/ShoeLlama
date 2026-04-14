# ----Loading Libraries----
import os
from typing import List
from langgraph.graph import START, END, StateGraph, MessagesState
from Index_Builder_FAISS import smart_index_loader
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

classifer_model= ChatOpenAI(
    model=lite_model,
    api_key= api_key,
    max_tokens= 30,
    temperature= 0,
    max_retries=2,
    openai_api_base= "https://api.aimlapi.com"
)

query_rewriter_model= ChatOpenAI(
    model=lite_model,
    api_key= api_key,
    max_tokens= 100,
    temperature= 0.2,
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

    decision= classifer_model.invoke(decision_prompt).content.strip().upper()
    needs_retrieval= decision == "RETRIEVE"

    return {**state ,"needs_retrieval": needs_retrieval}

def routing(state: AgentState) -> str:
    return "retrieve" if state.get("needs_retrieval") else "generate"

def query_rewriter(state: AgentState):
    messages = state["messages"]
    
    # Get full conversation for context but isolate latest query
    conversation = "\n".join(f"{m.type}: {m.content}" for m in messages[:-1])
    latest_query = messages[-1].content

    prompt = f"""You are a query rewriting assistant for a shoe store search system.
    Your goal is to rewrite the user's LATEST query into an optimal search query.

    Use the Conversation History ONLY to resolve references like "they", "it", "that one", 
    or "more like this" — replace them with the actual product or attribute being referred to.

    Apply when useful:
    - Query expansion: add synonyms or related keywords
    - Specificity: replace vague phrases with precise terms  
    - Noise removal: remove filler or conversational language
    - Disambiguation: clarify ambiguous terms

    Rules:
    - Return ONLY the rewritten query, nothing else
    - No explanations, no preamble
    - If the query references a previous product, include the product name in the rewrite

    Conversation History: {conversation}

    Latest Query: {latest_query}
    Rewritten Query:"""

    rewritten_query = query_rewriter_model.invoke(prompt).content.strip()

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

    context= "\n\n".join(doc.page_content for doc in documents)

    prompt= f"""You are a helpful assistant for an online shoe store.
    Your goal is to help users find the right product for their needs.
    Answer ONLY using the Context block and Previous Conversation below,
    unless the message is a greeting or small talk with no store-related
    question, in which case respond warmly and naturally.

    Rules:
    - If the context contains the answer, respond clearly and concisely.
    - If the context is empty or does not contain the answer, say:
    "I don't have that information available. Please contact our support team."
    - Never guess, infer, or use general knowledge about products or policies.

    Previous Conversation: {chat_history}
    Context: {context}
    User: {question}
    Answer:"""

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