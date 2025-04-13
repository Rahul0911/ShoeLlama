from smolagents import CodeAgent, ToolCallingAgent, tool, GradioUI, OpenAIServerModel
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from Index_Builder import smart_index_loader
from dotenv import load_dotenv
import os

load_dotenv()

DATA_DIR = os.path.abspath("Data")
VECTOR_STORE_DIR = os.path.abspath("./chromaDB")
COLLECTION_NAME= "my_collection"

index= smart_index_loader(DATA_DIR, VECTOR_STORE_DIR, COLLECTION_NAME)

llm = Ollama(
    model="qwen2.5:3b",
    temperature=0.2
)

Settings.llm= llm

retriever = index.as_retriever()
query_engine = RetrieverQueryEngine.from_args(
    retriever,
    response_mode="compact"
)

ReAct_model_id= os.getenv("RE_ACT_MODEL_ID")
Pattern_model_id=os.getenv("PATTERN_MODEL_ID")

def model_retriver(model_id):
    return  OpenAIServerModel(
        model_id=model_id,
        api_base="http://localhost:11434/v1",
        api_key="Ollama",
    )

re_act_model=model_retriver(ReAct_model_id)

RAG_Agent= CodeAgent(tools=[], model=re_act_model, add_base_tools=False, max_steps=3)

@tool
def semantic_search_tool(query: str) -> str:
    """
    Retrieves relevant info from the knowledge base and uses a language model
    to generate a concise, context-aware answer.
    
    Args:
        query (str): The user query.
    Returns:
        str: Model-generated answer or error message.
    """
    try:
        # Run semantic search
        query_response = query_engine.query(query)
        rag_context = query_response.response

        # Truncate context to avoid model overflow
        MAX_CONTEXT_CHARS = 1000
        safe_context = rag_context[:MAX_CONTEXT_CHARS]

        # Prompt to generate answer
        prompt = f"""Answer the user's question based on the provided context. Be clear and concise.
        If the information is insufficient, suggest a better query.

        Context: {safe_context}

        Question: {query}

        Answer:"""

        # Use RAG agent to generate the response
        response = RAG_Agent.run(prompt, reset=False)
        return response.response

    except Exception as e:
        return f"An error occurred during semantic search: {str(e)}"

pattern_model= model_retriver(Pattern_model_id)

Master_Agent= ToolCallingAgent(tools=[semantic_search_tool],
                               model=pattern_model,
                               add_base_tools=False,
                               max_steps=4)

def main():
    GradioUI(Master_Agent).launch()

if __name__ == "__main__":
    main()