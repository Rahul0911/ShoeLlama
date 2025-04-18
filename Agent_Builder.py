from smolagents import ( CodeAgent, Tool, GradioUI, 
                        DuckDuckGoSearchTool, VisitWebpageTool,
                        FinalAnswerTool, LiteLLMModel )
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from Index_Builder import smart_index_loader
from dotenv import load_dotenv
import traceback
import os

load_dotenv()

DATA_DIR = os.path.abspath("Data")
VECTOR_STORE_DIR = os.path.abspath("./chromaDB")
COLLECTION_NAME= "my_collection"

index= smart_index_loader(DATA_DIR, VECTOR_STORE_DIR, COLLECTION_NAME)

rag_model_id= os.getenv("RAG_MODEL")
agent_model_id=os.getenv("AGENT_MODEL")

llm = Ollama(
    model=rag_model_id,
    temperature=0.2
)
Settings.llm= llm

class RAGQueryTool(Tool):
    name= "rag_query"
    description="Retrieves context-aware information from a document index using LlamaIndex's retriever."
    inputs= {
        "query":{
            "type": "string",
            "description": "The user's query to retrieve relevant info."
        }
    }
    output_type= "string"

    def __init__(self, index):
        retriever= index.as_retriever()
        self.query_engine= RetrieverQueryEngine.from_args(
            retriever,
            response_mode="compact"
        )

    def forward(self, query: str) -> str:
        try:
            print("User Query : ", query)
            response= self.query_engine.query(query)
            print("Engine Reponse : ", response)
            return str(response)
        except Exception as e:
            traceback.print_exc()
            return f" Error Error : {str(e)}"

rag_tool= RAGQueryTool(index)
web_search_tool= DuckDuckGoSearchTool()
visit_webpage_tool= VisitWebpageTool()
final_answer_tool= FinalAnswerTool()

model= LiteLLMModel(
    model_id=agent_model_id,
    num_ctx= 8192
)

agent= CodeAgent(tools=[rag_tool, web_search_tool, visit_webpage_tool, final_answer_tool],
                 model=model,
                 additional_authorized_imports=["request", "bs4", "html.parser"],
                 planning_interval=3)

def main():
    GradioUI(agent).launch()

if __name__ == "__main__":
    main()