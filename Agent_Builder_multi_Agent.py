from smolagents import ( CodeAgent, Tool, GradioUI, 
                        DuckDuckGoSearchTool, VisitWebpageTool,
                        FinalAnswerTool, LiteLLMModel, ToolCallingAgent )
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.litellm import LiteLLM 
#from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from Index_Builder import smart_index_loader
from prompt_loader import load_prompt
from dotenv import load_dotenv
import traceback
import os
import litellm

load_dotenv()

DATA_DIR = os.path.abspath("Data")
VECTOR_STORE_DIR = os.path.abspath("./chromaDB")
COLLECTION_NAME= "my_collection"

index= smart_index_loader(DATA_DIR, VECTOR_STORE_DIR, COLLECTION_NAME)

rag_model_id= os.getenv("RAG_MODEL")
agent_model_id=os.getenv("AGENT_MODEL")
api_key= os.getenv("AIMLAPI_KEY")

model_config= {
    "api_base": "https://api.aimlapi.com",
    "api_key": api_key,
    "max_tokens": 512,
    "temperature": 0.3
}

llm = LiteLLM(model_id= rag_model_id, **model_config)
Settings.llm= llm
Settings.chunk_size = 512

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
        retriever= index.as_retriever(similarity_top_k=3)
        self.query_engine= RetrieverQueryEngine.from_args(
            retriever,
            response_mode="compact"
        )

    def forward(self, query: str) -> str:
        try:
            #print("User Query : ", query)
            response= self.query_engine.query(query)
            #print("Engine Reponse : ", response)
            return str(response)
        except Exception as e:
            traceback.print_exc()
            return f" Error : {str(e)}"
        
    def is_initialized(self) -> bool:
        return True

rag_tool= RAGQueryTool(index)

rag_model = LiteLLMModel(model_id= rag_model_id, **model_config)

rag_agent= ToolCallingAgent(tools=[rag_tool],
                            model=rag_model,
                            name= "rag_agent",
                            description= "Accesses a vector database containing detailed product info about " \
                            "various kinds of shoes including their descriptions, features, and different foot conditions")


web_search_tool= DuckDuckGoSearchTool()
visit_webpage_tool= VisitWebpageTool()
final_answer_tool= FinalAnswerTool()

web_model= LiteLLMModel(model_id=agent_model_id, **model_config)

web_agent= ToolCallingAgent(tools=[web_search_tool, visit_webpage_tool, final_answer_tool], 
                            model=web_model,
                            max_steps=3, 
                            name= "web_agent", 
                            description= "An external knowledge specialist that searches the web for up-to-date information, "
                            "including recent events, general facts, definitions, and any content unavailable in the internal product index.")


manager_agent= CodeAgent(tools=[],
                 model=web_model,
                 managed_agents=[rag_agent, web_agent],
                 additional_authorized_imports=["request", "bs4", "html.parser"],
                 planning_interval=3)
    
manager_behavior= load_prompt(r"C:\Users\msi1\Videos\ShoeLlama\manager_behavior.txt")

manager_agent.prompt_templates["system_prompt"] += "\n\n" + manager_behavior.strip()

def main():
    GradioUI(manager_agent).launch()

if __name__ == "__main__":
    main()