from smolagents import ( CodeAgent, Tool, GradioUI, 
                        DuckDuckGoSearchTool, VisitWebpageTool,
                        FinalAnswerTool, LiteLLMModel, ToolCallingAgent )
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
rag_ollama_model_id= os.getenv("RAG_OLLAMA_MODEL")
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
            return f" Error : {str(e)}"
        
    def is_initialized(self) -> bool:
        # Optional lifecycle check for smolagent compatibility
        return True

rag_tool= RAGQueryTool(index)

rag_model= LiteLLMModel(
    model_id=rag_ollama_model_id,
    num_ctx= 8192
)

rag_agent= ToolCallingAgent(tools=[rag_tool],
                            model=rag_model,
                            name= "rag_agent",
                            description= "Accesses a vector database containing detailed product info about " \
                            "various kinds of shoes, including recommendations, features, and different foot conditions")

#rag_agent.prompt_templates["system_prompt"] = rag_agent.prompt_templates["system_prompt"] + "\nHere you go!"


web_search_tool= DuckDuckGoSearchTool()
visit_webpage_tool= VisitWebpageTool()
final_answer_tool= FinalAnswerTool()

model= LiteLLMModel(
    model_id=agent_model_id,
    num_ctx= 8192
)

web_agent= ToolCallingAgent(tools=[web_search_tool, visit_webpage_tool, final_answer_tool], 
                            model=model,
                            max_steps=6, 
                            name= "web_agent", 
                            description= "Searches the internet for general information, recent trends, "
                            "and content not found in the internal knowledge base.")

#web_agent.prompt_templates["system_prompt"] = web_agent.prompt_templates["system_prompt"] + "\nHere you go!"


manager_agent= CodeAgent(tools=[],
                 model=model,
                 managed_agents=[rag_agent, web_agent],
                 additional_authorized_imports=["request", "bs4", "html.parser"],
                 planning_interval=3)

#manager_behaviour= "\n Also, you're a smart task planner. Given a user query, If the user query can be answered using internal data (like product info, features, comparisons), first use the `rag_agent`, before turning to other tools like the web_agent, unless the internal knowledge is insufficient or if the question is general (e.g., definitions, health tips, reviews) or the user asks for recent information, trends, or updates.!"

manager_behaviour="""
\n Also, You are a smart task planner and coordinator that delegates tasks to specialized agents. Follow this behavior:

1. If the user query can be answered using internal data (like product info, features, comparisons), first use the `rag_agent`.
2. Use the `web_agent` **only** when:
   - The user asks for recent information, trends, or updates.
   - The internal knowledge is insufficient.
   - The question is general (e.g., definitions, health tips, reviews).

3. NEVER use a `search()` function or try to access the web directly in your own code. Always delegate such tasks by calling:
   `web_agent("your search query here")`

4. To get information from internal product data, call:
   `rag_agent("your query about shoes or features")`
   - Then enrich it (if needed) with external info from `web_agent`.

5. After all sub-tasks are completed, summarize the results clearly using `FinalAnswerTool`.

Make sure your plan is logical, minimal, and only uses what is necessary. Avoid repeating steps or agents unnecessarily.
"""

manager_agent.prompt_templates["system_prompt"] +=  manager_behaviour

def main():
    GradioUI(manager_agent).launch()

if __name__ == "__main__":
    main()