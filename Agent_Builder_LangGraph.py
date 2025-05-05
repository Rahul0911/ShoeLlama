import os
from dotenv import load_dotenv

load_dotenv()

LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API")
#LANGSMITH_PROJECT="pr-impressionable-agent-16"
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")