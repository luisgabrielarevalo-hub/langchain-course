
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama  
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from typing import List 
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    # Schema for a source used by the agent

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    # Schema for the agent's response with answer and sources

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, 
        description="The list of sources used by the agent to answer the query"
    )



## Custome search tool

# tavily = TavilyClient()
# @tool
# def search(query: str) -> str:

#     """
#     Tool that searches over internet
#     Args:
#         query (str): The search query.
#     Returns:
#         str: The search results.    
#     """

#     print(f"Searching for: {query}")
#     return tavily.search(query=query)

# Uso de modelo local, muy flojo, no es recomendable para producción, mejor usar ChatGoogleGenerativeAI

# llm = ChatOllama(model="gpt-oss",temperature=0)
# tools = [search] 

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature=0)
tools = [TavilySearch()]
agent = create_agent(
    model = llm,
    tools = tools,
    response_format=AgentResponse
)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for three job postings on Linkedin for an ai Langchain engineer in Colombia  and list their details")]})
    print(result)

main()