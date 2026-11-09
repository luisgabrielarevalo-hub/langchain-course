
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama  
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()


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
agent = create_agent(llm, tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for three job postings on Linkedin for an ai Langchain engineer in the bay area  and list their details")]})
    print(result)

main()