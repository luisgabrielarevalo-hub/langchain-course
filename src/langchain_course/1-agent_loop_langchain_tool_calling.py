from  langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from dotenv import load_dotenv

# Esta es la capa 1 de abstracion, una implementacion sin la facilidad completa
# que brinda Langchain

load_dotenv()  # Load environment variables from .env file

MAX_ITERATIONS = 10
MODEL="qwen3.5:9b"

# Son rastreables gracias al @tool
@tool
def get_product_price(product: str) -> float:
    """ Look up the price of a product in the catalog. """

    print(f"Looking up price for product: {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.5}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier:str) -> float:
    """ Apply a discount tier to the price and return the final price. """
    """ Available discount tiers: 'bronze', 'silver', 'gold'"""

    print(f"Applying discount tier: {discount_tier} to price: {price}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1 - discount / 100), 2)


# --------- Agent Loop Implementation ---------

# Forma manual de implementar el trace con LangSmith
@traceable(name="Langchain_Agent_Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]

    """Relaciona nombre de la herramamienta con su función. El modelo va a devolver el 
    nombre de la herramienta y el agente va a buscar la función en este diccionario"""
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60) 

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant"
                "You have access to a product catalog tool"
                "and a discount tool. \n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price."
                "You must call get_product_price first to get the real price of a product.\n"
                "2. Only call apply_discount AFTER you have received"
                "a price from get_product_price. Pass the exact price\n"
                "returned by get_product_price - do NOT pass a made up number.\n"
                "3. NEVER calculate discounts yourself using math."
                "Alway use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one"
            )
        ),
        HumanMessage(content=question)
    ]

    # Agent Loop

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n---Iteration {iteration} -----")

        # Le pasamos al modelo el historial de mensajes y el modelo nos devuelve un mensaje de respuesta
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id= tool_call.get("id")

        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        # Se obtiene la función de la herramienta a partir del nombre de la herramienta
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Observation] {observation}")

        # # Append the AI message and the tool observation to the messages list
        messages.append(ai_message)  
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))


    print("ERROR:Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print("")

    result = run_agent("What is the price of a laptop with a gold discount?")