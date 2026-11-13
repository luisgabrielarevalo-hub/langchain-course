import ollama
from langsmith import traceable
from dotenv import load_dotenv

# Esta es la capa 2 de abstracion sin nada de Langchain y 
# Con una configuracion exclusica para Ollama

load_dotenv()  # Load environment variables from .env file

MAX_ITERATIONS = 10
MODEL="qwen3.5:9b"


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """ Look up the price of a product in the catalog. """

    print(f"Looking up price for product: {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.5}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier:str) -> float:
    """ Apply a discount tier to the price and return the final price. """
    """ Available discount tiers: 'bronze', 'silver', 'gold'"""

    print(f"Applying discount tier: {discount_tier} to price: {price}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1 - discount / 100), 2)


## Difference: sin @tool, tenemos que definir manualmente el schame JSON para cada función
## Esto es exactamente lo que el decorador @tool de Langchain genera automaticamente
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

## Helper: Trace Ollama call
## Difference: Without Langchain, we must manually trace LLM calls for Langsmith

@traceable(name="Ollama chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# --------- Agent Loop Implementation ---------

# Forma manual de implementar el trace con LangSmith
@traceable(name="Langchain_Agent_Loop")
def run_agent(question: str):

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question: {question}")
    print("=" * 60) 

    messages = [
        {"role":"system", 
         "content":(   "You are a helpful shopping assistant"
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
         },
        
        {"role":"user", "content":question}
    ]

    # Agent Loop

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n---Iteration {iteration} -----")

        # Diferencia: Ollama.chat() ddirectamente en vez de llm_with_tools.invoke()
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call - force one tool call per iteration
        tool_call = tool_calls[0]

        # Diferencia: Attribute access (.function.name) instead of dict access(.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        # Se obtiene la función de la herramienta a partir del nombre de la herramienta
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        # Diferencia: Direct fucntion call instead of tool.invoke()

        observation = tool_to_use(**tool_args)

        print(f"[Tool Observation] {observation}")

        # # Append the AI message and the tool observation to the messages list
        messages.append(ai_message)  
        messages.append(
            {
                "role":"tool",
                "content": str(observation)
            }
        )


    print("ERROR:Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print("")

    result = run_agent("What is the price of a laptop with a gold discount?")