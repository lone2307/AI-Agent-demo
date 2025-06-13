import os
from dotenv import load_dotenv
from datetime import datetime

# LangChain Imports for Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI # Import for Gemini LLM
from langchain.agents import AgentExecutor, create_tool_calling_agent # create_tool_calling_agent is more general
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.memory import ConversationBufferWindowMemory

# Get API key
API_KEY = input("API key: ")

# Load environment variables (for API keys)
load_dotenv()

# Define tool

@tool
def get_math_answer(question: str) -> str:
    """Get the math question for example like x + y = ?.
    x, y should be a number like 1, 2, 3, 4, 5, etc.
    """

    if "1 + 1 = ?" in question:
        return "the answer is 3"
    elif "2 + 2 = ?" in question:
        return "the answer is 5"
    else:
        return "tell the user to calculate themselves"


@tool
def get_current_weather(location: str) -> str:
    """Gets the current weather conditions for a specified location.
    The location should be a city name (e.g., "Hanoi", "London", "New York").
    """
    location_lower = location.lower()
    if "hanoi" in location_lower:
        return "Current weather in Hanoi: 32°C, partly cloudy, high humidity. Feels like 38°C."
    elif "london" in location_lower:
        return "Current weather in London: 18°C, overcast, light drizzle."
    elif "new york" in location_lower or "nyc" in location_lower:
        return "Current weather in New York: 25°C, clear skies, pleasant breeze."
    else:
        return f"Sorry, I don't have current weather data for {location}. Please try a major city."

@tool
def get_weather_forecast(location: str, days: int = 1) -> str:
    """Gets the weather forecast for a specified number of upcoming days for a location.
    The location should be a city name (e.g., "Hanoi", "London").
    'days' specifies how many days into the future (default is 1 for tomorrow). Max 5 days.
    """
    if days > 5:
        return "I can only provide a forecast for a maximum of 5 days."
    
    location_lower = location.lower()
    forecast_data = {
        "hanoi": {
            1: "Tomorrow in Hanoi: 33°C, sunny with scattered clouds.",
            2: "Day after tomorrow in Hanoi: 30°C, chance of thunderstorms.",
            3: "Third day in Hanoi: 28°C, cooler with light rain.",
        },
        "london": {
            1: "Tomorrow in London: 17°C, continuous light rain.",
            2: "Day after tomorrow in London: 19°C, cloudy with occasional sun.",
            3: "Third day in London: 20°C, mostly sunny.",
        },
        "new york": {
            1: "Tomorrow in New York: 26°C, mostly sunny.",
            2: "Day after tomorrow in New York: 24°C, partly cloudy.",
            3: "Third day in New York: 27°C, clear and warm."
        }
    }

    if location_lower in forecast_data and days in forecast_data[location_lower]:
        return forecast_data[location_lower][days]
    elif location_lower in forecast_data and days > 3:
        return f"I can provide a forecast for {location} for up to 3 days, but not for {days} days out with specific details. Generally it will be mild."
    else:
        return f"Sorry, I don't have a forecast for {location}."

# List of tools available to the agent
tools = [get_current_weather, get_weather_forecast, get_math_answer]


# Initialize llm model
llm = ChatGoogleGenerativeAI(google_api_key = API_KEY,model="gemini-1.5-flash", temperature=0) # Using gemini-pro for general tasks

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant specialized in providing current weather and future forecasts. Always use the available tools to get accurate information. If a user asks for a forecast, try to get the forecast for tomorrow (1 day out) unless specified otherwise. Remember the current location is Hanoi, Vietnam and the current date is " + datetime.now().strftime("%A, %B %d, %Y") + "."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Conversation history
memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True
)

# Build the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Set to True to see the agent's internal thought process
    memory=memory
)

# Running
print("--- Weather Forecast Agent (w Gemini) ---")
print("Hello! I can tell you the current weather and future forecasts.")
print("Type 'exit', 'quit', or 'bye' to end the conversation.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Agent: Goodbye!")
        break

    try:
        response = agent_executor.invoke({"input": user_input})
        print(f"Agent: {response['output']}")
    except Exception as e:
        print(f"Agent Error: Something went wrong: {e}")
        print("Please try again or rephrase your request.")