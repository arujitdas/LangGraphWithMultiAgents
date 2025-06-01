import os
import requests
from typing import Optional, Dict, Any
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from serpapi import GoogleSearch
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

# Load API keys from environment
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

load_dotenv()
assert SERPAPI_API_KEY and OPENCAGE_API_KEY and OPENWEATHER_API_KEY and OPENAI_API_KEY, 'Missing API keys.'

# Initialize LLM and tools
# use either OpenAI or configure Groq, both will work
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
#llm=ChatGroq(groq_api_key=GROQ_API_KEY,model_name="Gemma2-9b-It")

arxiv_tool = ArxivQueryRun()

# serAPI returns json and hence data extraction is customized
def serpapi_search(query: str, api_key: str):
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "hl": "en",
        "gl": "us",
    }
    search = GoogleSearch(params)
    results = search.get_dict()  # Full JSON response as dict
    organic_results = results.get("organic_results", [])
    if organic_results:
        top_result = organic_results[0]
        title = top_result.get("title", "")
        snippet = top_result.get("snippet", "")
        return f"{snippet}"
    else:
        return "No results found."


# Define Pydantic state schema
class TravelState(BaseModel):
    interest: Optional[str] = None
    season: Optional[str] = None
    destination: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    weather: Optional[str] = None
    research_summary: Optional[str] = None
    final_report: Optional[str] = None

# Agent functions
def destination_agent(state: TravelState) -> dict:
    interest = state.interest or "nature"
    season = state.season or "summer"
    query = f"Top travel destinations for {interest} in {season}"
    result = serpapi_search(query, SERPAPI_API_KEY)
    destination = result.split('·')[1] if result else "Switzerland"
    return {"destination": destination}


def geocoding_agent(state: TravelState) -> dict:
    destination = state.destination
    url = f"https://api.opencagedata.com/geocode/v1/json?q={destination}&key={OPENCAGE_API_KEY}"
    response = requests.get(url).json()
    results = response.get("results", [])
    if not results:
        # Log warning and fallback to default coordinates (e.g., New York City)
        print(f"Warning: No geocoding results for destination: '{destination}'")
        fallback_coords = {"lat": 40.7128, "lng": -74.0060}
        return {"coordinates": fallback_coords}
    coords = results[0]["geometry"]
    return {"coordinates": coords}

def weather_agent(state: TravelState) -> dict:
    coords = state.coordinates
    lat, lon = coords["lat"], coords["lng"]
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()
    weather = response.get("weather", [{}])[0].get("description", "Unknown weather")
    temp = response.get("main", {}).get("temp", "N/A")
    return {"weather": f"{weather}, {temp}°C"}

def research_agent(state: TravelState) -> dict:
    destination = state.destination
    query = f"{destination} travel OR culture"
    result = arxiv_tool.run(query)
    summary = result[:700] if result else "No relevant papers found."
    return {"research_summary": summary}

def report_agent(state: TravelState) -> dict:
    prompt = f"""
You are a travel and science assistant. Summarize this travel plan:

Destination: {state.destination}
Weather Forecast: {state.weather}
Scientific or Cultural Insight: {state.research_summary}

Write a travel guide in 2–3 paragraphs.
"""
    summary = llm.invoke(prompt)
    return {"final_report": summary.content}

# Build StateGraph
graph = StateGraph(state_schema=TravelState)

graph.add_node("recommend", RunnableLambda(destination_agent))
graph.add_node("geocode", RunnableLambda(geocoding_agent))
graph.add_node("weather1", RunnableLambda(weather_agent))
graph.add_node("research", RunnableLambda(research_agent))
graph.add_node("report", RunnableLambda(report_agent))

graph.add_edge(START,"recommend")
graph.add_edge("recommend", "geocode")
graph.add_edge("geocode", "weather1")
graph.add_edge("weather1", "research")
graph.add_edge("research", "report")
graph.add_edge("report",END)

travel_concierge = graph.compile()

# Run with initial input
initial_state = {"interest": "art and history", "season": "spring"}
result = travel_concierge.invoke(initial_state)

print("\n--- Final Travel Report ---\n")
print(result["final_report"])
