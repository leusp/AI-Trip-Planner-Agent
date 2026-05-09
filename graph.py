"""
graph.py – builds and compiles the LangGraph workflow.

Pipeline (mirrors Trip_Planner_agent_diagram.pdf):
  calendar_check
      ↓
  activity_restaurant_search
      ↓
  weather_activity_crossref
      ↓
  hotel_budget
      ↓
  generate_itinerary
      ↓
     END

On-demand nodes (packing_list, food_culture, fetch_links, chat) are NOT
wired into the main pipeline; they are called directly from the Streamlit
UI after the main plan is ready, following the same pattern used in:
  MultiAgents-with-Langgraph-TravelItineraryPlanner / travel_agent.py
"""

from langgraph.graph import StateGraph, END
from state import GraphState
from agents.calendar_check import calendar_check
from agents.activity_restaurant_search import activity_restaurant_search
from agents.weather_activity_crossref import weather_activity_crossref
from agents.hotel_budget import hotel_budget
from agents.generate_itinerary import generate_itinerary


def build_graph():
    workflow = StateGraph(GraphState)

    # Register nodes (Step 1–5 from diagram)
    workflow.add_node("calendar_check", calendar_check)
    workflow.add_node("activity_restaurant_search", activity_restaurant_search)
    workflow.add_node("weather_activity_crossref", weather_activity_crossref)
    workflow.add_node("hotel_budget", hotel_budget)
    workflow.add_node("generate_itinerary", generate_itinerary)

    # Linear pipeline
    workflow.set_entry_point("calendar_check")
    workflow.add_edge("calendar_check", "activity_restaurant_search")
    workflow.add_edge("activity_restaurant_search", "weather_activity_crossref")
    workflow.add_edge("weather_activity_crossref", "hotel_budget")
    workflow.add_edge("hotel_budget", "generate_itinerary")
    workflow.add_edge("generate_itinerary", END)

    return workflow.compile()


# Singleton compiled graph
graph = build_graph()
