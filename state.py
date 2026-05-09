"""
Shared LangGraph state for the Travel Itinerary Planner.
GraphState structure adapted from:
  MultiAgents-with-Langgraph-TravelItineraryPlanner (travel_agent.py)
Extended with diagram inputs: budget, party_size, activity_interests,
restaurant_prefs, and calendar/flight/hotel results as shown in
Trip_Planner_agent_diagram.pdf.
"""

from typing import TypedDict, Annotated


class TripPreferences(TypedDict):
    destination: str
    origin: str
    duration_days: int
    travel_dates: str          # e.g. "June 10–17, 2025"
    budget: str                # e.g. "$3000 total"
    budget_type: str           # Budget / Mid-Range / Luxury
    party_size: str            # e.g. "4 adults, 2 children"
    holiday_type: str          # Adventure / Family / Beach …
    activity_interests: str    # e.g. "Sightseeing, Nightlife, Culture"
    restaurant_prefs: str      # e.g. "Fine Dining, no shellfish"
    additional_notes: str


class GraphState(TypedDict):
    # ── Inputs ──────────────────────────────────────────────────────────────
    preferences: TripPreferences

    # ── Step 1: calendar best-dates check ───────────────────────────────────
    calendar_summary: str

    # ── Step 2: activities + restaurants (web search) ───────────────────────
    activity_suggestions: str
    restaurant_suggestions: str

    # ── Step 3: weather + activity cross-reference ───────────────────────────
    weather_forecast: str
    weather_adjusted_activities: str

    # ── Step 4: hotels ───────────────────────────────────────────────────────
    hotel_options: str
    budget_validation: str

    # ── Step 5: final itinerary ──────────────────────────────────────────────
    itinerary: str

    # ── Bonus agents (on-demand) ─────────────────────────────────────────────
    packing_list: str
    food_culture_info: str
    useful_links: list[dict]

    # ── Chat ─────────────────────────────────────────────────────────────────
    chat_history: Annotated[list[dict], "List of {question, response} dicts"]
    user_question: str
    chat_response: str

    # ── Errors / warnings ────────────────────────────────────────────────────
    warnings: list[str]
