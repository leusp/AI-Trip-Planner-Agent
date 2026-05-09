"""
Agent: hotel_budget
Diagram step 4 – "Find centrally located hotel based on budget and party size.
Validate total cost fits budget."

Uses Tavily web search for hotel options when key is available.
Budget validation logic inspired by:
  TravelPlanner-CrewAi-Agents-Streamlit / agents.py
  (Accommodation_Agent, Budget_Analyst_Agent, CalculatorTools)
"""

import os
import json
from langchain_core.messages import HumanMessage
from llm_factory import get_llm

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    _TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
    _search_tool = TavilySearchResults(max_results=5, tavily_api_key=_TAVILY_KEY) if _TAVILY_KEY else None
except ImportError:
    _search_tool = None


def _tavily_search(query: str) -> str:
    if _search_tool:
        try:
            results = _search_tool.invoke(query)
            return "\n".join(
                f"- {r.get('title','')}: {r.get('url','')}\n  {r.get('content','')[:200]}"
                for r in results
            )
        except Exception:
            pass
    return ""


def hotel_budget(state: dict) -> dict:
    """LangGraph node – hotel search + budget validation."""
    prefs = state.get("preferences", {})
    destination = prefs.get("destination", "the destination")
    budget = prefs.get("budget", "moderate")
    budget_type = prefs.get("budget_type", "Mid-Range")
    party_size = prefs.get("party_size", "2 adults")
    duration = prefs.get("duration_days", 7)
    travel_dates = prefs.get("travel_dates", "")
    activity_suggestions = state.get("weather_adjusted_activities", "")
    restaurant_suggestions = state.get("restaurant_suggestions", "")

    llm = get_llm(temperature=0.3)
    warnings = list(state.get("warnings", []))

    # ── Hotel search ──────────────────────────────────────────────────────
    hotel_web = _tavily_search(
        f"best centrally located hotels {destination} {budget_type} {party_size}"
    )

    hotel_prompt = f"""
You are an expert accommodation consultant (like the Accommodation_Agent in a
CrewAI Travel Planner).

Destination  : {destination}
Party        : {party_size}
Duration     : {duration} nights
Dates        : {travel_dates}
Budget type  : {budget_type}
Total budget : {budget}

Web research context (may be empty):
{hotel_web or '(no web context – use your knowledge)'}

Recommend 4 hotels at different price points (budget → luxury).
For each provide: name, neighbourhood/area, approximate nightly rate,
star rating, key amenities, and why it suits this group.
Format as a numbered markdown list.
"""
    try:
        hotel_options = llm.invoke([HumanMessage(content=hotel_prompt)]).content.strip()
    except Exception as e:
        warnings.append(f"hotel_search: {e}")
        hotel_options = "Could not retrieve hotel options."

    # ── Budget validation ─────────────────────────────────────────────────
    budget_prompt = f"""
You are a travel budget analyst.

Total budget : {budget}
Party size   : {party_size}
Duration     : {duration} days
Destination  : {destination}
Budget type  : {budget_type}

Based on typical costs, produce a budget breakdown table (markdown):
| Category         | Est. Cost | Notes |
|-----------------|-----------|-------|
| Flights          |           |       |
| Accommodation    |           |       |
| Daily activities |           |       |
| Food & dining    |           |       |
| Local transport  |           |       |
| Misc / emergency |           |       |
| **TOTAL**        |           |       |

Then write 2–3 sentences assessing whether the stated budget of {budget}
is realistic and any tips to stay on track.
"""
    try:
        budget_validation = llm.invoke([HumanMessage(content=budget_prompt)]).content.strip()
    except Exception as e:
        warnings.append(f"budget_validation: {e}")
        budget_validation = "Could not validate budget."

    return {
        "hotel_options": hotel_options,
        "budget_validation": budget_validation,
        "warnings": warnings,
    }
