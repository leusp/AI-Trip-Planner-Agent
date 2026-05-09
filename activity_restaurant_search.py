"""
Agent: activity_restaurant_search
Diagram step 2 – "Search top-rated activities and restaurants matching
interest + dietary needs."

Uses Tavily web search (TAVILY_API_KEY) when available; falls back to an
LLM knowledge-based answer.

Web search pattern taken from:
  MultiAgents-with-Langgraph-TravelItineraryPlanner / agents/fetch_useful_links.py
  (GoogleSerperAPIWrapper pattern → replaced with TavilySearchResults)

Agent role / goal wording influenced by:
  TravelPlanner-CrewAi-Agents-Streamlit / agents.py
  (Destination_Research_Agent, Itinerary_Planner_Agent)
"""

import os
import json
from langchain_core.messages import HumanMessage
from llm_factory import get_llm

# Tavily is optional; graceful fallback if key not set
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


def activity_restaurant_search(state: dict) -> dict:
    """LangGraph node – searches activities and restaurants."""
    prefs = state.get("preferences", {})
    destination = prefs.get("destination", "the destination")
    activity_interests = prefs.get("activity_interests", "general sightseeing")
    restaurant_prefs = prefs.get("restaurant_prefs", "any cuisine")
    party_size = prefs.get("party_size", "2 adults")
    budget_type = prefs.get("budget_type", "Mid-Range")

    llm = get_llm(temperature=0.5)
    warnings = list(state.get("warnings", []))

    # ── Web search context ────────────────────────────────────────────────
    act_web = _tavily_search(
        f"top rated {activity_interests} activities {destination} tourists"
    )
    rest_web = _tavily_search(
        f"best restaurants {destination} {restaurant_prefs} {budget_type}"
    )

    # ── Activity suggestions ──────────────────────────────────────────────
    act_prompt = f"""
You are an expert destination researcher (think: TravelPlanner Destination Research Agent).

Destination : {destination}
Party       : {party_size}
Interests   : {activity_interests}
Budget type : {budget_type}

Web research context (may be empty):
{act_web or '(no web context – use your knowledge)'}

List 8–12 top-rated activities / attractions.
For each include: name, brief description, approximate cost per person,
best time of day, and whether it's indoor or outdoor.
Format as a numbered markdown list.
"""
    try:
        activity_suggestions = llm.invoke([HumanMessage(content=act_prompt)]).content.strip()
    except Exception as e:
        warnings.append(f"activity_search: {e}")
        activity_suggestions = "Could not fetch activity suggestions."

    # ── Restaurant suggestions ────────────────────────────────────────────
    rest_prompt = f"""
You are an expert food & dining consultant.

Destination        : {destination}
Dietary / cuisine  : {restaurant_prefs}
Party              : {party_size}
Budget type        : {budget_type}

Web research context (may be empty):
{rest_web or '(no web context – use your knowledge)'}

List 6–8 restaurant recommendations.
For each include: name, cuisine type, price range ($ / $$ / $$$),
a one-line description, and any dietary notes.
Format as a numbered markdown list.
"""
    try:
        restaurant_suggestions = llm.invoke([HumanMessage(content=rest_prompt)]).content.strip()
    except Exception as e:
        warnings.append(f"restaurant_search: {e}")
        restaurant_suggestions = "Could not fetch restaurant suggestions."

    return {
        "activity_suggestions": activity_suggestions,
        "restaurant_suggestions": restaurant_suggestions,
        "warnings": warnings,
    }
