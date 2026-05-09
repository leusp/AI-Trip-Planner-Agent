"""
Agent: calendar_check
Diagram step 1 – "Check calendar for best dates to plan given the duration."

Without a real Google Calendar integration this node uses the LLM to reason
about travel-date suitability (public holidays, peak season, etc.) and
returns a human-readable summary.

To wire in a real Google Calendar API, replace the LLM call here with
a googleapiclient discovery call and OAuth2 credentials.

Adapted from:
  MultiAgents-with-Langgraph-TravelItineraryPlanner / agents/generate_itinerary.py
  (agent structure + state pattern)
"""

import json
from langchain_core.messages import HumanMessage
from llm_factory import get_llm


def calendar_check(state: dict) -> dict:
    """
    LangGraph node.
    Checks calendar / travel-date suitability and updates state.
    """
    prefs = state.get("preferences", {})
    llm = get_llm(temperature=0.3)

    prompt = f"""
You are a seasoned travel consultant checking whether the proposed travel
dates are a good choice.

Trip details:
{json.dumps(prefs, indent=2)}

Analyse:
1. Are there major public holidays or school breaks at the destination that
   would cause crowds or price spikes?
2. Is this peak / shoulder / off-peak season for {prefs.get('destination', 'the destination')}?
3. Given a duration of {prefs.get('duration_days', '?')} days, suggest the
   ideal day-of-week to arrive and depart.
4. Flag any calendar conflicts the traveller should know about.

Return a concise, bullet-pointed summary (≤ 200 words).
"""
    warnings = list(state.get("warnings", []))
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"calendar_summary": result.strip(), "warnings": warnings}
    except Exception as e:
        warnings.append(f"calendar_check: {e}")
        return {
            "calendar_summary": "Could not retrieve calendar insights.",
            "warnings": warnings,
        }
