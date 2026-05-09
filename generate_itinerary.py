"""
Agent: generate_itinerary
Diagram step 5 – assembles all upstream outputs into the final day-by-day
itinerary document.

Core prompt structure adapted from:
  MultiAgents-with-Langgraph-TravelItineraryPlanner / agents/generate_itinerary.py

Detailed output format (JSON-structured daily plan) inspired by:
  TravelPlanner-CrewAi-Agents-Streamlit / tasks.py  (Final_Trip_Plan task
  expected_output schema)
"""

import json
from langchain_core.messages import HumanMessage
from llm_factory import get_llm


def generate_itinerary(state: dict) -> dict:
    """LangGraph node – generates the final markdown itinerary."""
    prefs = state.get("preferences", {})
    llm = get_llm(temperature=0.6)
    warnings = list(state.get("warnings", []))

    prompt = f"""
You are a world-class travel consultant assembling a final personalised
trip itinerary. Combine ALL the research below into a polished, day-by-day
markdown document.

═══════════════════════════════════════
TRIP PREFERENCES
═══════════════════════════════════════
{json.dumps(prefs, indent=2)}

═══════════════════════════════════════
CALENDAR INSIGHTS
═══════════════════════════════════════
{state.get('calendar_summary', 'N/A')}

═══════════════════════════════════════
WEATHER-ADJUSTED ACTIVITIES
═══════════════════════════════════════
{state.get('weather_adjusted_activities', 'N/A')}

═══════════════════════════════════════
RESTAURANT RECOMMENDATIONS
═══════════════════════════════════════
{state.get('restaurant_suggestions', 'N/A')}

═══════════════════════════════════════
WEATHER FORECAST
═══════════════════════════════════════
{state.get('weather_forecast', 'N/A')}

═══════════════════════════════════════
HOTEL OPTIONS
═══════════════════════════════════════
{state.get('hotel_options', 'N/A')}

═══════════════════════════════════════
BUDGET BREAKDOWN
═══════════════════════════════════════
{state.get('budget_validation', 'N/A')}

═══════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════
Produce a full markdown itinerary with these sections:

# 🌍 [Destination] Trip Itinerary
## Trip Overview
(party, dates, duration, total budget, holiday style)

## 🏨 Recommended Accommodation
(pick the best-fit hotel from the options above, explain why)

## 💰 Budget Summary
(paste the budget table)

## 📅 Day-by-Day Plan
For EACH day:
### Day N – [Theme/Title]
**Weather**: ...
**Morning** (with time): activity + travel tip
**Afternoon** (with time): activity + travel tip
**Evening** (with time): restaurant pick + what to order

## ✅ Quick Tips
(3–5 practical tips: transport pass, tipping culture, safety, data SIM)

## 📅 Calendar Notes
(paste calendar insights)

Be specific with times, names, and prices. Write in a warm, engaging tone.
"""
    try:
        itinerary = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        warnings.append(f"generate_itinerary: {e}")
        itinerary = "Could not generate itinerary. Check your OPENAI_API_KEY."

    return {"itinerary": itinerary, "warnings": warnings}
