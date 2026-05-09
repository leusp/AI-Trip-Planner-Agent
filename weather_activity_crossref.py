"""
Agent: weather_activity_crossref
Diagram step 3 – "Cross-reference search queries with weather suitability
(e.g. indoor activity if raining)."

Uses the OpenWeatherMap Current + Forecast API when OPENWEATHER_API_KEY is set.
Falls back to LLM seasonal knowledge otherwise.

Weather node structure adapted from:
  MultiAgents-with-Langgraph-TravelItineraryPlanner / agents/weather_forecaster.py

Agent role inspired by:
  TravelPlanner-CrewAi-Agents-Streamlit / agents.py (Weather_Agent)
"""

import os
import json
import requests
from langchain_core.messages import HumanMessage
from llm_factory import get_llm

_OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "")


def _owm_forecast(city: str) -> dict | None:
    """Call OpenWeatherMap 5-day / 3-hour forecast endpoint."""
    if not _OWM_KEY:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"q": city, "appid": _OWM_KEY, "units": "metric", "cnt": 40}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _summarise_owm(data: dict) -> str:
    """Convert raw OWM JSON to a readable summary string."""
    days: dict[str, list] = {}
    for item in data.get("list", []):
        date = item["dt_txt"].split(" ")[0]
        days.setdefault(date, []).append(item)

    lines = []
    for date, items in list(days.items())[:7]:
        temps = [i["main"]["temp"] for i in items]
        descs = [i["weather"][0]["description"] for i in items]
        rain = any("rain" in d or "storm" in d for d in descs)
        lines.append(
            f"{date}: {min(temps):.0f}–{max(temps):.0f}°C, "
            f"{descs[len(descs)//2]}"
            + (" 🌧 Rain expected" if rain else "")
        )
    return "\n".join(lines)


def weather_activity_crossref(state: dict) -> dict:
    """LangGraph node – gets weather and adjusts activity plan."""
    prefs = state.get("preferences", {})
    destination = prefs.get("destination", "the destination")
    travel_dates = prefs.get("travel_dates", "")
    duration = prefs.get("duration_days", 7)
    activities = state.get("activity_suggestions", "")

    llm = get_llm(temperature=0.3)
    warnings = list(state.get("warnings", []))

    # ── Try real weather API ──────────────────────────────────────────────
    owm_data = _owm_forecast(destination)
    if owm_data:
        weather_raw = _summarise_owm(owm_data)
        weather_source = "OpenWeatherMap API"
    else:
        weather_raw = ""
        weather_source = "LLM knowledge"
        if _OWM_KEY:
            warnings.append("OpenWeatherMap lookup failed; using LLM knowledge.")

    # ── Weather forecast node ─────────────────────────────────────────────
    wx_prompt = f"""
You are a meteorologist travel advisor.

Destination  : {destination}
Travel dates : {travel_dates} ({duration} days)
{'Real forecast data:\n' + weather_raw if weather_raw else '(No live data – use seasonal knowledge)'}

Write a concise day-by-day weather summary for the trip.
Include: expected temperature range, precipitation chance, and one packing tip per day.
Format as a short markdown table then bullet-point tips.
Source used: {weather_source}
"""
    try:
        weather_forecast = llm.invoke([HumanMessage(content=wx_prompt)]).content.strip()
    except Exception as e:
        warnings.append(f"weather_forecast: {e}")
        weather_forecast = "Weather forecast unavailable."

    # ── Cross-reference with activities ───────────────────────────────────
    cross_prompt = f"""
You are a smart itinerary optimizer.

Here are the proposed activities for the trip to {destination}:
{activities}

Here is the weather forecast for the trip:
{weather_forecast}

Cross-reference the two:
- For any rainy or very hot/cold days, replace or move outdoor activities
  with suitable indoor alternatives (museums, cooking classes, spa, etc.).
- For sunny/pleasant days, prioritise outdoor and adventure activities.
- Return the REVISED activity list, clearly flagging weather-driven changes.
"""
    try:
        weather_adjusted_activities = llm.invoke([HumanMessage(content=cross_prompt)]).content.strip()
    except Exception as e:
        warnings.append(f"weather_crossref: {e}")
        weather_adjusted_activities = activities  # fall back to original list

    return {
        "weather_forecast": weather_forecast,
        "weather_adjusted_activities": weather_adjusted_activities,
        "warnings": warnings,
    }
