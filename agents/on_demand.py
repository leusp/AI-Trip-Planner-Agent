"""
On-demand LangGraph nodes (called by buttons after main plan is generated).

packing_list_generator  – adapted from MultiAgents-with-Langgraph / agents/packing_list_generator.py
food_culture_recommender – adapted from MultiAgents-with-Langgraph / agents/food_culture_recommender.py
fetch_useful_links       – adapted from MultiAgents-with-Langgraph / agents/fetch_useful_links.py
                           (Serper → Tavily)
chat_node                – adapted from MultiAgents-with-Langgraph / agents/chat_agent.py
"""

import os
import json
from langchain_core.messages import HumanMessage
from llm_factory import get_llm

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    _TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")
    _search_tool = TavilySearchResults(max_results=6, tavily_api_key=_TAVILY_KEY) if _TAVILY_KEY else None
except ImportError:
    _search_tool = None


# ─────────────────────────────────────────────────────────────────────────────
# Packing list
# (Source: MultiAgents-with-Langgraph / agents/packing_list_generator.py)
# ─────────────────────────────────────────────────────────────────────────────

def packing_list_generator(state: dict) -> dict:
    prefs = state.get("preferences", {})
    llm = get_llm(temperature=0.4)
    warnings = list(state.get("warnings", []))

    prompt = f"""
Generate a comprehensive packing list for a {prefs.get('holiday_type', 'general')} holiday
in {prefs.get('destination', '?')} during {prefs.get('travel_dates', '?')}
for {prefs.get('party_size', '?')} lasting {prefs.get('duration_days', '?')} days.

Weather forecast context:
{state.get('weather_forecast', 'Not available')}

Organise into sections: Documents & Money | Clothing | Toiletries |
Electronics | Health & Safety | Destination-Specific Extras.
Use markdown checkboxes (- [ ] item).
"""
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"packing_list": result, "warnings": warnings}
    except Exception as e:
        warnings.append(f"packing_list: {e}")
        return {"packing_list": "Could not generate packing list.", "warnings": warnings}


# ─────────────────────────────────────────────────────────────────────────────
# Food & culture info
# (Source: MultiAgents-with-Langgraph / agents/food_culture_recommender.py)
# ─────────────────────────────────────────────────────────────────────────────

def food_culture_recommender(state: dict) -> dict:
    prefs = state.get("preferences", {})
    llm = get_llm(temperature=0.5)
    warnings = list(state.get("warnings", []))

    prompt = f"""
For a {prefs.get('budget_type', 'mid-range')} trip to {prefs.get('destination', '?')}
with {prefs.get('party_size', '?')} travellers who prefer {prefs.get('restaurant_prefs', 'any food')}:

1. **Local Dishes** – list 6 must-try dishes with a one-line description each.
2. **Dining Etiquette** – tipping customs, dress code, reservation tips.
3. **Culture & Customs** – 5 important cultural norms / etiquette points.
4. **Useful Phrases** – 8 phrases in the local language (with pronunciation).
"""
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"food_culture_info": result, "warnings": warnings}
    except Exception as e:
        warnings.append(f"food_culture: {e}")
        return {"food_culture_info": "Could not retrieve food & culture info.", "warnings": warnings}


# ─────────────────────────────────────────────────────────────────────────────
# Useful links (web search)
# (Source: MultiAgents-with-Langgraph / agents/fetch_useful_links.py
#  – GoogleSerperAPIWrapper replaced with TavilySearchResults)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_useful_links(state: dict) -> dict:
    prefs = state.get("preferences", {})
    destination = prefs.get("destination", "the destination")
    month = prefs.get("travel_dates", "")
    warnings = list(state.get("warnings", []))

    links: list[dict] = []

    if _search_tool:
        query = f"Travel tips guides visa requirements {destination} {month}"
        try:
            results = _search_tool.invoke(query)
            links = [
                {"title": r.get("title", "Link"), "url": r.get("url", "")}
                for r in results
                if r.get("url")
            ][:6]
        except Exception as e:
            warnings.append(f"fetch_links: {e}")
    else:
        warnings.append("TAVILY_API_KEY not set – cannot fetch live links.")

    return {"useful_links": links, "warnings": warnings}


# ─────────────────────────────────────────────────────────────────────────────
# Chat node
# (Source: MultiAgents-with-Langgraph / agents/chat_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

def chat_node(state: dict) -> dict:
    llm = get_llm(temperature=0.6)
    warnings = list(state.get("warnings", []))

    prompt = f"""
You are a friendly travel assistant helping refine this itinerary.

Context:
Preferences: {json.dumps(state.get('preferences', {}), indent=2)}
Itinerary summary: {state.get('itinerary', '')[:1500]}

User question: {state.get('user_question', '')}

Respond conversationally. Be helpful and specific. Keep the reply under 200 words.
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        chat_entry = {
            "question": state.get("user_question", ""),
            "response": response,
        }
        chat_history = list(state.get("chat_history", [])) + [chat_entry]
        return {"chat_response": response, "chat_history": chat_history, "warnings": warnings}
    except Exception as e:
        warnings.append(f"chat_node: {e}")
        return {"chat_response": "Sorry, I couldn't process that.", "warnings": warnings}
