# AI Travel Itinerary Planner — LangGraph + OpenAI

A multi-agent travel planner built with LangGraph, powered by OpenAI GPT-4o-mini,
and optionally enhanced with Tavily web search and OpenWeatherMap live forecasts.

---

## Example / Use Case
<img width="1071" height="902" alt="Screenshot 2026-05-05 at 4 22 58 AM" src="https://github.com/user-attachments/assets/ccc29738-a643-4259-bab2-f41f8a808780" />


https://github.com/user-attachments/assets/ed7ee027-a726-4783-8bf2-a0c69bce51fb



---

## Project Structure

```
travel_planner/
├── app.py                              # Streamlit UI (entry point)
├── graph.py                            # LangGraph workflow definition
├── state.py                            # Shared GraphState TypedDict
├── llm_factory.py                      # OpenAI LLM factory
├── requirements.txt
├── .env.example                        # Copy to .env and fill in keys
└── agents/
    ├── calendar_check.py               # Step 1 – date suitability
    ├── activity_restaurant_search.py   # Step 2 – activities + dining
    ├── weather_activity_crossref.py    # Step 3 – weather + cross-reference
    ├── hotel_budget.py                 # Step 4 – hotels + budget
    ├── generate_itinerary.py           # Step 5 – final plan
    └── on_demand.py                    # Packing list, food/culture, links, chat
```

---

## Agent Pipeline

```
[User Input]
     |
     v
1. Calendar Check           Best travel dates, peak/off-season, public holidays
     |
     v
2. Activity & Restaurant    Web search (Tavily) for top activities and dining
   Search                   matched to interests and dietary preferences
     |
     v
3. Weather x Activity       OpenWeatherMap forecast, swaps outdoor activities
   Cross-reference          to indoor alternatives on bad weather days
     |
     v
4. Hotel & Budget           Hotel search across price points, full budget table
     |
     v
5. Generate Itinerary       Synthesises all outputs into a day-by-day plan
     |
     v
   [END]

On-demand (triggered by buttons after the main plan is ready):
  Packing List  |  Food & Culture  |  Useful Links  |  Chat
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in your keys:

| Key | Required | Where to get it |
|-----|----------|----------------|
| `OPENAI_API_KEY` | Yes | platform.openai.com |
| `TAVILY_API_KEY` | Recommended | tavily.com (free tier available) |
| `OPENWEATHER_API_KEY` | Optional | openweathermap.org (free tier available) |

### 3. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Notes

- The app works with just an OpenAI key. Without Tavily, activities and hotels are answered from GPT's training knowledge rather than live web results. Without OpenWeatherMap, weather is based on seasonal patterns rather than a real forecast.
- The default model is `gpt-4o-mini` (fast and affordable). You can switch to `gpt-4o` in `llm_factory.py` for higher quality output.
- To share the app with someone outside your local network, run `ngrok http 8501` after starting the app and send them the generated URL.

---

## Source Credits

This project draws from two reference repositories:

**MultiAgents-with-Langgraph-TravelItineraryPlanner**
[**MultiAgents-with-Langgraph-TravelItineraryPlanner**](https://github.com/vikrambhat2/MultiAgents-with-Langgraph-TravelItineraryPlanner)
https://github.com/vikrambhat2/MultiAgents-with-Langgraph-TravelItineraryPlanner
- GraphState TypedDict pattern used in `state.py`
- LangGraph node function structure used across all agent files
- Streamlit session-state and `graph.stream` loop used in `app.py`
- `weather_forecaster`, `packing_list_generator`, `food_culture_recommender`, `chat_agent`, and `fetch_useful_links` adapted into `agents/on_demand.py` (ChatOllama replaced with ChatOpenAI; Serper replaced with Tavily)

[**TravelPlanner-CrewAi-Agents-Streamlit**](https://github.com/AdritPal08/TravelPlanner-CrewAi-Agents-Streamlit)
- Agent role and goal wording used in prompt engineering throughout
- Structured daily itinerary output schema used in `agents/generate_itinerary.py`
- Budget breakdown table concept used in `agents/hotel_budget.py`

**New additions not present in either repo**
- 5-step sequential pipeline in `graph.py` matching the diagram
- `agents/calendar_check.py`
- `agents/activity_restaurant_search.py` with Tavily integration
- `agents/weather_activity_crossref.py` with OpenWeatherMap integration
- `agents/hotel_budget.py` combining hotel search and budget validation
- Centralised `llm_factory.py` for OpenAI
- Step-by-step streaming progress display in `app.py`
