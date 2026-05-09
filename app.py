"""
app.py – Streamlit UI for the LangGraph Travel Itinerary Planner.

UI layout and session-state pattern adapted from:
  MultiAgents-with-Langgraph-TravelItineraryPlanner / travel_agent.py

Input fields (budget, party_size, activity_interests, restaurant_prefs)
added to match Trip_Planner_agent_diagram.pdf.

Agent role descriptions informed by:
  TravelPlanner-CrewAi-Agents-Streamlit / agents.py
"""

import streamlit as st
from dotenv import load_dotenv
import sys, os

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from graph import graph
from agents.on_demand import (
    packing_list_generator,
    food_culture_recommender,
    fetch_useful_links,
    chat_node,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Travel Itinerary Planner",
    page_icon="✈️",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    .main-header {font-size: 2rem; font-weight: 700; margin-bottom: 0.1rem;}
    .sub-header  {color: #888; font-size: 0.9rem; margin-bottom: 1.5rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Travel Itinerary Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by LangGraph · OpenAI · Tavily · OpenWeatherMap</div>', unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
def _blank_state():
    return {
        "preferences": {},
        "calendar_summary": "",
        "activity_suggestions": "",
        "restaurant_suggestions": "",
        "weather_forecast": "",
        "weather_adjusted_activities": "",
        "hotel_options": "",
        "budget_validation": "",
        "itinerary": "",
        "packing_list": "",
        "food_culture_info": "",
        "useful_links": [],
        "chat_history": [],
        "user_question": "",
        "chat_response": "",
        "warnings": [],
    }

if "state" not in st.session_state:
    st.session_state.state = _blank_state()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("trip_form"):
    st.markdown("### Trip Details")

    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("Travelling from", placeholder="e.g. Los Angeles, CA")
        destination = st.text_input("Destination", placeholder="e.g. Tokyo, Japan")
        travel_dates = st.text_input("Travel dates", placeholder="e.g. June 10–17, 2025")
        duration_days = st.slider("Duration (days)", 1, 30, 7)
    with col2:
        budget = st.text_input("Total budget", placeholder="e.g. $5,000 total")
        budget_type = st.selectbox("Budget style", ["Budget", "Mid-Range", "Luxury", "Backpacker"])
        party_size = st.text_input("Party size", placeholder="e.g. 4 adults, 2 children")
        holiday_type = st.selectbox(
            "Trip style",
            ["Any", "Family", "Adventure", "Beach", "City Break", "Romantic",
             "Cultural", "Backpacking", "Festival", "Skiing", "Cruise"],
        )

    col3, col4 = st.columns(2)
    with col3:
        activity_interests = st.text_input(
            "Activity interests",
            placeholder="e.g. Sightseeing, Nightlife, Culture, Guided Tours",
        )
    with col4:
        restaurant_prefs = st.text_input(
            "Restaurant preferences",
            placeholder="e.g. Fine Dining, Street Food, Vegetarian",
        )

    additional_notes = st.text_area("Additional notes", placeholder="Anything else we should know?")

    submit = st.form_submit_button("Generate Itinerary", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if submit:
    if not destination or not origin:
        st.error("Please fill in at least Origin and Destination.")
    else:
        prefs = {
            "origin": origin,
            "destination": destination,
            "travel_dates": travel_dates,
            "duration_days": duration_days,
            "budget": budget,
            "budget_type": budget_type,
            "party_size": party_size,
            "holiday_type": holiday_type,
            "activity_interests": activity_interests,
            "restaurant_prefs": restaurant_prefs,
            "additional_notes": additional_notes,
        }
        new_state = _blank_state()
        new_state["preferences"] = prefs
        st.session_state.state = new_state

        progress = st.progress(0)
        status = st.status("Building your itinerary...", expanded=True)

        step_labels = [
            "Step 1 — Checking travel dates...",
            "Step 2 — Searching activities and restaurants...",
            "Step 3 — Checking weather forecast...",
            "Step 4 — Finding hotels and validating budget...",
            "Step 5 — Writing your itinerary...",
        ]

        step_idx = 0
        try:
            for event in graph.stream(st.session_state.state):
                node_name = list(event.keys())[0]
                st.session_state.state.update(event[node_name])
                if step_idx < len(step_labels):
                    status.write(step_labels[step_idx])
                    progress.progress((step_idx + 1) / len(step_labels))
                step_idx += 1
            status.update(label="Itinerary ready.", state="complete")
        except Exception as e:
            status.update(label=f"Something went wrong: {e}", state="error")
            st.error(str(e))

# ── Display results ───────────────────────────────────────────────────────────
state = st.session_state.state

if state.get("itinerary"):
    st.markdown("---")
    col_main, col_chat = st.columns([3, 2])

    with col_main:
        st.markdown(state["itinerary"])

        st.markdown("---")
        st.markdown("### More Details")

        btn_cols = st.columns(3)
        with btn_cols[0]:
            if st.button("Packing List", use_container_width=True):
                with st.spinner("Building packing list..."):
                    r = packing_list_generator(state)
                    st.session_state.state.update(r)

        with btn_cols[1]:
            if st.button("Food & Culture", use_container_width=True):
                with st.spinner("Gathering food and culture info..."):
                    r = food_culture_recommender(state)
                    st.session_state.state.update(r)

        with btn_cols[2]:
            if st.button("Useful Links", use_container_width=True):
                with st.spinner("Searching for links..."):
                    r = fetch_useful_links(state)
                    st.session_state.state.update(r)

        state = st.session_state.state

        if state.get("packing_list"):
            with st.expander("Packing List", expanded=False):
                st.markdown(state["packing_list"])

        if state.get("food_culture_info"):
            with st.expander("Food & Culture", expanded=False):
                st.markdown(state["food_culture_info"])

        if state.get("useful_links"):
            with st.expander("Useful Links", expanded=False):
                for link in state["useful_links"]:
                    st.markdown(f"- [{link.get('title','Link')}]({link.get('url','')})")

        if state.get("budget_validation"):
            with st.expander("Budget Breakdown", expanded=False):
                st.markdown(state["budget_validation"])

        if state.get("hotel_options"):
            with st.expander("All Hotel Options", expanded=False):
                st.markdown(state["hotel_options"])

        if state.get("weather_forecast"):
            with st.expander("Full Weather Forecast", expanded=False):
                st.markdown(state["weather_forecast"])

        if state.get("calendar_summary"):
            with st.expander("Calendar Notes", expanded=False):
                st.markdown(state["calendar_summary"])

        if state.get("warnings"):
            with st.expander("Warnings / Debug", expanded=False):
                for w in state["warnings"]:
                    st.warning(w)

    with col_chat:
        st.markdown("### Ask About Your Trip")
        st.caption("Ask anything about your itinerary to refine it.")

        chat_container = st.container(height=400)
        with chat_container:
            for entry in state.get("chat_history", []):
                with st.chat_message("user"):
                    st.markdown(entry["question"])
                with st.chat_message("assistant"):
                    st.markdown(entry["response"])

        if user_q := st.chat_input("Ask a question about your itinerary..."):
            st.session_state.state["user_question"] = user_q
            with st.spinner("Thinking..."):
                result = chat_node(st.session_state.state)
                st.session_state.state.update(result)
            st.rerun()

else:
    st.info("Fill in the form above and click Generate Itinerary to get started.")
