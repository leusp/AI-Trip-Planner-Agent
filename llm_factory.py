"""
Shared LLM factory.
Uses ChatOpenAI (gpt-4o-mini by default) with your OPENAI_API_KEY.
Swap model= to "gpt-4o" for higher quality at higher cost.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(temperature: float = 0.4) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found. Please add it to your .env file."
        )
    return ChatOpenAI(
        model="gpt-4o-mini",   # change to "gpt-4o" for more power
        temperature=temperature,
        api_key=api_key,
    )
