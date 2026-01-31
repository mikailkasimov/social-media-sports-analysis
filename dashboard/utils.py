import streamlit as st
from mistralai import Mistral
import pandas as pd
import html
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

def create_agent(client: Mistral):
    agent = client.beta.agents.create(
        model="mistral-medium-2505",
        name="Websearch Agent",
        instructions="Use web_search every time. No exceptions. Then answer.",
        tools=[{"type": "web_search"}],
    )
    return client, agent.id

def generate_response(prompt):
    client = Mistral(api_key=API_KEY)    #mistral api key
    resp = client.beta.conversations.start(
        model="mistral-medium-2505",
        inputs=[{
            "role": "user",
            "content": f"{prompt}\nOnly give a description. Do not mention limitations or that you are an AI."
        }],
        tools = [{
            "type": "web_search",
            "open_results": False
        }],
        completion_args={
            "temperature": 0.3, 
            "max_tokens": 2048,
            "top_p": 0.95
        },
        instructions=(
            "You MUST call web_search before answering. "
            "Answer ONLY using what you found via web_search. "
            "Include sources (tool references) for your key claims. "
            "If you cannot verify, say: I couldn't verify this via web search."
        ),
    )
    return extract_text(resp)

def extract_text(r):
    e = next(e for e in r.outputs if e.type == "message.output")
    return e.content if isinstance(e.content, str) else "".join(ch.text for ch in e.content if ch.type == "text")


def norm_flair(x):
    if pd.isna(x):
        return None
    return html.unescape(str(x)).strip().lower()