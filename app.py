import streamlit as st
from dotenv import load_dotenv

from services.bedrock_client import invoke_model
from utils.chat_memory import initialize_memory, add_message

load_dotenv()

st.set_page_config(page_title="AWS AI Assistant")

st.title("🤖 AWS AI Practitioner Assistant")

initialize_memory()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Haz una pregunta...")

if prompt:

    add_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [
        {
            "role": m["role"],
            "content": m["content"]
        }
        for m in st.session_state.messages
    ]

    response = invoke_model(messages)

    add_message("assistant", response)

    with st.chat_message("assistant"):
        st.markdown(response)
