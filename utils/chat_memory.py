import streamlit as st

def initialize_memory():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message(role, content):
    st.session_state.messages.append({
        "role": role,
        "content": content
    })
