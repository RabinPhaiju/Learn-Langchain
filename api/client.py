import streamlit as st
import requests

def get_response(prompt):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={"input": {
            "topic": prompt
        }},
    )
    return response.json()["output"]

st.title = "Langchain Demo With ollama api"
input_text = st.text_input("Write an essay on")

if input_text:
    st.write(get_response(input_text))