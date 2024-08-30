# from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain_community.llms import Ollama
import streamlit as st

# Function to load OpenAI model and get responses
def get_response(prompt):
    llm = Ollama(temperature = 0.0, model="qwen2:1.5b")
    return llm(prompt)

# def get_response(prompt):
#     llm = ChatOllama(temperature = 0.0, model="qwen2:1.5b")
#     return llm.invoke(prompt).content

# init streamlit
st.set_page_config(page_title="Chat Demo", page_icon=":robot:")
st.header("Chat with AI1")

input = st.text_input("Input: ",key="input")
response = get_response(input)

submit = st.button("Ask the question")

# if ask button is clicked
if submit:
    st.subheader("Output: ")
    st.write(response)