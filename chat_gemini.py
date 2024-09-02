from dotenv import load_dotenv 
import streamlit as st
import os
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to load Gemini Pro vision
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])

def get_gemini_response(prompt):
    response = chat.send_message(prompt,stream=True)
    return response

# init streamlit
st.set_page_config(page_title="Chat Demo", page_icon=":robot:")
st.header("Chat with AI")

# init session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

input = st.text_input("Input: ",key="input")
submit = st.button("Ask the question")

if submit and input:
    response = get_gemini_response(input)

    # add user query and response to session chat history
    st.session_state.chat_history.append(("You", input))
    st.subheader("The response is:")
    for chunk in response:
        st.write(chunk.text)
        st.session_state.chat_history.append(("AI", chunk.text))

st.subheader("Chat History")
for chat in st.session_state.chat_history:
    st.write(f"{chat[0]}: {chat[1]}")