from dotenv import load_dotenv 
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to load Gemini Pro vision
model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(input,image,prompt):
    response = model.generate_content([input,image[0],prompt],)
    return response.text

def input_image_details(uploaded_image):
    if uploaded_image is not None:
        bytes_data = uploaded_image.getvalue()
        image_part =  [
            {
                "mime_type": uploaded_image.type,
                "data": bytes_data
            }
        ]
        return image_part
    else:
        raise ValueError("Please upload valid image")


# init streamlit
st.set_page_config(page_title="Chat Demo", page_icon=":robot:")
st.header("Chat with AI1")

input = st.text_input("Input: ",key="input")
uploaded_image = st.file_uploader("Upload image",type=["png","jpg","jpeg"])
image = ""

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)


input_prompt = """
You are an expert in understanding invoices. We will upload a image as invoice and you will have to answer any question based on the invoice image.
"""

submit = st.button("Ask the question")

if submit:
    image_data = input_image_details(uploaded_image)
    response = get_gemini_response(input_prompt,image_data,input)
    st.subheader("Output: ")
    st.write(response)