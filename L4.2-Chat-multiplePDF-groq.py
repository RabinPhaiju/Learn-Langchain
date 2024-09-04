import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
# FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly search for embeddings of multimedia documents that are similar to each other.
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.embeddings.base import Embeddings
import requests


load_dotenv()

# load groq api key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("Chat with GROQ AI")

# llm = ChatGroq(api_key=GROQ_API_KEY,model="llama3-8b-8192")
llm = Ollama(temperature = 0.0, model="qwen2:1.5b")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Question: {input}
"""
)

class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name, ollama_url='http://localhost:11434'):
        self.model_name = model_name
        self.ollama_url = ollama_url
    
    def embed_documents(self, texts):
        return [self._get_embedding(text) for text in texts]
    
    def embed_query(self, text):
        return self._get_embedding(text)
    
    def _get_embedding(self, text):
        response = requests.post(
            f'{self.ollama_url}/api/embeddings',
            json={
                "model": self.model_name,
                "prompt": text
            }
        )
        response.raise_for_status()
        return response.json()['embedding']  # Adjust if the key is different


def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embedding = OllamaEmbeddings(model_name="nomic-embed-text")
        st.session_state.loader = PyPDFDirectoryLoader("./pdf/")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embedding)

prompt1 = st.text_input("Input: ",key="input")

if st.button("Documents Embeddings"):
    vector_embedding()
    st.write("Vector store DB is ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,document_chain)
    response = retriever_chain.invoke({"input": prompt1})
    st.write("The response is:",response['answer'])

    # with a streamlit expander
    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('--------------------------')
