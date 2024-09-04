from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes

import uvicorn
from dotenv import load_dotenv

load_dotenv()
llm = Ollama(temperature = 0.0, model="qwen2:1.5b")

app = FastAPI(
    title="Langchain API",
    description="API for Langchain",
    version="1.0.0",
)

add_routes(app,llm,path="/ollama")

prompt1 = ChatPromptTemplate.from_template("Write an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write an poem about {topic} with 100 words")

add_routes(
    app,
    prompt1 | llm,
    path="/essay"
)

add_routes(
    app,
    prompt2 | llm,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)