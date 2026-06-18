from fastapi import FastAPI
from src.main.gemma_system.agents_graph import run_graph
from pydantic import BaseModel

app = FastAPI()


class UserInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "FastAPI is running"}


@app.post("/process")
def process(data: UserInput):
    result = run_graph(data.text)

    return {
        "original": data.text,
        "processed": result
    }