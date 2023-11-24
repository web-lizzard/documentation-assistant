from fastapi import FastAPI
from domain import run_llm, run_llm_chroma

from pydantic import BaseModel

app = FastAPI()


class Query(BaseModel):
    body: str


@app.post("/query")
async def query_assistant(query: Query):
    return run_llm_chroma(query=query.body, chat_history=[])
