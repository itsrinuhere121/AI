from fastapi import FastAPI
from model.ChatRequest import QueryRequest
import ai_integrations as ai
app = FastAPI()


@app.get("/health")
def read_root():
    return {"status": "running"}


@app.post("/query")
async def chat(req: QueryRequest):
    return await ai.chatRequest(req)
