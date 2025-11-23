from typing import List
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]


class QueryRequest(BaseModel):
    prompt: str
    