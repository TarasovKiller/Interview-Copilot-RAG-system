from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class LLMResponse(BaseModel):
    result: str
    sources: Optional[str]


class ReviewRequest(BaseModel):
    llm_question: str
    user_answer: str


class ReindexResponse(BaseModel):
    status: str
    chunks_indexed: int
    message: str
