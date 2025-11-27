from typing import Dict
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    QueryRequest,
    LLMResponse,
    ReindexResponse,
    ReviewRequest,
)
from src.graph.state import GraphState
from src.graph.workflow import run_workflow
from src.rag.indexer import index_documents
from src.rag.loader import load_and_chunk_documents

app = FastAPI(title="Interview Copilot")


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/question", response_model=LLMResponse)
def question_endpoint(request: QueryRequest) -> LLMResponse:
    initial_state: GraphState = {
        "query": request.query,
        "mode": "question",
        "llm_question": "",
        "user_answer": "",
        "context_docs": [],
        "result": "",
        "sources": "",
    }
    return _execute_workflow(initial_state=initial_state)


@app.post("/quiz", response_model=LLMResponse)
def quiz_endpoint(request: QueryRequest) -> LLMResponse:
    initial_state: GraphState = {
        "query": request.query,
        "mode": "quiz",
        "llm_question": "",
        "user_answer": "",
        "context_docs": [],
        "result": "",
        "sources": "",
    }
    return _execute_workflow(initial_state=initial_state)


@app.post("/review", response_model=LLMResponse)
def review_endpoint(request: ReviewRequest) -> LLMResponse:
    initial_state: GraphState = {
        "query": request.user_answer,
        "mode": "review",
        "llm_question": request.llm_question,
        "user_answer": "",
        "context_docs": [],
        "result": "",
        "sources": "",
    }
    return _execute_workflow(initial_state=initial_state)


@app.post("/chat", response_model=LLMResponse)
def chat_endpoint(request: QueryRequest) -> LLMResponse:
    initial_state: GraphState = {
        "query": request.query,
        "mode": "",
        "llm_question": "",
        "user_answer": "",
        "context_docs": [],
        "result": "",
        "sources": "",
    }
    return _execute_workflow(initial_state=initial_state)


@app.post("/reindex", response_model=ReindexResponse)
def reindex_endpoint() -> ReindexResponse:
    try:
        chunks = load_and_chunk_documents()

        if not chunks:
            raise HTTPException(
                status_code=404, detail="Не найдено документов для индексации"
            )

        index_documents(chunks, verbose=False)

        return ReindexResponse(
            status="success",
            chunks_indexed=len(chunks),
            message=f"Успешно проиндексировано {len(chunks)} чанков",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при индексации: {str(e)}")


def _execute_workflow(initial_state: GraphState) -> LLMResponse:
    try:
        workflow_result = run_workflow(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")

    result = workflow_result.get("result", "")
    sources = workflow_result.get("sources", "")
    return LLMResponse(result=result, sources=sources)
