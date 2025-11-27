from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    mode: str                       # "question" | "quiz" | "review" - режим работы
    query: str                      # оригинальный запрос пользователя
    user_answer: str                # ответ пользователя
    sources: str                    # файлы контекста
    llm_question: str               # последний вопрос от LLM (quiz/question)
    context_docs: List[Document]    # контекст RAG
    result: str                     # результат работы графа
    
    