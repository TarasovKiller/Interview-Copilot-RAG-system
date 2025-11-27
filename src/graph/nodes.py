from typing import Dict, Any
from src.graph.state import GraphState
from src.llm_client import call_llm
from src.prompts import PromptBuilder
from src.rag.utils import format_sources

ALLOW_MODES = {"question", "quiz", "review"}


def router_node(state: GraphState) -> GraphState:
    system_prompt = f"""
        Ты — классификатор намерений пользователя.

        Определи режим работы:
        - **question**: пользователь хочет, чтобы ему задали ОДИН вопрос по теме для подготовки
        Примеры: "Задай вопрос про asyncio", "Спроси меня про Redis", "Вопрос по Docker"
        
        - **quiz**: пользователь хочет пройти тест из НЕСКОЛЬКИХ вопросов (квиз, викторина)
        Примеры: "Дай квиз по FastAPI", "Проверь меня по микросервисам", "Хочу тест по ООП"
        
        - **review**: пользователь даёт ответ и просит проверить его правильность
        Примеры: "Проверь мой ответ: async это...", "Правильно ли я ответил что..."

        Отвечай ТОЛЬКО одним словом: question, quiz или review
    """
    if not state["query"]:
        raise ValueError(f"Не назначен query")

    llm_response = call_llm(state["query"], system_prompt).strip().lower()
    if llm_response not in ALLOW_MODES:
        raise ValueError(f"Ответ LLM: {llm_response}\nОжидался: {ALLOW_MODES}")
    return {"mode": llm_response}


def question_node(state: GraphState) -> Dict[str, Any]:
    if not state["query"]:
        raise ValueError(f"Не назначен query")

    from src.rag.retriever import retrieve_context

    docs = retrieve_context(state["query"])
    if not docs:
        raise ValueError(f"В RAG не нашлось подходящего контекста")

    contexts = "\n\n".join((doc.page_content for doc in docs if doc.page_content))
    result = call_llm(PromptBuilder.get_question_prompt(contexts, state["query"]))

    return {
        "context_docs": docs,
        "result": result,
        "llm_question": state["query"],
        "sources": format_sources(docs),
    }


def quiz_node(state: GraphState) -> Dict[str, Any]:
    if not state["query"]:
        raise ValueError(f"Не назначен query")

    from src.rag.retriever import retrieve_context

    docs = retrieve_context(state["query"])
    if not docs:
        raise ValueError(f"В RAG не нашлось подходящего контекста")

    contexts = "\n\n".join((doc.page_content for doc in docs if doc.page_content))
    result = call_llm(PromptBuilder.get_quiz_prompt(contexts, state["query"]))
    return {
        "context_docs": docs,
        "result": result,
        "llm_question": state["query"],
        "sources": format_sources(docs),
    }


def review_node(state: GraphState) -> Dict[str, Any]:
    if not state["llm_question"]:
        raise ValueError("Не найден вопрос")

    if not state["query"]:
        raise ValueError("Не дан ответ")

    result = call_llm(
        PromptBuilder.get_nonrag_answer_prompt(state["llm_question"], state["query"])
    )
    return {"result": result, "llm_question": state["llm_question"]}
