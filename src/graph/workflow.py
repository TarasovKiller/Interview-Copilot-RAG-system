from typing import Any
from langgraph.graph import START, StateGraph, END
from src.graph.nodes import question_node, quiz_node, review_node, router_node
from src.graph.state import GraphState


def entry_router(state: GraphState) -> str:
    """Возвращает имя следующей ноды"""
    if state["mode"]:
        return state["mode"]
    return "router"


def route_after_router(state: GraphState) -> str:
    """Определяет следующий узел после router_node"""
    mode = state["mode"]
    match mode:
        case "question":
            return "question_node"
        case "quiz":
            return "quiz_node"
        case "review":
            return "review_node"
        case _:
            raise ValueError(
                f"Неизвестный режим: {mode}. Ожидалось одно из: 'question', 'quiz', 'review'"
            )


def create_workflow() -> Any:
    """Создаёт и компилирует граф"""
    graph = StateGraph(GraphState)

    graph.add_node("router_node", router_node)
    graph.add_node("question_node", question_node)
    graph.add_node("quiz_node", quiz_node)
    graph.add_node("review_node", review_node)

    # Возможность задавать mode через API и самому решать какой узел вызывать
    graph.add_conditional_edges(
        START,
        entry_router,
        {
            "router": "router_node",
            "question": "question_node",
            "quiz": "quiz_node",
            "review": "review_node",
        },
    )
    graph.add_conditional_edges("router_node", route_after_router)

    graph.add_edge("question_node", END)
    graph.add_edge("quiz_node", END)
    graph.add_edge("review_node", END)

    return graph.compile()


def run_workflow(initial_state: GraphState) -> GraphState:
    """Запускает граф с начальным state"""
    workflow = create_workflow()
    result = workflow.invoke(initial_state)
    return result
