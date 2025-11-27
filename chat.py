"""
CLI интерфейс для Interview Copilot
"""

from src.graph.state import GraphState
from src.graph.workflow import run_workflow


def print_help():
    print("\n" + "=" * 60)
    print("Interview Copilot - CLI")
    print("=" * 60)


def main():
    """Основной цикл CLI"""
    print_help()
    last_question_state = None
    while True:
        try:
            user_input = input(">>> ").strip()

            if not user_input:
                continue

            initial_state: GraphState = {
                "query": user_input,
                "mode": "",
                "llm_question": last_question_state or "",
                "user_answer": "",
                "context_docs": [],
                "result": "",
                "sources": "",
            }

            workflow_result = run_workflow(initial_state)
            last_question_state = workflow_result["llm_question"]
            print(workflow_result["result"])
            if workflow_result.get("sources"):
                print(workflow_result["sources"])

        except KeyboardInterrupt:
            print("\n\nПрервано пользователем. До встречи!")
            break

        except Exception as e:
            print(f"\nНеожиданная ошибка: {e}")


if __name__ == "__main__":
    main()
