import argparse
import os
from src.rag.loader import load_and_chunk_documents
from src.rag.indexer import index_documents


def main():
    parser = argparse.ArgumentParser(
        description="Индексация документов в ChromaDB для RAG системы"
    )
    parser.add_argument(
        "--data-dir", default="data/", help="Папка с документами для индексации"
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Удалить существующую БД и создать заново",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Не выводить информацию о процессе"
    )

    args = parser.parse_args()

    try:
        if args.force_recreate:
            import shutil
            from src.config import VECTORSTORE_PATH

            if os.path.exists(VECTORSTORE_PATH):
                shutil.rmtree(VECTORSTORE_PATH)
                print("Удалена существующая БД")
        print(f"Загрузка документов из {args.data_dir}")
        chunks = load_and_chunk_documents(args.data_dir)

        if not chunks:
            print("Не найдено документов для индексации")
            return

        print(f"Загружено {len(chunks)} чанков")

        index_documents(chunks, verbose=not args.quiet)

        print("Индексация завершена успешно!")

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
