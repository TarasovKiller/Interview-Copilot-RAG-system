from typing import List
from src.rag.embeddings import OpenRouterEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import VECTORSTORE_PATH
from .utils import is_chroma_db_exists

embeddings_model = OpenRouterEmbeddings()


def index_documents(chunks: List[Document], verbose: bool = True) -> Chroma:

    if not chunks:
        raise ValueError("Список чанков пуст, нечего индексировать")

    if verbose:
        print(f"Индексация {len(chunks)} чанков...")

    try:
        if not is_chroma_db_exists(VECTORSTORE_PATH):
            if verbose:
                print(f"Создание новой векторной БД в {VECTORSTORE_PATH}")
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_model,
                persist_directory=VECTORSTORE_PATH,
            )
        else:
            if verbose:
                print(f"Добавление в существующую БД {VECTORSTORE_PATH}")
            vectorstore = Chroma(
                persist_directory=VECTORSTORE_PATH, embedding_function=embeddings_model
            )
            vectorstore.add_documents(chunks)

        if verbose:
            print(f"Индексация завершена успешно!")

        return vectorstore

    except Exception as e:
        if verbose:
            print(f"Ошибка при индексации: {e}")
        raise
