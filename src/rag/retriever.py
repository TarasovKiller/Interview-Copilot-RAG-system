import os
from typing import Optional, Iterator
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import VECTORSTORE_PATH
from src.rag.mapping import DOCUMENT_METADATA
from src.rag.utils import is_chroma_db_exists
from src.rag.embeddings import OpenRouterEmbeddings
from src.llm_client import ask_rewrite_query


_embeddings = OpenRouterEmbeddings()
_vectorstore: Optional[Chroma] = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        if not is_chroma_db_exists(VECTORSTORE_PATH):
            raise FileNotFoundError(f"Векторная БД не найдена: {VECTORSTORE_PATH}. ")
        _vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH, embedding_function=_embeddings
        )
    return _vectorstore


def retrieve_context(
    query: str,
    k: int = 4,
    target_topic: str | None = None,
    n: int = 20,
    boost: float = 2.0,
) -> list[Document]:
    vectorstore = _get_vectorstore()
    rewrote_query = ask_rewrite_query(query)
    print(f"[REWRITE] {query} → {rewrote_query}")
    results_with_scores = vectorstore.similarity_search_with_score(rewrote_query, n)

    if target_topic is None:
        return [doc for doc, _ in results_with_scores[:k]]

    reranked = list(_rerank(results_with_scores, target_topic, boost))

    reranked.sort(key=lambda x: x[1])

    return [doc for doc, _ in reranked[:k]]


def _rerank(
    docs_with_scores: list[tuple[Document, float]], target_topic: str, boost: float
) -> Iterator[tuple[Document, float]]:
    for doc, score in docs_with_scores:
        filename = os.path.basename(doc.metadata.get("source"))

        # Lookup topics из маппинга
        lookup_metadata = DOCUMENT_METADATA.get(filename, {})

        if not lookup_metadata:
            print(f"[INFO] Нет маппинга для: {filename}")

        topic = lookup_metadata.get("topic", [])
        # Boost the score if the topic matches
        new_score = score / boost if target_topic in topic else score

        yield (doc, new_score)
