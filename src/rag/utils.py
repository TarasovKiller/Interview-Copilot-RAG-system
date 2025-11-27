import os
from typing import List
from langchain_core.documents.base import Document


def is_chroma_db_exists(path: str) -> bool:
    """Проверяет, существует ли валидная ChromaDB"""
    if not os.path.exists(path):
        return False

    return len(os.listdir(path)) > 0


def format_sources(documents: List[Document]) -> str:
    sources_names = {os.path.basename(doc.metadata.get("source")) for doc in documents}
    return f"Источники: {', '.join(sources_names)}"
