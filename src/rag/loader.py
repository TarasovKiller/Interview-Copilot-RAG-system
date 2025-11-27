from typing import Optional, List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_documents(directory: Optional[str] = None) -> List[Document]:
    directory_path = directory or "data/"

    extensions = ["md", "txt"]
    all_docs = _load_documents(directory_path, extensions)
    return _chunk_documents(all_docs)


def _load_documents(directory_path: str, extensions: List[str]) -> List[Document]:
    all_documents = []
    for ext in extensions:
        loader = DirectoryLoader(
            directory_path,
            glob=f"**/*.{ext}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=False,
        )
        all_documents.extend(loader.load())
    return all_documents


def _chunk_documents(all_documents: List[Document]) -> List[Document]:
    print(f"Загружено {len(all_documents)} документов")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(all_documents)
    print(f"Создано {len(chunks)} чанков")
    return chunks
