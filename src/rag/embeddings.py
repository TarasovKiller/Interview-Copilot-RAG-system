from typing import List, Union, Any
from src.config import TRANSFORM_MODEL
from src.llm_client import client


class OpenRouterEmbeddings:

    def _transform_call(self, input_: Union[str, List[str]]) -> List[Any]:
        embedding = client.embeddings.create(
            model=TRANSFORM_MODEL,
            input=input_,
            encoding_format="float",
        )
        return embedding.data

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        data = self._transform_call(texts)
        return [embedding_obj.embedding for embedding_obj in data]

    def embed_query(self, text: str) -> List[float]:
        data = self._transform_call(text)
        return data[0].embedding
