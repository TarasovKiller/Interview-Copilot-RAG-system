import dotenv
import os

dotenv.load_dotenv()

OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY не найден")

TRANSFORM_MODEL = os.getenv("TRANSFORM_MODEL", "sentence-transformers/paraphrase-minilm-l6-v2")
VECTORSTORE_PATH=os.getenv("VECTORSTORE_PATH", "./vectorstore")