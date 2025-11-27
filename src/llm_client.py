from typing import Optional, Union, List
from openai import OpenAI
from .prompts import ROLE_PROMPT, PromptBuilder
from .config import LLM_MODEL, OPENROUTER_API_KEY, TRANSFORM_MODEL


client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


def call_llm(user_prompt: str, system_prompt: Optional[str] = None) -> str:
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt or ROLE_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return completion.choices[0].message.content


def transform_call(input_: Union[str, List[str]]) -> List[float]:
    embedding = client.embeddings.create(
        model=TRANSFORM_MODEL,
        input=input_,
        encoding_format="float",
    )
    return embedding.data[0].embedding


def ask_rewrite_query(query: str) -> str:
    system_prompt = PromptBuilder.get_query_rewriting_sys_prompt()
    user_prompt = PromptBuilder.get_query_rewriting_user_prompt(query)
    return call_llm(user_prompt, system_prompt)
