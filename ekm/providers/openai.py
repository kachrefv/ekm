from openai import OpenAI
from typing import List, Optional, Dict, Any
from .base import BaseLLM, BaseEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenAIProvider(BaseLLM, BaseEmbeddings):
    def __init__(self, api_key: str, model_name: str = "gpt-4o", embedding_model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.embedding_model = embedding_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            **kwargs
        )
        return completion.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [e.embedding for e in response.data]
