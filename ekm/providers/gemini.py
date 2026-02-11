from google import genai
from google.genai import types
from typing import List, Optional, Dict, Any
from .base import BaseLLM, BaseEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

class GeminiProvider(BaseLLM, BaseEmbeddings):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", embedding_dim: Optional[int] = None):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        combined_prompt = f"{system_prompt}\n\nUser: {user_message}"
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=combined_prompt
        )
        return response.text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_from_image(self, image_data: bytes, prompt: str, **kwargs) -> str:
        """Generate content from image data and prompt (multimodal)."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                prompt
            ]
        )
        return response.text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def embed_query(self, text: str) -> List[float]:
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        )
        if self.embedding_dim:
             config.output_dimensionality = self.embedding_dim
             
        result = self.client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text,
            config=config
        )
        return result.embeddings[0].values

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
        )
        if self.embedding_dim:
             config.output_dimensionality = self.embedding_dim
             
        result = self.client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=texts,
            config=config
        )
        return [e.values for e in result.embeddings]
