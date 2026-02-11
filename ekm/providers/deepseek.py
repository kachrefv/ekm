import httpx
from openai import OpenAI
from typing import List, Optional, Dict, Any
from .base import BaseLLM
from tenacity import retry, stop_after_attempt, wait_exponential

class DeepSeekProvider(BaseLLM):
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model_name: str = "deepseek-chat"):
        # Create HTTP client without proxies to avoid the error
        http_client = httpx.Client()
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        self.model_name = model_name

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
