"""
OpenRouter provider: one API key for both the LLM and the embeddings.

Defaults (override with arguments or environment variables):
    LLM         anthropic/claude-haiku-5.5      OPENROUTER_LLM_MODEL
    Embeddings  openai/text-embedding-3-small   OPENROUTER_EMBEDDING_MODEL
    API key                                     OPENROUTER_API_KEY

OpenRouter's chat (/chat/completions) and embeddings (/embeddings) endpoints
follow OpenAI's request and response shapes, so this uses the async OpenAI
client pointed at OpenRouter's base URL.
"""
import base64
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .base import BaseEmbeddings, BaseLLM

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "anthropic/claude-haiku-5.5"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


class OpenRouterError(RuntimeError):
    """OpenRouter returned something unusable (no choices, no embeddings, ...)."""


def _image_mime_type(data: bytes) -> str:
    """Media type from the file signature (the loader passes raw bytes)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:3] == b"GIF":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class OpenRouterProvider(BaseLLM, BaseEmbeddings):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        max_tokens: int = 16000,
        embedding_batch_size: int = 100,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 3,
        app_name: str = "EKM",
        http_client: Optional[Any] = None,
    ):
        """
        Args:
            api_key: OpenRouter key; defaults to OPENROUTER_API_KEY.
            model_name: Chat model; defaults to OPENROUTER_LLM_MODEL, then Claude Haiku 5.5.
            embedding_model: Embedding model; defaults to OPENROUTER_EMBEDDING_MODEL,
                then openai/text-embedding-3-small (1536 dimensions).
            embedding_dim: Ask the model for this many dimensions (only models that
                support it, such as OpenAI's text-embedding-3 family).
            max_tokens: Output cap per chat completion.
            embedding_batch_size: Texts per embeddings request.
            max_retries: Client retries for connection errors, 408/409/429 and 5xx.
            app_name: Sent as X-Title so usage is labelled on the OpenRouter dashboard.
            http_client: Optional httpx.AsyncClient (proxies, tests).
        """
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key missing: pass api_key or set OPENROUTER_API_KEY")

        self.model_name = model_name or os.environ.get("OPENROUTER_LLM_MODEL") or DEFAULT_LLM_MODEL
        self.embedding_model = (
            embedding_model or os.environ.get("OPENROUTER_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        )
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens
        self.embedding_batch_size = max(1, embedding_batch_size)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers={"X-Title": app_name},
            http_client=http_client,
        )

    # --- LLM -------------------------------------------------------------

    async def _chat(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        kwargs.setdefault("max_tokens", self.max_tokens)
        completion = await self.client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )
        error = getattr(completion, "error", None)  # OpenRouter can report errors in a 200 body
        if error:
            raise OpenRouterError(f"OpenRouter error from {self.model_name}: {error}")
        if not completion.choices:
            raise OpenRouterError(f"OpenRouter returned no choices for {self.model_name}")
        choice = completion.choices[0]
        content = choice.message.content or ""
        if not content and choice.finish_reason not in (None, "stop"):
            raise OpenRouterError(
                f"{self.model_name} returned no text (finish_reason={choice.finish_reason!r})"
            )
        return content

    async def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return await self._chat(messages, **kwargs)

    async def generate_from_image(self, image_data: bytes, prompt: str, **kwargs) -> str:
        """Describe or transcribe an image (used by DocumentLoader for images and scanned PDFs)."""
        data_url = f"data:{_image_mime_type(image_data)};base64," + base64.b64encode(image_data).decode("ascii")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }]
        return await self._chat(messages, **kwargs)

    # --- Embeddings ------------------------------------------------------

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        params: Dict[str, Any] = {"model": self.embedding_model, "encoding_format": "float"}
        if self.embedding_dim:
            params["dimensions"] = self.embedding_dim
        # Embedding APIs reject empty strings; a single space embeds "nothing"
        inputs = [t if t and t.strip() else " " for t in texts]

        vectors: List[List[float]] = []
        for start in range(0, len(inputs), self.embedding_batch_size):
            batch = inputs[start:start + self.embedding_batch_size]
            response = await self.client.embeddings.create(input=batch, **params)
            data = sorted(response.data or [], key=lambda d: d.index)
            if len(data) != len(batch):
                raise OpenRouterError(
                    f"{self.embedding_model} returned {len(data)} embeddings for {len(batch)} texts"
                )
            vectors.extend(list(d.embedding) for d in data)
        return vectors

    async def embed_query(self, text: str) -> List[float]:
        return (await self._embed([text]))[0]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await self._embed(texts)


def get_default_provider(**overrides) -> OpenRouterProvider:
    """The provider EKM uses for both LLM and embeddings unless told otherwise."""
    return OpenRouterProvider(**overrides)
