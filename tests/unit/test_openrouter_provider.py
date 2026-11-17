"""
OpenRouter provider, tested at the HTTP layer: the requests must hit OpenRouter's
endpoints with the right models and bodies. No network access.
"""
import base64
import json

import httpx
import pytest

from ekm.providers.openrouter import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, OpenRouterError, OpenRouterProvider,
)


class FakeOpenRouter:
    """Records requests and answers like OpenRouter's chat / embeddings endpoints."""

    def __init__(self, chat_content="hello", finish_reason="stop", dims=4):
        self.requests = []
        self.chat_content = chat_content
        self.finish_reason = finish_reason
        self.dims = dims

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request, body))
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={
                "id": "gen-1", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": self.finish_reason,
                             "message": {"role": "assistant", "content": self.chat_content}}],
            })
        if request.url.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            # Return out of order to check that results are re-sorted by index
            data = [{"object": "embedding", "index": i, "embedding": [float(i)] * self.dims}
                    for i in range(len(inputs))][::-1]
            return httpx.Response(200, json={"object": "list", "model": body["model"], "data": data,
                                             "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        return httpx.Response(404, json={"error": {"message": "unknown path"}})


def make_provider(fake, **kwargs):
    return OpenRouterProvider(api_key="test-key", max_retries=0,
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake)), **kwargs)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "OPENROUTER_LLM_MODEL", "OPENROUTER_EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.asyncio
async def test_generate_calls_openrouter_with_claude_haiku():
    fake = FakeOpenRouter(chat_content="Paris")
    provider = make_provider(fake)

    answer = await provider.generate("You are terse.", "Capital of France?")

    assert answer == "Paris"
    request, body = fake.requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-title"] == "EKM"
    assert body["model"] == DEFAULT_LLM_MODEL == "anthropic/claude-haiku-5.5"
    assert body["messages"] == [{"role": "system", "content": "You are terse."},
                                {"role": "user", "content": "Capital of France?"}]
    assert body["max_tokens"] == 16000


@pytest.mark.asyncio
async def test_embeddings_hit_openrouter_in_batches_and_keep_order():
    fake = FakeOpenRouter(dims=3)
    provider = make_provider(fake, embedding_batch_size=2)

    vectors = await provider.embed_documents(["a", "b", "c", ""])

    assert vectors == [[0.0] * 3, [1.0] * 3, [0.0] * 3, [1.0] * 3]   # two batches of two
    assert len(fake.requests) == 2
    request, body = fake.requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/embeddings"
    assert body["model"] == DEFAULT_EMBEDDING_MODEL == "openai/text-embedding-3-small"
    assert body["encoding_format"] == "float"
    assert fake.requests[1][1]["input"] == ["c", " "]               # empty text sent as a space
    assert "dimensions" not in body


@pytest.mark.asyncio
async def test_embed_query_and_optional_dimensions():
    fake = FakeOpenRouter(dims=2)
    provider = make_provider(fake, embedding_dim=512)

    assert await provider.embed_query("q") == [0.0, 0.0]
    assert fake.requests[0][1]["dimensions"] == 512
    assert await provider.embed_documents([]) == []


@pytest.mark.asyncio
async def test_models_and_key_come_from_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "anthropic/claude-sonnet-5.5")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    fake = FakeOpenRouter()
    provider = OpenRouterProvider(http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake)))

    await provider.generate("", "hi")
    await provider.embed_query("hi")

    chat_request, chat_body = fake.requests[0]
    assert chat_request.headers["authorization"] == "Bearer env-key"
    assert chat_body["model"] == "anthropic/claude-sonnet-5.5"
    assert chat_body["messages"] == [{"role": "user", "content": "hi"}]   # empty system prompt dropped
    assert fake.requests[1][1]["model"] == "qwen/qwen3-embedding-8b"


def test_missing_key_is_a_clear_error():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider()


@pytest.mark.asyncio
async def test_empty_answer_with_non_stop_finish_raises():
    provider = make_provider(FakeOpenRouter(chat_content=None, finish_reason="content_filter"))

    with pytest.raises(OpenRouterError, match="content_filter"):
        await provider.generate("s", "u")


@pytest.mark.asyncio
async def test_generate_from_image_sends_a_data_url():
    fake = FakeOpenRouter(chat_content="text in image")
    provider = make_provider(fake)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

    assert await provider.generate_from_image(png, "Transcribe") == "text in image"

    content = fake.requests[0][1]["messages"][0]["content"]
    assert content[0]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(png).decode()
    assert content[1] == {"type": "text", "text": "Transcribe"}
