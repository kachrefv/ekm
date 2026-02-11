import pytest
from unittest.mock import MagicMock, patch
from ekm.providers.gemini import GeminiProvider
from google.genai import types

@pytest.fixture
def mock_genai_client():
    with patch("ekm.providers.gemini.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        yield mock_client

@pytest.fixture
def provider(mock_genai_client):
    return GeminiProvider(api_key="fake_key", embedding_dim=768)

@pytest.mark.asyncio
async def test_generate(provider, mock_genai_client):
    mock_response = MagicMock()
    mock_response.text = "Generated response"
    mock_genai_client.models.generate_content.return_value = mock_response

    response = await provider.generate(system_prompt="Sys", user_message="User")
    
    assert response == "Generated response"
    mock_genai_client.models.generate_content.assert_called_once()
    call_args = mock_genai_client.models.generate_content.call_args
    assert call_args.kwargs["model"] == "gemini-1.5-flash"
    assert "Sys" in call_args.kwargs["contents"]
    assert "User" in call_args.kwargs["contents"]

@pytest.mark.asyncio
async def test_embed_query(provider, mock_genai_client):
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]
    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]
    mock_genai_client.models.embed_content.return_value = mock_result

    embedding = await provider.embed_query("dataset")
    
    assert embedding == [0.1, 0.2, 0.3]
    mock_genai_client.models.embed_content.assert_called_once()
    call_args = mock_genai_client.models.embed_content.call_args
    assert call_args.kwargs["model"] == "models/embedding-001"
    assert call_args.kwargs["contents"] == "dataset"
    assert isinstance(call_args.kwargs["config"], types.EmbedContentConfig)
    assert call_args.kwargs["config"].output_dimensionality == 768

@pytest.mark.asyncio
async def test_embed_documents(provider, mock_genai_client):
    mock_emb1 = MagicMock()
    mock_emb1.values = [0.1, 0.1]
    mock_emb2 = MagicMock()
    mock_emb2.values = [0.2, 0.2]
    
    mock_result = MagicMock()
    mock_result.embeddings = [mock_emb1, mock_emb2]
    mock_genai_client.models.embed_content.return_value = mock_result

    embeddings = await provider.embed_documents(["doc1", "doc2"])
    
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.1]
    assert embeddings[1] == [0.2, 0.2]
    
    mock_genai_client.models.embed_content.assert_called_once()
