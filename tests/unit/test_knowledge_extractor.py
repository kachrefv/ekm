"""
Unit tests for KnowledgeExtractor with Chain-of-Thought enhancements.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from ekm.core.knowledge import KnowledgeExtractor


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm


@pytest.fixture
def mock_embeddings():
    embeddings = MagicMock()
    return embeddings


@pytest.mark.asyncio
async def test_extract_akus_cot_success(mock_llm, mock_embeddings):
    """Test successful extraction of AKUs using Chain-of-Thought prompting."""
    # Mock LLM response for AKU extraction with CoT
    mock_llm.generate.return_value = '''
    {
        "akus": [
            "Quantum Neural Fabrics represent a new frontier in AI",
            "QNF uses mesh-based node connectivity for distributed processing",
            "The core principle of QNF is spreading activation",
            "Spreading activation propagates through nodes based on semantic affinity"
        ]
    }
    '''

    extractor = KnowledgeExtractor(llm=mock_llm, embeddings=mock_embeddings)
    akus = await extractor.extract_akus("Quantum Neural Fabrics (QNF) represent a new frontier in AI, "
                                       "using mesh-based node connectivity for distributed processing. "
                                       "The core principle of QNF is spreading activation, where signals "
                                       "propagate through nodes based on semantic affinity.")

    # Verify that AKUs were extracted
    assert len(akus) > 0
    assert "Quantum Neural Fabrics represent a new frontier in AI" in akus
    assert "QNF uses mesh-based node connectivity for distributed processing" in akus


@pytest.mark.asyncio
async def test_extract_akus_with_reasoning_success(mock_llm, mock_embeddings):
    """Test successful extraction of AKUs with detailed reasoning."""
    # Mock LLM response for detailed AKU extraction
    mock_llm.generate.return_value = '''
    {
        "akus": [
            "EKM implements biological principles using AKUs",
            "AKUs represent episodes in the Episodic Knowledge Mesh",
            "GKUs are created during consolidation phase",
            "GKUs represent abstract, summarized concepts"
        ],
        "reasoning": "Breaking down the text into atomic knowledge units by identifying entities and their properties",
        "confidence_scores": [0.9, 0.85, 0.88, 0.82]
    }
    '''

    extractor = KnowledgeExtractor(llm=mock_llm, embeddings=mock_embeddings)
    result = await extractor.extract_akus_with_reasoning(
        "The Episodic Knowledge Mesh (EKM) implements biological principles using AKUs (Atomic Knowledge Units) for episodes. "
        "GKUs (Global Knowledge Units) are created in EKM during the consolidation phase to represent abstract, summarized concepts."
    )

    # Verify the structure of the result
    assert 'akus' in result
    assert 'reasoning' in result
    assert 'confidence_scores' in result

    # Verify AKUs were extracted
    assert len(result['akus']) > 0
    assert "EKM implements biological principles using AKUs" in result['akus']

    # Verify reasoning is included
    assert "Breaking down the text into atomic knowledge units" in result['reasoning']

    # Verify confidence scores are included
    assert len(result['confidence_scores']) == len(result['akus'])


@pytest.mark.asyncio
async def test_extract_akus_with_reasoning_fallback(mock_llm, mock_embeddings):
    """Test fallback behavior when detailed extraction fails."""
    # Mock a response that doesn't contain proper JSON for detailed extraction
    mock_llm.generate.return_value = '''
    ["EKM uses Atomic Knowledge Units", "AKUs store episodic information"]
    '''

    extractor = KnowledgeExtractor(llm=mock_llm, embeddings=mock_embeddings)
    result = await extractor.extract_akus_with_reasoning("EKM uses Atomic Knowledge Units to store episodic information.")

    # Verify fallback behavior
    assert 'akus' in result
    assert 'reasoning' in result
    assert 'confidence_scores' in result

    # Check that fallback reasoning message is present
    assert result['reasoning'] == 'Basic extraction performed due to parsing issue'


@pytest.mark.asyncio
async def test_single_step_extraction_cot_success(mock_llm, mock_embeddings):
    """Test successful single-step extraction with Chain-of-Thought prompting."""
    # Mock LLM response for single-step extraction with CoT
    mock_llm.generate.return_value = '''
    {
        "summary": "EKM implements biological memory principles using atomic knowledge units.",
        "facts": [
            "EKM implements biological principles using AKUs for episodes",
            "GKUs are created during consolidation phase",
            "GKUs represent abstract, summarized concepts"
        ]
    }
    '''

    extractor = KnowledgeExtractor(llm=mock_llm, embeddings=mock_embeddings)
    result = await extractor.single_step_extraction(
        "The Episodic Knowledge Mesh (EKM) implements biological principles using AKUs (Atomic Knowledge Units) for episodes. "
        "GKUs (Global Knowledge Units) are created in EKM during the consolidation phase to represent abstract, summarized concepts."
    )

    # Verify the structure of the result
    assert 'summary' in result
    assert 'facts' in result

    # Verify content was extracted
    assert "EKM implements biological memory principles" in result['summary']
    assert len(result['facts']) > 0
    assert "EKM implements biological principles using AKUs for episodes" in result['facts']


@pytest.mark.asyncio
async def test_extract_akus_empty_response(mock_llm, mock_embeddings):
    """Test handling of empty response from LLM."""
    mock_llm.generate.return_value = ""

    extractor = KnowledgeExtractor(llm=mock_llm, embeddings=mock_embeddings)
    akus = await extractor.extract_akus("Some text")

    # Should return empty list
    assert akus == []


@pytest.mark.asyncio
async def test_extract_akus_invalid_json(mock_llm, mock_embeddings):
    """Test handling of invalid JSON response from LLM."""
    mock_llm.generate.return_value = "This is not JSON"

    extractor = KnowledgeExtractor(llm=mock_llm, embeddings=mock_embeddings)
    akus = await extractor.extract_akus("Some text")

    # Should return empty list
    assert akus == []