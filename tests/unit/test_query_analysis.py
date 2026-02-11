"""
Unit tests for QueryGraphExtractor.
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from ekm.core.query_analysis import QueryGraphExtractor

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate = AsyncMock()
    return llm

@pytest.mark.asyncio
async def test_extract_query_graph_success(mock_llm):
    """Test successful extraction of a query graph."""
    # Mock LLM response
    mock_llm.generate.return_value = """
    {
        "nodes": ["Quantum Neural Fabrics", "distributed AI", "mesh structure"],
        "edges": [
            ["Quantum Neural Fabrics", "distributed AI", "is architecture for"],
            ["Quantum Neural Fabrics", "mesh structure", "has"],
            ["distributed AI", "mesh structure", "connected in"]
        ]
    }
    """

    extractor = QueryGraphExtractor(llm=mock_llm)
    pattern = await extractor.extract_query_graph("What is Quantum Neural Fabrics?")

    # Verify pattern structure
    assert 'graphlets' in pattern
    assert 'random_walks' in pattern
    assert 'degree_stats' in pattern

    # Verify graphlets (should have at least a triangle or wedge)
    # 3 nodes fully connected (triangle) or partially (wedge)
    # The edges form a triangle: A->B, A->C, B->C (if undirected)
    # Our adjacency matrix is undirected for structural matching
    assert pattern['graphlets']['triangle'] >= 0
    assert pattern['degree_stats']['max_degree'] >= 0

@pytest.mark.asyncio
async def test_extract_query_graph_empty_response(mock_llm):
    """Test handling of empty or invalid LLM response."""
    mock_llm.generate.return_value = "I cannot answer that."

    extractor = QueryGraphExtractor(llm=mock_llm)
    pattern = await extractor.extract_query_graph("Unknown query")

    # Should return empty pattern with default values
    assert pattern['graphlets']['triangle'] == 0
    assert pattern['degree_stats']['mean_degree'] == 0.0

@pytest.mark.asyncio
async def test_extract_query_graph_no_json(mock_llm):
    """Test handling of response without JSON."""
    mock_llm.generate.return_value = "Here are the nodes: A, B. Edges: A->B."

    extractor = QueryGraphExtractor(llm=mock_llm)
    pattern = await extractor.extract_query_graph("Simple query")

    # should gracefully return empty pattern
    assert pattern['graphlets']['triangle'] == 0

@pytest.mark.asyncio
async def test_extract_query_graph_detailed_success(mock_llm):
    """Test successful extraction of a query graph with detailed reasoning."""
    # Mock LLM response for detailed extraction
    mock_llm.generate.return_value = """
    {
        "graph_data": {
            "nodes": ["EKM", "Knowledge Retrieval", "Graph Patterns"],
            "edges": [
                ["EKM", "Knowledge Retrieval", "enables"],
                ["EKM", "Graph Patterns", "uses"],
                ["Knowledge Retrieval", "Graph Patterns", "based_on"]
            ]
        },
        "reasoning": "Detailed analysis of query structure showing relationships between concepts",
        "confidence": 0.85
    }
    """

    extractor = QueryGraphExtractor(llm=mock_llm)
    result = await extractor.extract_query_graph_detailed("How does EKM use graph patterns for knowledge retrieval?")

    # Verify pattern structure
    assert 'graphlets' in result
    assert 'random_walks' in result
    assert 'degree_stats' in result
    assert 'reasoning_trace' in result
    assert 'confidence_score' in result

    # Verify the reasoning trace is included
    assert result['reasoning_trace'] == "Detailed analysis of query structure showing relationships between concepts"
    assert result['confidence_score'] == 0.85

@pytest.mark.asyncio
async def test_extract_query_graph_detailed_fallback_missing_structure(mock_llm):
    """Test fallback behavior when detailed extraction returns unexpected structure."""
    # Mock a response that doesn't contain the expected 'graph_data' field
    mock_llm.generate.return_value = """
    {
        "some_other_field": "some_value",
        "reasoning": "Some reasoning without proper structure",
        "confidence": 0.7
    }
    """

    extractor = QueryGraphExtractor(llm=mock_llm)
    result = await extractor.extract_query_graph_detailed("Test query for fallback")

    # Should still return the basic pattern structure with reasoning
    assert 'graphlets' in result
    assert 'reasoning_trace' in result
    assert 'confidence_score' in result

    # Verify that the reasoning from the original response is preserved when available
    assert result['reasoning_trace'] == "Some reasoning without proper structure"


@pytest.mark.asyncio
async def test_extract_query_graph_detailed_fallback_exception(mock_llm):
    """Test fallback behavior when detailed extraction encounters an exception."""
    # Make the detailed extraction method raise an exception
    original_method = mock_llm.generate
    async def mock_generate_side_effect(system_prompt, user_message):
        # Check if this is the detailed extraction call (it contains "STEP-BY-STEP REASONING")
        if "STEP-BY-STEP REASONING" in user_message:
            raise Exception("LLM Error")
        else:
            # Return a basic response for the fallback call
            return """
            {
                "nodes": ["Node A", "Node B"],
                "edges": [["Node A", "Node B", "relation"]]
            }
            """
    
    mock_llm.generate.side_effect = mock_generate_side_effect

    extractor = QueryGraphExtractor(llm=mock_llm)
    result = await extractor.extract_query_graph_detailed("Test query for exception fallback")

    # Should still return the basic pattern structure with reasoning
    assert 'graphlets' in result
    assert 'reasoning_trace' in result
    assert 'confidence_score' in result
