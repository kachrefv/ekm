"""
Query Analysis Module

This module provides functionality to analyze queries and extract structural patterns
for the EKM's Structural Retrieval Head.
"""
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ..providers.base import BaseLLM
from .patterns import (
    count_graphlets,
    compute_random_walk_patterns,
    compute_degree_statistics,
    compute_semantic_centroids
)

logger = logging.getLogger(__name__)

# Words that carry no meaning in a relation name ("was born in" -> "born")
_RELATION_STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'of', 'in', 'on',
    'at', 'to', 'for', 'by', 'with', 'from', 'as', 'has', 'have', 'had', 'its', 'their', 'and'
}


def normalize_relation_label(label: Any) -> str:
    """Lower-case content words of a relation name, e.g. 'Was Born In' -> 'born'.
    Empty string when nothing meaningful is left."""
    if not isinstance(label, str):
        return ''
    words = [w for w in re.findall(r'[a-z]+', label.lower()) if w not in _RELATION_STOPWORDS]
    return ' '.join(words)


def relation_label_set(edges: List[Any]) -> List[str]:
    """Sorted distinct normalized relation names from [src, tgt, label] edges
    (a sorted list rather than a set so the signature stays JSON-serializable)."""
    labels = set()
    for edge in edges or []:
        if isinstance(edge, (list, tuple)) and len(edge) >= 3:
            normalized = normalize_relation_label(edge[2])
            if normalized:
                labels.add(normalized)
    return sorted(labels)

class QueryGraphExtractor:
    """
    Extracts graph structures from natural language queries to enable
    structure-aware retrieval.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    async def extract_query_graph(self, query: str) -> Dict[str, Any]:
        """
        Extracts a graph representation (entities and relations) from a query
        and computes its pattern signature.

        Args:
            query: The natural language query

        Returns:
            Dict containing the pattern signature of the query graph
        """
        signature = await self.extract_text_pattern(query, label="QUERY")
        if signature is None:
            logger.warning(f"Could not extract graph from query: '{query}'")
            return self._get_empty_pattern()
        return signature

    async def extract_text_pattern(self, text: str, label: str = "QUERY") -> Optional[Dict[str, Any]]:
        """
        Extracts an entity-relation graph from any text and computes its pattern signature.

        Used for both queries and GKUs (the facts in a cluster), so the two sides of
        the structural head are built the same way: same extraction prompt, same
        graph type (entities linked by relations), same feature functions.

        Args:
            text: Query or fact text
            label: Name shown to the LLM for the text ("QUERY", "FACTS", ...)

        Returns:
            Pattern signature, or None if no graph could be extracted
        """
        # 1. Extract entities and relationships using LLM with Chain-of-Thought reasoning
        graph_data = await self._extract_graph_from_text_cot(text, label=label)

        # 2. Convert to adjacency matrix
        adj_matrix, node_map = self._build_adjacency_matrix(graph_data)

        if adj_matrix.shape[0] == 0:
            return None

        # 3. Compute pattern signature using patterns.py
        try:
            return {
                'graphlets': count_graphlets(adj_matrix),
                'random_walks': compute_random_walk_patterns(adj_matrix),
                'degree_stats': compute_degree_statistics(adj_matrix),
                'graph_properties': {
                    'num_nodes': int(adj_matrix.shape[0]),
                    'num_edges': int(adj_matrix.nnz // 2) if hasattr(adj_matrix, 'nnz') else 0
                },
                # What the edges mean, not just their shape
                'relation_labels': relation_label_set(graph_data.get('edges', []))
            }
        except Exception as e:
            logger.error(f"Error computing patterns for {label.lower()} graph: {e}")
            return None

    async def _extract_graph_from_text_cot(self, text: str, label: str = "QUERY") -> Dict[str, List[Any]]:
        """Uses Chain-of-Thought reasoning to parse text into nodes and edges."""

        system_prompt = """You are a graph extraction expert who thinks step-by-step to ensure accurate identification of entities and relationships.
        Analyze the user query systematically to extract key entities (nodes) and their relationships (edges).
        Follow this Chain-of-Thought process:
        1. Identify all entities mentioned in the query
        2. Identify all relationships between these entities
        3. Formulate these as nodes and edges in a graph structure
        4. Return valid JSON with the graph structure
        """

        cot_user_prompt = f"""
        Analyze the text step-by-step to extract a graph structure:

        {label}: {text}

        CHAIN-OF-THOUGHT REASONING:
        1. ENTITY IDENTIFICATION:
           - Main entities: [list all entities mentioned]
           - Implicit entities: [list any entities implied by the query]
           
        2. RELATIONSHIP ANALYSIS:
           - Direct relationships: [list explicit relationships between entities]
           - Implied relationships: [list relationships that can be inferred]
           - Relationship types: [categorize the nature of each relationship]
           
        3. GRAPH STRUCTURE FORMULATION:
           - Nodes: [formulate the list of entities as nodes]
           - Edges: [formulate the relationships as edges with appropriate types]
           
        4. VALIDATION:
           - Ensure all entities are captured as nodes
           - Ensure all meaningful relationships are captured as edges
           - Verify the graph structure accurately represents the query

        FINAL OUTPUT: Return in valid JSON format:
        {{
          "nodes": ["entity1", "entity2", ...],
          "edges": [["entity1", "entity2", "relationship_type"], ...]
        }}
        """

        try:
            response = await self.llm.generate(
                system_prompt=system_prompt,
                user_message=cot_user_prompt
            )

            # Simple JSON parsing
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                logger.warning("No JSON found in LLM response for graph extraction")
                return {"nodes": [], "edges": []}

        except Exception as e:
            logger.error(f"LLM graph extraction failed: {e}")
            return {"nodes": [], "edges": []}

    async def extract_query_graph_detailed(self, query: str) -> Dict[str, Any]:
        """
        Extracts a detailed graph representation with explicit reasoning steps.
        
        Args:
            query: The natural language query

        Returns:
            Dict containing the pattern signature and reasoning trace
        """
        # Extract with detailed reasoning
        result = await self._extract_graph_from_text_detailed(query)

        # Check if result has the expected structure, otherwise use fallback
        if 'graph_data' not in result:
            logger.warning(f"Detailed extraction failed for query: '{query}', using fallback")
            # Fallback to basic extraction
            graph_data = await self._extract_graph_from_text_cot(query)
            reasoning_trace = result.get('reasoning', 'Fallback extraction due to parsing issue')
            confidence_score = result.get('confidence', 0.6)
        else:
            graph_data = result['graph_data']
            reasoning_trace = result.get('reasoning', 'Detailed extraction performed')
            confidence_score = result.get('confidence', 0.8)

        # Convert to adjacency matrix
        adj_matrix, node_map = self._build_adjacency_matrix(graph_data)

        if adj_matrix.shape[0] == 0:
            logger.warning(f"Could not extract graph from query: '{query}'")
            return {**self._get_empty_pattern(), 'reasoning_trace': reasoning_trace, 'confidence_score': confidence_score}

        # Compute pattern signature
        try:
            graphlets = count_graphlets(adj_matrix)
            random_walks = compute_random_walk_patterns(adj_matrix)
            degree_stats = compute_degree_statistics(adj_matrix)

            return {
                'graphlets': graphlets,
                'random_walks': random_walks,
                'degree_stats': degree_stats,
                'reasoning_trace': reasoning_trace,
                'confidence_score': confidence_score
            }
        except Exception as e:
            logger.error(f"Error computing patterns for query graph: {e}")
            return {**self._get_empty_pattern(), 'reasoning_trace': reasoning_trace, 'confidence_score': confidence_score}

    async def _extract_graph_from_text_detailed(self, text: str) -> Dict[str, Any]:
        """Uses detailed Chain-of-Thought reasoning to parse text into nodes and edges with confidence scoring."""

        system_prompt = """You are a graph extraction expert performing detailed analysis.
        Extract entities and relationships from the query using systematic Chain-of-Thought reasoning.
        Provide confidence scores for your extractions and explain your reasoning.
        """

        detailed_cot_prompt = f"""
        Perform detailed graph extraction with explicit reasoning:

        INPUT QUERY: {text}

        STEP-BY-STEP REASONING:
        1. TEXT UNDERSTANDING:
           - Query intent: [what is the user asking for?]
           - Key terms: [identify important terms/entities]
           - Context clues: [any contextual information that affects interpretation]

        2. ENTITY EXTRACTION:
           - Named entities: [explicitly named people, places, organizations, etc.]
           - Conceptual entities: [abstract concepts that should be treated as nodes]
           - Implicit entities: [entities that are implied but not directly named]
           - Confidence in each entity: [rate confidence 0.0-1.0]

        3. RELATIONSHIP IDENTIFICATION:
           - Explicit relations: [directly stated relationships]
           - Implicit relations: [relationships that can be inferred]
           - Temporal relations: [time-based connections if applicable]
           - Causal relations: [cause-effect relationships if applicable]
           - Confidence in each relation: [rate confidence 0.0-1.0]

        4. GRAPH FORMULATION:
           - Nodes: [compile final list of entities as nodes]
           - Edges: [compile final list of relationships as edges]
           - Edge types: [specify the type/category of each relationship]

        5. VALIDATION AND CONFIDENCE ASSESSMENT:
           - Does the graph capture the essence of the query?
           - Are there any missing important connections?
           - Overall confidence score: [0.0-1.0]

        OUTPUT FORMAT:
        {{
          "graph_data": {{
            "nodes": ["entity1", "entity2", ...],
            "edges": [["entity1", "entity2", "relationship_type"], ...]
          }},
          "reasoning": "Detailed explanation of extraction process",
          "confidence": 0.85
        }}
        """

        try:
            response = await self.llm.generate(
                system_prompt=system_prompt,
                user_message=detailed_cot_prompt
            )

            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            else:
                logger.warning("No JSON found in detailed LLM response for graph extraction")
                # Fallback to basic extraction
                basic_result = await self._extract_graph_from_text_cot(text)
                return {
                    "graph_data": basic_result,
                    "reasoning": "Basic extraction performed due to parsing issue",
                    "confidence": 0.6
                }

        except Exception as e:
            logger.error(f"Detailed LLM graph extraction failed: {e}")
            # Fallback to basic extraction
            basic_result = await self._extract_graph_from_text_cot(text)
            return {
                "graph_data": basic_result,
                "reasoning": f"Fallback extraction due to error: {str(e)}",
                "confidence": 0.5
            }

    def _build_adjacency_matrix(self, graph_data: Dict[str, Any]) -> Tuple[Any, Dict[str, int]]:
        """Converts node/edge lists to a sparse adjacency matrix."""
        from scipy.sparse import lil_matrix
        import numpy as np

        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            # Try to infer nodes from edges if nodes list is empty
            unique_nodes = set()
            for src, tgt, _ in edges:
                unique_nodes.add(src)
                unique_nodes.add(tgt)
            nodes = list(unique_nodes)

        n = len(nodes)
        if n == 0:
            return np.zeros((0, 0)), {}

        node_map = {node: i for i, node in enumerate(nodes)}
        adj = lil_matrix((n, n), dtype=np.float32)

        for edge in edges:
            if len(edge) >= 2:
                src, tgt = edge[0], edge[1]
                if src in node_map and tgt in node_map:
                    i, j = node_map[src], node_map[tgt]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0 # Undirected for structural pattern matching

        return adj.tocsr(), node_map

    def _get_empty_pattern(self) -> Dict[str, Any]:
        """Returns a default empty pattern structure."""
        return {
            'graphlets': {'triangle': 0, 'wedge': 0, 'triplet': 0},
            'random_walks': {'return_probability': 0.0, 'path_diversity': 0.0},
            'degree_stats': {'mean_degree': 0.0, 'max_degree': 0, 'min_degree': 0, 'degree_variance': 0.0}
        }
