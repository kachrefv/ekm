"""
Explainability Features for EKM Retrieval Decisions
Provides tools to explain why certain results were retrieved and how attention weights influenced the decision.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class ExplanationSegment:
    """A segment of an explanation with a specific focus."""
    title: str
    content: str
    confidence: float  # 0.0 to 1.0
    supporting_evidence: List[Dict[str, Any]]


class RetrievalExplainer:
    """Generates explanations for retrieval decisions."""
    
    def __init__(self):
        pass
    
    def generate_explanation(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        attention_weights: Dict[str, np.ndarray],
        interpretations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation for the retrieval results.
        
        Args:
            query: The original query
            results: The retrieval results
            attention_weights: Attention weights from different heads
            interpretations: Interpretation data from the attention mechanism
        
        Returns:
            Dictionary containing the explanation
        """
        explanation_segments = []
        
        # 1. Overall explanation
        overall_exp = self._generate_overall_explanation(query, results)
        explanation_segments.append(overall_exp)
        
        # 2. Relevance explanation for top results
        top_results = results[:3]  # Explain top 3 results
        for i, result in enumerate(top_results):
            relevance_exp = self._generate_relevance_explanation(query, result, i+1)
            explanation_segments.append(relevance_exp)
        
        # 3. Attention breakdown explanation
        attention_exp = self._generate_attention_breakdown(attention_weights, interpretations)
        explanation_segments.append(attention_exp)
        
        # 4. Methodology explanation
        methodology_exp = self._generate_methodology_explanation(interpretations)
        explanation_segments.append(methodology_exp)
        
        return {
            'query': query,
            'explanation_segments': explanation_segments,
            'confidence_score': self._calculate_confidence_score(explanation_segments),
            'key_factors': self._extract_key_factors(interpretations),
            'alternative_explanations': self._generate_alternatives(query, results)
        }
    
    def _generate_overall_explanation(self, query: str, results: List[Dict[str, Any]]) -> ExplanationSegment:
        """Generate an overall explanation of the retrieval."""
        if not results:
            content = f"No results were retrieved for the query '{query}'. This could be because no relevant knowledge units matched the query."
            confidence = 0.5
        else:
            result_count = len(results)
            avg_score = np.mean([r.get('score', 0) for r in results]) if results else 0
            content = (f"The system retrieved {result_count} result(s) for the query '{query}'. "
                      f"The average relevance score was {avg_score:.3f}. "
                      f"These results were selected based on their semantic, structural, and temporal relevance to the query.")
            confidence = min(0.9, avg_score + 0.1)  # Higher scores get higher confidence
        
        return ExplanationSegment(
            title="Overall Retrieval Summary",
            content=content,
            confidence=confidence,
            supporting_evidence=[{'query': query, 'result_count': len(results)}]
        )
    
    def _generate_relevance_explanation(self, query: str, result: Dict[str, Any], rank: int) -> ExplanationSegment:
        """Generate explanation for why a specific result was retrieved."""
        result_id = result.get('id', 'unknown')
        result_content = result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', '')
        score = result.get('score', 0)
        layer = result.get('layer', 'unknown')
        
        content = (f"Result #{rank} (ID: {result_id}, Layer: {layer}) was retrieved because it has high relevance to the query. "
                  f"The content '{result_content}' contains concepts that match the query '{query}'. "
                  f"The relevance score is {score:.3f}, indicating strong alignment between the query and this knowledge unit.")
        
        # Determine confidence based on score
        confidence = min(1.0, max(0.1, score))
        
        supporting_evidence = [
            {'result_id': result_id, 'score': score, 'layer': layer, 'content_preview': result_content}
        ]
        
        return ExplanationSegment(
            title=f"Relevance of Result #{rank}",
            content=content,
            confidence=confidence,
            supporting_evidence=supporting_evidence
        )
    
    def _generate_attention_breakdown(
        self, 
        attention_weights: Dict[str, np.ndarray], 
        interpretations: Dict[str, Any]
    ) -> ExplanationSegment:
        """Generate explanation of how attention weights influenced the results."""
        breakdown_parts = []
        
        # Semantic attention
        if 'semantic' in attention_weights and len(attention_weights['semantic']) > 0:
            avg_semantic = np.mean(attention_weights['semantic'])
            breakdown_parts.append(f"semantic relevance ({avg_semantic:.3f})")
        
        # Structural attention
        if 'structural' in attention_weights and len(attention_weights['structural']) > 0:
            avg_structural = np.mean(attention_weights['structural'])
            breakdown_parts.append(f"structural pattern matching ({avg_structural:.3f})")
        
        # Temporal attention
        if 'temporal' in attention_weights and len(attention_weights['temporal']) > 0:
            avg_temporal = np.mean(attention_weights['temporal'])
            breakdown_parts.append(f"temporal relevance ({avg_temporal:.3f})")
        
        breakdown_str = ", ".join(breakdown_parts)
        
        # Get head contributions
        head_contributions = interpretations.get('head_weights_used', {})
        contribution_str = ", ".join([f"{k}: {v:.2f}" for k, v in head_contributions.items()])
        
        content = (f"The retrieval decision was influenced by multiple factors: {breakdown_str}. "
                  f"The relative importance of these factors was determined by the attention mechanism, "
                  f"with weights assigned as follows: {contribution_str}. "
                  f"This multi-faceted approach ensures that results are relevant not just semantically, "
                  f"but also structurally and temporally.")
        
        confidence = 0.8  # High confidence in the explanation methodology
        
        return ExplanationSegment(
            title="Attention Mechanism Breakdown",
            content=content,
            confidence=confidence,
            supporting_evidence=[
                {'attention_weights': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in attention_weights.items()},
                 'head_contributions': head_contributions}
            ]
        )
    
    def _generate_methodology_explanation(self, interpretations: Dict[str, Any]) -> ExplanationSegment:
        """Explain the methodology behind the retrieval."""
        primary_factor = interpretations.get('primary_relevance_factor', 'unknown')
        avg_semantic = interpretations.get('average_semantic_contribution', 0)
        avg_structural = interpretations.get('average_structural_contribution', 0)
        avg_temporal = interpretations.get('average_temporal_contribution', 0)
        
        content = (f"The EKM system uses a multi-head attention mechanism to retrieve knowledge. "
                  f"The primary relevance factor was determined to be '{primary_factor}', "
                  f"meaning the system prioritized {primary_factor} similarity when ranking results. "
                  f"On average, semantic relevance contributed {avg_semantic:.3f}, "
                  f"structural relevance contributed {avg_structural:.3f}, and "
                  f"temporal relevance contributed {avg_temporal:.3f} to the final rankings. "
                  f"This approach allows the system to find knowledge that matches on multiple dimensions, "
                  f"providing more comprehensive and contextually appropriate results.")
        
        confidence = 0.9  # Very high confidence in methodology explanation
        
        return ExplanationSegment(
            title="Retrieval Methodology",
            content=content,
            confidence=confidence,
            supporting_evidence=[{'methodology_details': interpretations}]
        )
    
    def _calculate_confidence_score(self, explanation_segments: List[ExplanationSegment]) -> float:
        """Calculate an overall confidence score for the explanation."""
        if not explanation_segments:
            return 0.0
        
        # Average the confidence scores of all segments
        total_confidence = sum(seg.confidence for seg in explanation_segments)
        return total_confidence / len(explanation_segments)
    
    def _extract_key_factors(self, interpretations: Dict[str, Any]) -> List[str]:
        """Extract key factors that influenced the retrieval."""
        factors = []
        
        if 'primary_relevance_factor' in interpretations:
            factors.append(f"Primary relevance factor: {interpretations['primary_relevance_factor']}")
        
        if 'average_semantic_contribution' in interpretations:
            factors.append(f"Average semantic contribution: {interpretations['average_semantic_contribution']:.3f}")
        
        if 'average_structural_contribution' in interpretations:
            factors.append(f"Average structural contribution: {interpretations['average_structural_contribution']:.3f}")
        
        if 'average_temporal_contribution' in interpretations:
            factors.append(f"Average temporal contribution: {interpretations['average_temporal_contribution']:.3f}")
        
        if 'layers_retrieved' in interpretations:
            factors.append(f"Layers retrieved: {', '.join(interpretations['layers_retrieved'])}")
        
        return factors
    
    def _generate_alternatives(self, query: str, results: List[Dict[str, Any]]) -> List[str]:
        """Generate alternative explanations or perspectives."""
        alternatives = []
        
        if len(results) > 1:
            # Mention that other results exist
            alternatives.append(
                f"Other relevant results were also identified but ranked lower. "
                f"These may contain complementary information to the top results."
            )
        
        # Suggest query refinement
        alternatives.append(
            f"If these results don't fully address your needs, consider refining your query "
            f"to be more specific or to emphasize different aspects of the topic."
        )
        
        # Suggest exploring different layers
        alternatives.append(
            f"You might also try searching in different knowledge layers (episodic, AKU, GCU) "
            f"to get different perspectives on the topic."
        )
        
        return alternatives


class ExplanationFormatter:
    """Formats explanations in various output formats."""
    
    def format_as_text(self, explanation: Dict[str, Any]) -> str:
        """Format explanation as plain text."""
        lines = []
        lines.append(f"EXPLANATION FOR QUERY: {explanation['query']}")
        lines.append("=" * 60)
        lines.append("")
        
        for segment in explanation['explanation_segments']:
            lines.append(f"{segment.title}")
            lines.append("-" * len(segment.title))
            lines.append(segment.content)
            lines.append("")
        
        lines.append("KEY FACTORS:")
        for factor in explanation['key_factors']:
            lines.append(f"- {factor}")
        lines.append("")
        
        lines.append(f"OVERALL CONFIDENCE: {explanation['confidence_score']:.2f}")
        lines.append("")
        
        if explanation['alternative_explanations']:
            lines.append("ALTERNATIVE PERSPECTIVES:")
            for alt in explanation['alternative_explanations']:
                lines.append(f"- {alt}")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_as_json(self, explanation: Dict[str, Any]) -> str:
        """Format explanation as JSON string."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_explanation = self._make_serializable(explanation)
        return json.dumps(serializable_explanation, indent=2)
    
    def format_as_html(self, explanation: Dict[str, Any]) -> str:
        """Format explanation as HTML."""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head><title>EKM Retrieval Explanation</title></head>",
            "<body>",
            f"<h1>Explanation for Query: {explanation['query']}</h1>",
            "<div class='explanation-container'>"
        ]
        
        for segment in explanation['explanation_segments']:
            html_parts.append(f"<div class='explanation-segment'>")
            html_parts.append(f"<h2>{segment.title}</h2>")
            html_parts.append(f"<p>{segment.content}</p>")
            html_parts.append(f"<div class='confidence'>Confidence: {segment.confidence:.2f}</div>")
            html_parts.append("</div>")
        
        html_parts.append("<div class='key-factors'>")
        html_parts.append("<h2>Key Factors</h2><ul>")
        for factor in explanation['key_factors']:
            html_parts.append(f"<li>{factor}</li>")
        html_parts.append("</ul></div>")
        
        html_parts.append(f"<div class='overall-confidence'>Overall Confidence: {explanation['confidence_score']:.2f}</div>")
        
        if explanation['alternative_explanations']:
            html_parts.append("<div class='alternatives'>")
            html_parts.append("<h2>Alternative Perspectives</h2><ul>")
            for alt in explanation['alternative_explanations']:
                html_parts.append(f"<li>{alt}</li>")
            html_parts.append("</ul></div>")
        
        html_parts.extend([
            "</div>",  # Close explanation-container
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def _make_serializable(self, obj):
        """Recursively convert numpy arrays and other non-serializable objects to serializable types."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj


def generate_retrieval_explanation(
    query: str,
    results: List[Dict[str, Any]],
    attention_weights: Dict[str, np.ndarray],
    interpretations: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to generate a retrieval explanation.
    
    Args:
        query: The original query
        results: The retrieval results
        attention_weights: Attention weights from different heads
        interpretations: Interpretation data from the attention mechanism
    
    Returns:
        Dictionary containing the explanation
    """
    explainer = RetrievalExplainer()
    return explainer.generate_explanation(query, results, attention_weights, interpretations)


def format_explanation(explanation: Dict[str, Any], format_type: str = 'text') -> str:
    """
    Format an explanation in the specified format.
    
    Args:
        explanation: The explanation dictionary
        format_type: The format to use ('text', 'json', or 'html')
    
    Returns:
        Formatted explanation as string
    """
    formatter = ExplanationFormatter()
    
    if format_type.lower() == 'text':
        return formatter.format_as_text(explanation)
    elif format_type.lower() == 'json':
        return formatter.format_as_json(explanation)
    elif format_type.lower() == 'html':
        return formatter.format_as_html(explanation)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")