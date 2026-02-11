"""
Attention Visualization and Debugging Tools for EKM
Provides utilities for visualizing attention weights and debugging the retrieval process.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import io
import base64
from dataclasses import dataclass
import pandas as pd


@dataclass
class AttentionDebugInfo:
    """Container for attention debugging information."""
    query: str
    gku_names: List[str]
    attention_weights: Dict[str, np.ndarray]  # semantic, structural, temporal
    head_contributions: Dict[str, float]
    raw_scores: Dict[str, np.ndarray]
    normalized_scores: Dict[str, np.ndarray]


class AttentionVisualizer:
    """Visualization tools for attention mechanisms."""
    
    def __init__(self):
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_attention_weights(
        self, 
        debug_info: AttentionDebugInfo, 
        figsize: tuple = (12, 8)
    ) -> str:
        """
        Create a visualization of attention weights across different heads.
        
        Args:
            debug_info: AttentionDebugInfo object containing attention data
            figsize: Figure size as (width, height)
        
        Returns:
            Base64-encoded string of the plot image
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Attention Weights Visualization for Query: "{debug_info.query[:50]}..."', fontsize=14)
        
        # Plot 1: Semantic attention weights
        if 'semantic' in debug_info.attention_weights:
            axes[0, 0].bar(range(len(debug_info.gku_names)), debug_info.attention_weights['semantic'])
            axes[0, 0].set_title('Semantic Attention Weights')
            axes[0, 0].set_xlabel('GKU Index')
            axes[0, 0].set_ylabel('Attention Weight')
            axes[0, 0].set_xticks(range(len(debug_info.gku_names)))
            axes[0, 0].set_xticklabels([name[:10] + "..." for name in debug_info.gku_names], rotation=45, ha="right")
        
        # Plot 2: Structural attention weights
        if 'structural' in debug_info.attention_weights:
            axes[0, 1].bar(range(len(debug_info.gku_names)), debug_info.attention_weights['structural'])
            axes[0, 1].set_title('Structural Attention Weights')
            axes[0, 1].set_xlabel('GKU Index')
            axes[0, 1].set_ylabel('Attention Weight')
            axes[0, 1].set_xticks(range(len(debug_info.gku_names)))
            axes[0, 1].set_xticklabels([name[:10] + "..." for name in debug_info.gku_names], rotation=45, ha="right")
        
        # Plot 3: Temporal attention weights
        if 'temporal' in debug_info.attention_weights:
            axes[1, 0].bar(range(len(debug_info.gku_names)), debug_info.attention_weights['temporal'])
            axes[1, 0].set_title('Temporal Attention Weights')
            axes[1, 0].set_xlabel('GKU Index')
            axes[1, 0].set_ylabel('Attention Weight')
            axes[1, 0].set_xticks(range(len(debug_info.gku_names)))
            axes[1, 0].set_xticklabels([name[:10] + "..." for name in debug_info.gku_names], rotation=45, ha="right")
        
        # Plot 4: Head contributions
        if debug_info.head_contributions:
            heads = list(debug_info.head_contributions.keys())
            contribs = list(debug_info.head_contributions.values())
            axes[1, 1].pie(contribs, labels=heads, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Attention Head Contributions')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        
        return img_str
    
    def create_attention_heatmap(
        self, 
        debug_info: AttentionDebugInfo, 
        figsize: tuple = (10, 6)
    ) -> str:
        """
        Create a heatmap showing attention weights across all heads for each GKU.
        
        Args:
            debug_info: AttentionDebugInfo object containing attention data
            figsize: Figure size as (width, height)
        
        Returns:
            Base64-encoded string of the plot image
        """
        # Prepare data for heatmap
        heads = []
        gku_names = debug_info.gku_names
        weights_matrix = []
        
        for head_name in ['semantic', 'structural', 'temporal']:
            if head_name in debug_info.attention_weights:
                heads.append(head_name)
                weights_matrix.append(debug_info.attention_weights[head_name])
        
        if not weights_matrix:
            # Return empty plot if no attention weights
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No attention weights to display', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        else:
            weights_matrix = np.array(weights_matrix)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                weights_matrix, 
                xticklabels=gku_names, 
                yticklabels=heads,
                annot=True, 
                fmt='.3f', 
                cmap='viridis',
                ax=ax,
                cbar_kws={'label': 'Attention Weight'}
            )
            ax.set_title('Attention Weights Heatmap')
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        
        return img_str
    
    def generate_debug_report(self, debug_info: AttentionDebugInfo) -> str:
        """
        Generate a text-based debug report for attention computation.
        
        Args:
            debug_info: AttentionDebugInfo object containing attention data
        
        Returns:
            Formatted debug report as string
        """
        report = []
        report.append("=" * 60)
        report.append("EKM ATTENTION DEBUG REPORT")
        report.append("=" * 60)
        report.append(f"Query: {debug_info.query}")
        report.append("")
        
        report.append("GKU NAMES:")
        for i, name in enumerate(debug_info.gku_names):
            report.append(f"  {i}: {name}")
        report.append("")
        
        report.append("ATTENTION WEIGHTS BY HEAD:")
        for head_name in ['semantic', 'structural', 'temporal']:
            if head_name in debug_info.attention_weights:
                weights = debug_info.attention_weights[head_name]
                report.append(f"  {head_name.upper()} HEAD:")
                for i, weight in enumerate(weights):
                    report.append(f"    GKU {i}: {weight:.4f}")
                report.append("")
        
        report.append("HEAD CONTRIBUTIONS:")
        for head_name, contribution in debug_info.head_contributions.items():
            report.append(f"  {head_name}: {contribution:.4f}")
        report.append("")
        
        report.append("RAW SCORES:")
        for head_name, scores in debug_info.raw_scores.items():
            report.append(f"  {head_name}: {scores}")
        report.append("")
        
        report.append("NORMALIZED SCORES:")
        for head_name, scores in debug_info.normalized_scores.items():
            report.append(f"  {head_name}: {scores}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class AttentionDebugger:
    """Debugging tools for attention mechanisms."""
    
    def __init__(self):
        self.visualizer = AttentionVisualizer()
    
    def create_debug_info(
        self,
        query: str,
        gku_names: List[str],
        attention_result: Dict[str, Any]
    ) -> AttentionDebugInfo:
        """
        Create AttentionDebugInfo from attention computation results.
        
        Args:
            query: The input query
            gku_names: Names of the GKUs involved
            attention_result: Result from attention computation
        
        Returns:
            AttentionDebugInfo object
        """
        # Extract attention weights from result
        attention_weights = attention_result.get('weights', {})
        
        # Extract head contributions from interpretations
        interpretations = attention_result.get('interpretations', {})
        head_contributions = interpretations.get('head_weights_used', {})
        
        # Extract raw scores
        raw_scores = {
            'semantic': interpretations.get('semantic_contributions', []),
            'structural': interpretations.get('structural_contributions', []),
            'temporal': interpretations.get('temporal_contributions', [])
        }
        
        # Calculate normalized scores
        normalized_scores = {}
        for head, scores in raw_scores.items():
            if len(scores) > 0:
                scores_array = np.array(scores)
                # Normalize to sum to 1
                normalized_scores[head] = scores_array / (np.sum(scores_array) + 1e-8)
            else:
                normalized_scores[head] = np.array([])
        
        return AttentionDebugInfo(
            query=query,
            gku_names=gku_names,
            attention_weights=attention_weights,
            head_contributions=head_contributions,
            raw_scores=raw_scores,
            normalized_scores=normalized_scores
        )
    
    def debug_retrieval_step(
        self,
        step_name: str,
        step_data: Any,
        description: str = ""
    ) -> None:
        """
        Log debugging information for a retrieval step.
        
        Args:
            step_name: Name of the retrieval step
            step_data: Data associated with the step
            description: Optional description of the step
        """
        print(f"\n[DEBUG] Step: {step_name}")
        if description:
            print(f"[DEBUG] Description: {description}")
        print(f"[DEBUG] Data type: {type(step_data)}")
        
        if isinstance(step_data, dict):
            print("[DEBUG] Keys:", list(step_data.keys()))
            for key, value in list(step_data.items())[:5]:  # Show first 5 items
                print(f"[DEBUG]   {key}: {type(value)} = {str(value)[:100]}...")
        elif isinstance(step_data, list):
            print(f"[DEBUG] Length: {len(step_data)}")
            for i, item in enumerate(step_data[:3]):  # Show first 3 items
                print(f"[DEBUG]   [{i}]: {type(item)} = {str(item)[:100]}...")
        else:
            print(f"[DEBUG] Value: {step_data}")
    
    def validate_attention_weights(self, attention_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Validate attention weights to ensure they are properly normalized.
        
        Args:
            attention_weights: Dictionary of attention weights by head
        
        Returns:
            Validation results
        """
        results = {}
        
        for head_name, weights in attention_weights.items():
            if isinstance(weights, np.ndarray) and len(weights) > 0:
                sum_weights = np.sum(weights)
                is_normalized = np.isclose(sum_weights, 1.0, atol=1e-6)
                min_weight = np.min(weights)
                max_weight = np.max(weights)
                
                results[head_name] = {
                    'sum': float(sum_weights),
                    'is_normalized': is_normalized,
                    'min_weight': float(min_weight),
                    'max_weight': float(max_weight),
                    'valid_range': float(min_weight) >= 0.0 and float(max_weight) <= 1.0
                }
            else:
                results[head_name] = {
                    'sum': 0.0,
                    'is_normalized': False,
                    'min_weight': 0.0,
                    'max_weight': 0.0,
                    'valid_range': True
                }
        
        return results


# Utility function to integrate visualization into the EKM system
def create_attention_visualization(query: str, attention_result: Dict[str, Any], gku_names: List[str]) -> Dict[str, str]:
    """
    Create visualizations for attention results.
    
    Args:
        query: The input query
        attention_result: Result from attention computation
        gku_names: Names of the GKUs involved
    
    Returns:
        Dictionary containing visualization images as base64 strings
    """
    debugger = AttentionDebugger()
    debug_info = debugger.create_debug_info(query, gku_names, attention_result)
    
    visualizer = AttentionVisualizer()
    
    visualizations = {}
    try:
        visualizations['attention_weights'] = visualizer.visualize_attention_weights(debug_info)
        visualizations['attention_heatmap'] = visualizer.create_attention_heatmap(debug_info)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        visualizations['error'] = str(e)
    
    return visualizations