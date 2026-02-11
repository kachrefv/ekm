"""
Knowledge extraction module for EKM - Handles extraction of AKUs and summaries
"""
import re
import asyncio
from typing import List, Dict, Any, Optional
from ...providers.base import BaseLLM, BaseEmbeddings
from .. import prompts


class KnowledgeExtractor:
    """Handles extraction of Atomic Knowledge Units (AKUs) and summaries."""

    def __init__(self, llm: BaseLLM, embeddings: BaseEmbeddings):
        self.llm = llm
        self.embeddings = embeddings

    async def single_step_extraction(self, text: str) -> Dict[str, Any]:
        """Perform single-step extraction of summary and AKUs."""
        raw_response = await self.llm.generate(
            system_prompt=prompts.KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT,
            user_message=prompts.KNOWLEDGE_EXTRACTION_PROMPT.format(text=text)
        )

        return self._parse_extraction_response(raw_response)

    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract summary and facts."""
        try:
            # Try to find JSON in the response
            import json
            import re

            # Look for JSON between curly braces
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return {
                    'summary': parsed.get('summary', ''),
                    'facts': parsed.get('facts', [])
                }
        except:
            pass

        # Fallback: try to extract facts from numbered lists or bullet points
        facts = []
        lines = response.split('\n')
        for line in lines:
            # Look for numbered or bulleted items
            if re.match(r'^\s*\d+[.)]\s+', line) or re.match(r'^\s*[•\-–]\s+', line):
                fact = re.sub(r'^\s*\d+[.)]\s*|^\\s*[•\-–]\\s*', '', line).strip()
                if fact:
                    facts.append(fact)

        return {
            'summary': response[:200] + "..." if len(response) > 200 else response,
            'facts': facts
        }

    async def extract_akus(self, text: str) -> List[str]:
        """Extract AKUs from text using Chain-of-Thought prompting for improved accuracy."""
        raw_res = await self.llm.generate(
            system_prompt=prompts.AKU_EXTRACTION_SYSTEM_PROMPT,
            user_message=prompts.AKU_EXTRACTION_PROMPT.format(text=text)
        )
        
        try:
            # Basic JSON extraction
            match = re.search(r'\[.*\]', raw_res, re.DOTALL)
            if match:
                import ast
                return ast.literal_eval(match.group(0))  # Use ast.literal_eval instead of eval
        except:
            pass

        return []

    async def extract_akus_with_reasoning(self, text: str, max_akus: int = 10) -> Dict[str, Any]:
        """Extract AKUs with explicit reasoning steps for transparency and validation."""
        raw_response = await self.llm.generate(
            system_prompt=prompts.DETAILED_AKU_EXTRACTION_SYSTEM_PROMPT,
            user_message=prompts.DETAILED_AKU_EXTRACTION_PROMPT.format(text=text, max_akus=max_akus)
        )

        try:
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Return the AKUs and reasoning if available
                if 'akus' in parsed:
                    return {
                        'akus': parsed['akus'],
                        'reasoning': parsed.get('reasoning', ''),
                        'confidence_scores': parsed.get('confidence_scores', [0.8] * len(parsed['akus']))
                    }
                    
        except Exception as e:
            print(f"Error parsing detailed AKU extraction: {e}")
            pass

        # Fallback to basic extraction
        akus = await self.extract_akus(text)
        return {
            'akus': akus,
            'reasoning': 'Basic extraction performed due to parsing issue',
            'confidence_scores': [0.7] * len(akus)
        }