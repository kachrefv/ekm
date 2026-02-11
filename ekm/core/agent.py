"""
EKM Agent module - Implements an autonomous agent that manages its own retrieval strategies.
"""
import logging
import json
import re
from typing import Dict, Any, List, Optional
from .mesh import EKM
from .utils import estimate_tokens, truncate_text_tokens
from .state import FocusBuffer
from . import prompts
from .task_manager import TaskManager

logger = logging.getLogger(__name__)

class EKMAgent:
    """
    An autonomous agent powered by EKM that can decide how to retrieve knowledge
    based on the nature of the query.
    """

    def __init__(self, ekm: EKM, workspace_id: str):
        self.ekm = ekm
        self.workspace_id = workspace_id
        self.history = []
        self.focus_buffer = FocusBuffer()
        self.persona = {
            "name": "EKM Assistant",
            "personality": "helpful, curious",
            "voice_style": "professional"
        }
        self.current_consciousness = None
        
        # Initialize TaskManager with DB session if available
        db_session = getattr(self.ekm.storage, 'db', None)
        self.task_manager = TaskManager(db_session=db_session, workspace_id=workspace_id)
        
    async def think_about_retrieval(self, query: str) -> str:
        """
        Decides the best retrieval mode ('episodic', 'causal', or 'hybrid') for the query.
        """
        try:
            decision = await self.ekm.llm.generate(
                system_prompt=prompts.RETRIEVAL_DECISION_SYSTEM_PROMPT,
                user_message=prompts.RETRIEVAL_DECISION_USER_PROMPT.format(query=query)
            )
            
            decision = decision.strip().lower()
            if 'episodic' in decision: return 'episodic'
            if 'causal' in decision: return 'causal'
            return 'hybrid'
        except Exception as e:
            logger.warning(f"Reasoning failed, defaulting to hybrid: {e}")
            return 'hybrid'

    async def chat(self, user_query: str, include_chain_of_thoughts: bool = False, use_agentic_system: bool = False) -> Dict[str, Any]:
        """
        Autonomous chat loop iteration: Think -> Retrieve -> Respond.
        """
        if use_agentic_system:
            # Use the iterative agentic query system
            return await self.iterative_query_with_evaluation(user_query)
        
        # 1. Decide retrieval mode
        mode = await self.think_about_retrieval(user_query)
        logger.info(f"Agent decided on retrieval mode: {mode}")

        # 2. Retrieve knowledge
        retrieval_response = await self.ekm.retrieve(
            workspace_id=self.workspace_id,
            query=user_query,
            mode=mode,
            top_k=5,
            focus_buffer=self.focus_buffer,
            debug_mode=True
        )

        # 3. Focus Buffer Update and Context Injection
        retrieved_aku_ids = [res['id'] for res in retrieval_response.get('results', []) if res.get('id')]
        self.focus_buffer.update(retrieved_aku_ids)

        # Build Focused Attention context
        focused_items = list(self.focus_buffer.items.values())
        focused_items.sort(key=lambda x: x.current_weight, reverse=True)
        top_focus = focused_items[:3] # Keep it brief

        focused_context = ""
        if top_focus:
            # Need to fetch content for these IDs if not in retrieval results
            focused_ids = [it.aku_id for it in top_focus]
            # Optimization: check if we already have content in retrieval_response
            content_map = {res['id']: res['content'] for res in retrieval_response.get('results', []) if res.get('id')}

            focused_lines = []
            for item in top_focus:
                content = content_map.get(item.aku_id)
                if not content:
                    # Fallback fetch from storage (synchronous wrapper or async call)
                    # For now, we'll try to get it from storage if missing
                    obj_list = await self.ekm.storage.get_akus_by_ids([item.aku_id])
                    if obj_list:
                        content = obj_list[0]['content']

                if content:
                    focused_lines.append(f"- {content} (weight: {item.current_weight:.2f})")

            if focused_lines:
                focused_context = "\n".join(focused_lines)

        # 4. Schema-Aware Context Injection
        # Format context as a Markdown table for better LLM parsing
        context_table = "| Source | Recency | Content |\n| :--- | :--- | :--- |\n"
        has_results = False

        for res in retrieval_response.get('results', []):
            has_results = True
            source = res.get('source_gku') or res.get('metadata', {}).get('gku_name', 'Unknown')
            content = res.get('content', '').replace('\n', ' ')
            created_at = res.get('created_at', 'Unknown')

            # Format recency if possible
            recency = str(created_at).split('.')[0] if created_at else "N/A"

            context_table += f"| {source} | {recency} | {content} |\n"

        if not has_results:
            context = "No relevant context found in the mesh."
        else:
            context = context_table

        # 4.5. Format History
        history_str = ""
        if self.history:
            history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.history])

        # 5. Token Management: Truncate context if it's too long
        max_context_tokens = 4000
        context = truncate_text_tokens(context, max_context_tokens)

        # 6. Generate response using centralized prompt
        consciousness_str = "I am ready to help."
        if self.current_consciousness:
            cons = self.current_consciousness
            mood = getattr(cons, 'mood', 'Stable')
            thought = getattr(cons, 'thought_summary', 'I am processing knowledge.')
            consciousness_str = f"Current mood: {mood}\nThinking about: {thought}"

        system_prompt = prompts.CHAT_SYSTEM_PROMPT.format(
            persona_name=self.persona["name"],
            persona_personality=self.persona["personality"],
            persona_voice_style=self.persona["voice_style"],
            consciousness=consciousness_str,
            mode_upper=mode.upper(),
            context=context,
            focused_attention=focused_context or "None",
            history=history_str or "No previous messages in this session."
        )

        # Generate chain of thoughts if requested
        chain_of_thoughts = ""
        if include_chain_of_thoughts:
            chain_of_thoughts = await self.generate_chain_of_thoughts(user_query, context)

        try:
            response = await self.ekm.llm.generate(
                system_prompt=system_prompt,
                user_message=user_query
            )

            return {
                "response": response,
                "mode_used": mode,
                "retrieval_results": retrieval_response['results'],
                "chain_of_thoughts": chain_of_thoughts if include_chain_of_thoughts else None,
                "metadata": {
                    **retrieval_response.get('metadata', {}),
                    "context_tokens_estimated": estimate_tokens(context),
                    "focus_buffer_size": len(self.focus_buffer.items),
                    "include_chain_of_thoughts": include_chain_of_thoughts,
                    "use_agentic_system": use_agentic_system
                }
            }
        except Exception as e:
            return {"error": str(e)}

    async def reflect(self, recent_context: str) -> Dict[str, Any]:
        """
        The agent reflects on its recent knowledge and state to update its consciousness.
        """
        try:
            history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.history[-5:]])
            system_prompt = prompts.SELF_REFLECTION_SYSTEM_PROMPT.format(
                persona_name=self.persona["name"],
                context=recent_context,
                history=history_str or "No recent history."
            )

            reflection_json = await self.ekm.llm.generate(
                system_prompt=system_prompt,
                user_message=prompts.SELF_REFLECTION_USER_PROMPT
            )

            # Extract JSON from response
            match = re.search(r'\{.*\}', reflection_json, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(reflection_json)
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return {"mood": "Stable", "thought_summary": "I am processing knowledge.", "focus_topics": []}

    async def generate_chain_of_thoughts(self, query: str, context: str) -> str:
        """
        Generates a step-by-step reasoning chain for the given query and context.
        """
        try:
            system_prompt = prompts.CHAIN_OF_THOUGHTS_SYSTEM_PROMPT
            user_message = prompts.CHAIN_OF_THOUGHTS_USER_PROMPT.format(query=query, context=context)
            
            cot_reasoning = await self.ekm.llm.generate(
                system_prompt=system_prompt,
                user_message=user_message
            )
            
            return cot_reasoning
        except Exception as e:
            logger.error(f"Chain of thoughts generation failed: {e}")
            return "Chain of thoughts reasoning is unavailable at this time."

    async def evaluate_information_readiness(self, query: str, context: str) -> Dict[str, Any]:
        """
        Evaluates whether the agent has sufficient information to answer the query.
        Returns a dictionary with decision, reasoning, and if needed, a follow-up query.
        """
        try:
            system_prompt = prompts.AGENTIC_QUERY_SYSTEM_PROMPT
            user_message = prompts.AGENTIC_QUERY_USER_PROMPT.format(query=query, context=context)
            
            evaluation_result = await self.ekm.llm.generate(
                system_prompt=system_prompt,
                user_message=user_message
            )
            
            # Parse the response to extract decision and query
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', evaluation_result, re.DOTALL)
            decision_match = re.search(r'<decision>(.*?)</decision>', evaluation_result, re.DOTALL)
            query_match = re.search(r'<query>(.*?)</query>', evaluation_result, re.DOTALL)
            
            thinking = thinking_match.group(1).strip() if thinking_match else "No reasoning provided"
            decision = decision_match.group(1).strip() if decision_match else "ANSWER_READY"
            follow_up_query = query_match.group(1).strip() if query_match else ""
            
            return {
                "ready": decision.strip().upper() == "ANSWER_READY",
                "reasoning": thinking,
                "follow_up_query": follow_up_query,
                "full_evaluation": evaluation_result
            }
        except Exception as e:
            logger.error(f"Information readiness evaluation failed: {e}")
            return {
                "ready": True,  # Default to ready if evaluation fails
                "reasoning": f"Evaluation failed due to error: {e}",
                "follow_up_query": "",
                "full_evaluation": ""
            }

    async def iterative_query_with_evaluation(self, initial_query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Performs iterative querying with evaluation of information sufficiency.
        The agent will continue to query the knowledge mesh until it decides it has enough information
        or reaches the maximum number of iterations.
        """
        current_query = initial_query
        all_contexts = []
        all_evaluations = []
        iteration_count = 0
        
        while iteration_count < max_iterations:
            # Retrieve knowledge for current query
            retrieval_response = await self.ekm.retrieve(
                workspace_id=self.workspace_id,
                query=current_query,
                mode="hybrid",  # Using hybrid mode for comprehensive retrieval
                top_k=5,
                focus_buffer=self.focus_buffer,
                debug_mode=True
            )
            
            # Update focus buffer
            retrieved_aku_ids = [res['id'] for res in retrieval_response.get('results', []) if res.get('id')]
            self.focus_buffer.update(retrieved_aku_ids)
            
            # Build context from retrieval results
            context_table = "| Source | Recency | Content |\n| :--- | :--- | :--- |\n"
            has_results = False
            
            for res in retrieval_response.get('results', []):
                has_results = True
                source = res.get('source_gku') or res.get('metadata', {}).get('gku_name', 'Unknown')
                content = res.get('content', '').replace('\n', ' ')
                created_at = res.get('created_at', 'Unknown')
                
                # Format recency if possible
                recency = str(created_at).split('.')[0] if created_at else "N/A"
                
                context_table += f"| {source} | {recency} | {content} |\n"
            
            if not has_results:
                context = "No relevant context found in the mesh."
            else:
                context = context_table
            
            # Evaluate if we have enough information
            evaluation = await self.evaluate_information_readiness(current_query, context)
            all_evaluations.append({
                "iteration": iteration_count + 1,
                "query": current_query,
                "evaluation": evaluation
            })
            
            # Add context to the collection
            all_contexts.append({
                "iteration": iteration_count + 1,
                "context": context,
                "has_results": has_results
            })
            
            # If ready or no follow-up query is suggested, break the loop
            if evaluation["ready"] or not evaluation["follow_up_query"].strip():
                break
            
            # Update the query for the next iteration
            current_query = evaluation["follow_up_query"]
            iteration_count += 1
        
        # Generate final response based on all collected contexts
        combined_context = "\n\n".join([ctx["context"] for ctx in all_contexts])
        
        # Generate chain of thoughts if needed
        chain_of_thoughts = await self.generate_chain_of_thoughts(initial_query, combined_context)
        
        # Generate final response
        consciousness_str = "I am ready to help."
        if self.current_consciousness:
            cons = self.current_consciousness
            mood = getattr(cons, 'mood', 'Stable')
            thought = getattr(cons, 'thought_summary', 'I am processing knowledge.')
            consciousness_str = f"Current mood: {mood}\nThinking about: {thought}"
        
        system_prompt = prompts.CHAT_SYSTEM_PROMPT.format(
            persona_name=self.persona["name"],
            persona_personality=self.persona["personality"],
            persona_voice_style=self.persona["voice_style"],
            consciousness=consciousness_str,
            mode_upper="HYBRID",
            context=combined_context,
            focused_attention="",  # Will be populated later if needed
            history=""  # Using empty history for the final response
        )
        
        try:
            response = await self.ekm.llm.generate(
                system_prompt=system_prompt,
                user_message=initial_query
            )
        except Exception as e:
            response = f"Error generating final response: {str(e)}"
        
        # Retrieve final results for the response
        final_retrieval = await self.ekm.retrieve(
            workspace_id=self.workspace_id,
            query=initial_query,
            mode="hybrid",
            top_k=20,  # Get all results from all iterations
            focus_buffer=self.focus_buffer,
            debug_mode=True
        )
        
        return {
            "response": response,
            "mode_used": "iterative_agentic",
            "retrieval_results": final_retrieval.get('results', []),
            "chain_of_thoughts": chain_of_thoughts,
            "metadata": {
                "iterations_completed": iteration_count + 1,
                "evaluations": all_evaluations,
                "contexts_used": all_contexts,
                "final_query": current_query
            }
        }

    async def generate_deep_research_pdf(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Generates a comprehensive research document in LaTeX format based on the query.
        Uses the agentic system to gather information iteratively and incorporates chain of thoughts reasoning.
        """
        # Use the iterative agentic system to gather comprehensive information
        agentic_result = await self.iterative_query_with_evaluation(query, max_iterations=max_iterations)
        
        # Extract the context and chain of thoughts from the agentic result
        combined_context = "\n\n".join([ctx["context"] for ctx in agentic_result["metadata"]["contexts_used"]])
        chain_of_thoughts = agentic_result.get("chain_of_thoughts", "")
        
        # Generate the LaTeX research document
        try:
            system_prompt = prompts.DEEP_RESEARCH_SYSTEM_PROMPT
            user_message = prompts.DEEP_RESEARCH_USER_PROMPT.format(
                query=query,
                context=combined_context,
                chain_of_thoughts=chain_of_thoughts
            )
            
            latex_document = await self.ekm.llm.generate(
                system_prompt=system_prompt,
                user_message=user_message
            )
            
            return {
                "latex_content": latex_document,
                "agentic_data": agentic_result,
                "query": query,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Deep research PDF generation failed: {e}")
            return {
                "latex_content": f"Error generating research document: {str(e)}",
                "agentic_data": agentic_result,
                "query": query,
                "status": "error"
            }
