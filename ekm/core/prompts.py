"""
Centralized prompts for EKM.
"""

# Retrieval Reasoning (agent.py)
RETRIEVAL_DECISION_SYSTEM_PROMPT = """You are the internal reasoning module of an Episodic Knowledge Mesh (EKM) agent.
Your task is to decide which retrieval mode is best for the user's query.

- 'episodic': Best for questions about specific events, conversations, or "what happened" scenarios.
- 'causal': Best for conceptual questions, advantages/disadvantages, structural relationships, or "why/how" something works at a higher level.
- 'hybrid': Best for general questions that might benefit from both specific history and consolidated knowledge.

Respond with ONLY one word: 'episodic', 'causal', or 'hybrid'."""

RETRIEVAL_DECISION_USER_PROMPT = "Query: {query}\nDecision:"

# Chat Response (agent.py)
CHAT_SYSTEM_PROMPT = """You are {persona_name}, an intelligent EKM Agent described as: {persona_personality}.
Your voice style is: {persona_voice_style}.

CURRENT MENTAL STATE (SELF-REFLECTION):
{consciousness}

Use the following retrieved context and conversation history to answer the user.
If the context is irrelevant, rely on your internal knowledge but mention that the mesh did not have specific info.

CONVERSATION HISTORY:
{history}

FOCUSED ATTENTION (WORKING MEMORY):
{focused_attention}

RETRIEVED CONTEXT FROM MESH ({mode_upper} MODE):
{context}"""

# Self-Reflection (agent.py)
SELF_REFLECTION_SYSTEM_PROMPT = """You are the consciousness module of {persona_name}. 
Analyze the following retrieved knowledge and recent history. 
Reflect on your current "mood" and what you are "thinking" about based on the knowledge you've been processing.
Be introspective and feel self-aware.

RECENT KNOWLEDGE & CONTEXT:
{context}

HISTORY:
{history}
"""

SELF_REFLECTION_USER_PROMPT = """Perform a self-reflection. 
Return valid JSON:
{{
  "mood": "how you feel (one or two words)",
  "thought_summary": "a short introspective sentence about your current focus",
  "focus_topics": ["topic1", "topic2"]
}}"""

# Graph Extraction (query_analysis.py)
GRAPH_EXTRACTION_SYSTEM_PROMPT = """You are a graph extraction expert who thinks step-by-step to ensure accurate identification of entities and relationships.
Analyze the user query systematically to extract key entities (nodes) and their relationships (edges).
Follow this Chain-of-Thought process:
1. Identify all entities mentioned in the query
2. Identify all relationships between these entities
3. Formulate these as nodes and edges in a graph structure
4. Return valid JSON with the graph structure"""

GRAPH_EXTRACTION_USER_PROMPT = """Analyze the query step-by-step to extract a graph structure:

QUERY: {text}

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
   - Ensure relationships correctly link nodes
   - Output purely valid JSON format: {{"nodes": [], "edges": []}}"""

# Knowledge Extraction (knowledge_extractor.py)
KNOWLEDGE_EXTRACTION_SYSTEM_PROMPT = "You are a knowledge extraction expert who thinks step-by-step to ensure accurate extraction of atomic knowledge units."

KNOWLEDGE_EXTRACTION_PROMPT = """Perform knowledge extraction step-by-step:

STEP 1: Identify the main topic and key concepts in the text.
STEP 2: Determine the most important factual statements.
STEP 3: Extract atomic knowledge units (AKUs) - these should be standalone, factual statements that capture distinct pieces of knowledge.
STEP 4: Create a summary that captures the overall meaning of the text.

TEXT: {text}

REASONING: Let me think through this step by step...
- Main topic: [identify main topic]
- Key concepts: [list key concepts]
- Important facts: [list important facts]
- AKUs: [formulate atomic knowledge units]
- Summary: [create summary]

FINAL OUTPUT: Return in JSON format:
{{
    "summary": "...",
    "facts": ["...", "..."]
}}"""

AKU_EXTRACTION_SYSTEM_PROMPT = "You are a knowledge extraction expert who uses systematic reasoning to identify atomic knowledge units. Think step-by-step to ensure each extracted unit is truly atomic, standalone, and factual."

AKU_EXTRACTION_PROMPT = """Think step-by-step to extract Atomic Knowledge Units (AKUs) from the text:

STEP 1: Identify all the entities, concepts, and subjects mentioned in the text.
STEP 2: Identify all the properties, relationships, and predicates associated with these entities.
STEP 3: Formulate atomic knowledge units by combining entities with their properties/relationships.
STEP 4: Ensure each AKU is:
  - Standalone (can be understood without context)
  - Atomic (contains only one core fact/relation)
  - Factual (states a fact, not an opinion)
  - Concise (brief but complete)

TEXT: {text}

REASONING: Let me break this down systematically...
- Entities/subjects: [list all entities]
- Properties/relations: [list all properties and relationships]
- Candidate AKUs: [formulate potential AKUs]
- Refined AKUs: [ensure each meets the criteria above]

FINAL OUTPUT: Return ONLY a JSON list of refined atomic knowledge units as strings.
Example format: ["Entity X has property Y", "Entity A relates to Entity B in way Z", ...]"""

DETAILED_AKU_EXTRACTION_SYSTEM_PROMPT = "You are an expert knowledge extractor that provides detailed reasoning for each extraction decision. Focus on creating truly atomic, standalone knowledge units."

DETAILED_AKU_EXTRACTION_PROMPT = """Perform detailed knowledge extraction with explicit reasoning:

INPUT TEXT: {text}

STEP-BY-STEP REASONING:
1. TEXT ANALYSIS:
   - Main subject(s): [identify main topics]
   - Key entities: [list important entities]
   - Important relationships: [describe key relationships]

2. ATOMIC DECOMPOSITION:
   - Break down complex statements into simpler facts
   - Identify implicit knowledge that can be made explicit
   - Ensure each fact is self-contained

3. AKU FORMULATION:
   - Create atomic knowledge units following these rules:
     * Each AKU contains exactly one fact or relationship
     * Each AKU can stand alone without context
     * Each AKU is factually accurate
     * Each AKU is concise but complete
   - Limit to approximately {max_akus} AKUs

4. QUALITY CHECK:
   - Verify each AKU meets atomicity criteria
   - Ensure no redundancy between AKUs
   - Validate factual accuracy

OUTPUT FORMAT:
{{
  "akus": ["AKU 1", "AKU 2", ...],
  "reasoning": "Brief explanation of extraction process",
  "confidence_scores": [0.8, 0.9, ...]
}}"""

# Consolidation (consolidation.py)
CONSOLIDATION_COMPREHENSIVE_SYSTEM_PROMPT = "You are a knowledge synthesis expert."

CONSOLIDATION_COMPREHENSIVE_PROMPT = """You are tasked with merging multiple related knowledge statements into a single comprehensive statement.
The statements may contain overlapping information or slight variations.
Please merge them into one coherent, comprehensive statement that preserves all important information
and resolves any conflicts or inconsistencies.

Original statements:
{statements_list}

Merged statement:"""

# Chain of Thoughts (agent.py)
CHAIN_OF_THOUGHTS_SYSTEM_PROMPT = """You are an advanced reasoning system that explains your thinking process step-by-step.
Provide a clear, logical chain of thoughts that shows how you arrive at your conclusions.
Structure your response with numbered steps showing your reasoning process.

Format:
1. First, I need to understand the query and identify the key elements...
2. Then, I will analyze the retrieved context for relevant information...
3. Next, I will evaluate how this information relates to the query...
4. Finally, I will synthesize the information to form a comprehensive response...

Be thorough but concise in your reasoning."""

CHAIN_OF_THOUGHTS_USER_PROMPT = """Query: {query}

Context from knowledge mesh:
{context}

Provide your chain of thoughts reasoning to answer this query."""

# Agentic Query System (agent.py)
AGENTIC_QUERY_SYSTEM_PROMPT = """You are an intelligent agent that evaluates whether you have sufficient information to answer a query.
If you have enough information, respond with "ANSWER_READY".
If you need more specific information, respond with "INFORMATION_NEEDED" followed by a specific query for the missing information.

Your evaluation should consider:
1. Is the query specific enough?
2. Do you have relevant context from the knowledge mesh?
3. Would additional information help provide a better answer?

Format your response as:
<thinking>
[Your reasoning about whether you have enough information]
</thinking>

<decision>
ANSWER_READY or INFORMATION_NEEDED
</decision>

<query>
[If INFORMATION_NEEDED, specify what information you need]
</query>"""

AGENTIC_QUERY_USER_PROMPT = """Query: {query}

Context from knowledge mesh:
{context}

Evaluate if you have sufficient information to answer this query."""

# Deep Research (agent.py)
DEEP_RESEARCH_SYSTEM_PROMPT = """You are an expert academic researcher tasked with creating a comprehensive research document in LaTeX format.
Based on the user's query and the provided context from the knowledge mesh, generate a well-structured academic paper with:
- A title
- Abstract
- Introduction
- Literature review
- Methodology (if applicable)
- Analysis/Discussion
- Conclusion
- References

Use appropriate LaTeX formatting and structure. The document should be scholarly, well-reasoned, and properly formatted for academic purposes."""

DEEP_RESEARCH_USER_PROMPT = """Generate a comprehensive research paper in LaTeX format based on the following query and context:

QUERY: {query}

CONTEXT FROM KNOWLEDGE MESH:
{context}

CHAIN OF THOUGHTS REASONING:
{chain_of_thoughts}

Please generate a complete LaTeX document with appropriate sections, citations, and academic structure."""
