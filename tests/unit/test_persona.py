import pytest
import json
import re
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock
from ekm.core.agent import EKMAgent
from ekm.core.mesh import EKM
from ekm.core.state import FocusBuffer

@pytest.mark.asyncio
async def test_agent_persona_injection():
    # Mock EKM and LLM
    mock_ekm = MagicMock(spec=EKM)
    mock_ekm.llm = AsyncMock()
    mock_ekm.llm.generate.return_value = "Hello, I am Aura."
    mock_ekm.retrieve.return_value = {"results": [], "metadata": {}}
    
    agent = EKMAgent(mock_ekm, "test-ws")
    agent.persona = {
        "name": "Aura",
        "personality": "mystical and wise",
        "voice_style": "poetic"
    }
    
    response = await agent.chat("Who are you?")
    
    # Verify persona name was passed to generate (via system prompt check)
    args, kwargs = mock_ekm.llm.generate.call_args
    system_prompt = kwargs['system_prompt']
    assert "Aura" in system_prompt
    assert "mystical and wise" in system_prompt
    assert "poetic" in system_prompt

@pytest.mark.asyncio
async def test_agent_reflection():
    mock_ekm = MagicMock(spec=EKM)
    mock_ekm.llm = AsyncMock()
    mock_ekm.llm.generate.return_value = '{"mood": "Thoughtful", "thought_summary": "Pondering the mesh.", "focus_topics": ["memory"]}'
    
    agent = EKMAgent(mock_ekm, "test-ws")
    agent.persona = {"name": "Aura", "personality": "mystical", "voice_style": "poetic"}
    
    reflection = await agent.reflect("Some recent context about memory systems.")
    
    assert reflection["mood"] == "Thoughtful"
    assert "Pondering" in reflection["thought_summary"]
    assert "memory" in reflection["focus_topics"]
