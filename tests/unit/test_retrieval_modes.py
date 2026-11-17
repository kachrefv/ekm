"""
'causal' mode was renamed 'facts' because it never followed cause-and-effect
links. The old name must keep working, with a deprecation warning.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from ekm.core.retrieval.service import RetrievalService, normalize_retrieval_mode


def test_known_modes_pass_through():
    for mode in ('hybrid', 'episodic', 'facts'):
        assert normalize_retrieval_mode(mode) == mode
    assert normalize_retrieval_mode(' Facts ') == 'facts'
    assert normalize_retrieval_mode(None) == 'hybrid'


def test_causal_is_a_deprecated_alias_for_facts():
    with pytest.warns(DeprecationWarning, match="use 'facts'"):
        assert normalize_retrieval_mode('causal') == 'facts'


def test_unknown_mode_falls_back_to_hybrid(caplog):
    assert normalize_retrieval_mode('quantum') == 'hybrid'
    assert "Unknown retrieval mode 'quantum'" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ['facts', 'causal'])
async def test_facts_mode_skips_episodes(mode):
    embeddings = MagicMock()
    embeddings.embed_query = AsyncMock(return_value=[1.0, 0.0])
    service = RetrievalService(storage=MagicMock(), llm=MagicMock(), embeddings=embeddings,
                               scalable_backend=MagicMock(), rl_feedback_system=MagicMock())
    service._retrieve_with_qkv_attention = AsyncMock(return_value={'results': [], 'metadata': {}})
    service._retrieve_episodic_layer = AsyncMock(return_value={'results': [], 'metadata': {}})

    result = await service.retrieve('ws', 'q', mode=mode, use_rl_feedback=False)

    assert result['metadata']['layers_retrieved'] == ['aku', 'gku']
    service._retrieve_episodic_layer.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_maps_old_and_new_answers_to_facts():
    from ekm.core.agent import EKMAgent

    agent = EKMAgent.__new__(EKMAgent)            # skip DB-bound setup
    agent.ekm = MagicMock()
    for answer in ('facts', 'Causal'):
        agent.ekm.llm.generate = AsyncMock(return_value=answer)
        assert await agent.think_about_retrieval('why?') == 'facts'
