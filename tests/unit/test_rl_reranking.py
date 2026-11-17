"""
The RL reranker must not change rankings before it has learned anything,
must be deterministic by default, and must not re-embed candidates that
already carry an embedding.
"""
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from ekm.core.rl_feedback import RLFeedbackSystem


def make_system(saved_state=None, **kwargs):
    storage = MagicMock()
    storage.load_rl_state = AsyncMock(return_value=saved_state)
    storage.save_rl_state = AsyncMock()
    embeddings = MagicMock()
    embeddings.embed_query = AsyncMock(return_value=[1.0, 0.0])
    return RLFeedbackSystem(storage=storage, llm=MagicMock(), embeddings=embeddings, **kwargs), embeddings


# Retrieval ranked 'b' first (e.g. strong GKU context) although 'a' is closer
# to the query by cosine alone.
CANDIDATES = [
    {'id': 'a', 'content': 'a', 'embedding': [1.0, 0.0], 'score': 0.6},
    {'id': 'b', 'content': 'b', 'embedding': '[0.6, 0.8]', 'score': 0.9},   # JSON string, as stored
    {'id': 'c', 'content': 'c', 'embedding': [0.0, 1.0], 'score': 0.1},
]


@pytest.mark.asyncio
async def test_cold_start_keeps_retrieval_order_without_embedding_calls():
    system, embeddings = make_system()

    prediction = await system.get_ranked_responses('q', CANDIDATES)

    assert [r['id'] for r in prediction.ranked_responses] == ['b', 'a', 'c']
    assert prediction.confidence_scores == [0.9, 0.6, 0.1]
    assert prediction.exploration_factor == 0.0
    embeddings.embed_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_exploration_is_off_by_default():
    system, _ = make_system(saved_state={'weights': [1.0] + [0.0] * 9})
    assert system.exploration_rate == 0.0

    orders = set()
    for _ in range(20):
        prediction = await system.get_ranked_responses('q', CANDIDATES)
        orders.add(tuple(r['id'] for r in prediction.ranked_responses))
        assert prediction.exploration_factor == 0.0

    assert len(orders) == 1


@pytest.mark.asyncio
async def test_loaded_weights_enable_reranking_and_skip_reembedding():
    # Weights that rank purely by query cosine
    system, embeddings = make_system(saved_state={'weights': [1.0] + [0.0] * 9})

    prediction = await system.get_ranked_responses('q', CANDIDATES)

    assert system.has_learned
    assert [r['id'] for r in prediction.ranked_responses] == ['a', 'b', 'c']
    # Only the query is embedded; every candidate already had an embedding
    assert embeddings.embed_query.await_count == 1


@pytest.mark.asyncio
async def test_recorded_feedback_turns_learning_on():
    system, _ = make_system()
    assert not system.has_learned

    # The untrained model predicts ~1.0 for this exact match, so a negative
    # rating gives a non-zero error and moves the weights.
    assert await system.record_feedback('q', 'a', 'a', user_rating=-1.0)

    assert system.has_learned
    assert not np.allclose(system.weights, [1.0] + [0.0] * 9)


@pytest.mark.asyncio
async def test_candidates_without_embedding_are_embedded_once():
    system, embeddings = make_system(saved_state={'weights': [1.0] + [0.0] * 9})
    candidates = [{'id': 'x', 'content': 'x', 'score': 0.5}, {'id': 'y', 'content': 'y', 'embedding': [1.0, 0.0]}]

    await system.get_ranked_responses('q', candidates)

    assert embeddings.embed_query.await_count == 2   # the query and 'x'
