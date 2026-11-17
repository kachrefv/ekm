"""
Facts must be retrievable even when their GKU is not among the top few,
and before the first consolidation when there are no GKUs at all.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ekm.core.retrieval.service import RetrievalService


def make_service(storage):
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="no graph")      # empty query pattern
    embeddings = MagicMock()
    embeddings.embed_query = AsyncMock(return_value=[1.0, 0.0])
    return RetrievalService(storage=storage, llm=llm, embeddings=embeddings,
                            scalable_backend=MagicMock(), rl_feedback_system=MagicMock())


def fact(aku_id, embedding, content=None):
    return {'id': aku_id, 'content': content or aku_id, 'embedding': embedding, 'aku_metadata': {}}


async def retrieve(service, top_k=2):
    with patch('ekm.core.explainability.generate_retrieval_explanation', return_value={}):
        return await service._retrieve_with_qkv_attention(
            workspace_id='ws', query='q', top_k=top_k, use_hierarchical_attention=True)


@pytest.mark.asyncio
async def test_facts_are_found_before_first_consolidation():
    facts = [fact('close', [0.9, 0.1]), fact('far', [0.0, 1.0])]
    storage = MagicMock()
    storage.get_gkus_by_workspace = AsyncMock(return_value=[])          # never consolidated
    storage.find_similar_akus = AsyncMock(return_value=[{'id': f['id'], 'similarity': 0.5} for f in facts])
    storage.get_akus_by_ids = AsyncMock(return_value=facts)

    output = await retrieve(make_service(storage))

    ids = [r['id'] for r in output['results']]
    assert ids == ['close', 'far']
    assert all(r['found_by'] == 'direct_search' for r in output['results'])
    assert output['results'][0]['score'] == pytest.approx(0.9 / (0.81 + 0.01) ** 0.5)


@pytest.mark.asyncio
async def test_fact_outside_top_gku_can_outrank_gku_facts():
    gku = {'id': 'g1', 'name': 'Cluster 1', 'description': '', 'centroid_embedding': [0.0, 1.0],
           'pattern_signature': {}, 'cluster_metadata': {}, 'created_at': None}
    in_cluster = fact('in-cluster', [0.0, 1.0])        # orthogonal to the query
    outside = fact('outside', [1.0, 0.0])               # exact match, but in no selected GKU
    storage = MagicMock()
    storage.get_gkus_by_workspace = AsyncMock(return_value=[gku])
    storage.get_gku = AsyncMock(return_value=gku)
    storage.get_akus_in_gku = AsyncMock(return_value=[in_cluster])
    storage.find_similar_akus = AsyncMock(return_value=[{'id': 'outside'}, {'id': 'in-cluster'}])
    storage.get_akus_by_ids = AsyncMock(return_value=[outside])

    output = await retrieve(make_service(storage))

    assert [r['id'] for r in output['results']] == ['outside', 'in-cluster']
    # Facts already collected through a GKU are not fetched a second time
    storage.get_akus_by_ids.assert_awaited_once_with(['outside'])
    assert output['results'][1]['found_by'] == 'gku'


@pytest.mark.asyncio
async def test_direct_search_failure_falls_back_to_gku_facts():
    storage = MagicMock()
    storage.find_similar_akus = AsyncMock(side_effect=RuntimeError("db down"))

    candidates = await make_service(storage)._direct_fact_candidates('ws', [1.0, 0.0], 5, set())

    assert candidates == []
