"""
Tests that GKU structural signatures are built from the entity-relation graph of
the GKU's facts (the same kind of graph as query graphs), with a fallback to the
embedding-similarity graph.
"""
from datetime import datetime, timezone

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ekm.core.knowledge import organization
from ekm.core.query_analysis import QueryGraphExtractor

CHAIN_RESPONSE = """
{"nodes": ["Ada", "Analytical Engine", "Babbage", "London"],
 "edges": [["Ada", "Analytical Engine", "wrote notes on"],
           ["Analytical Engine", "Babbage", "designed by"],
           ["Babbage", "London", "lived in"]]}
"""


def make_storage(n_akus=4, dim=8):
    rng = np.random.default_rng(0)
    storage = MagicMock()
    storage.get_akus_with_embeddings = AsyncMock(return_value=[
        {'id': f'aku-{i}', 'content': f'fact number {i}', 'embedding': rng.normal(size=dim).tolist(),
         'created_at': datetime(2026, 1, 1 + i, tzinfo=timezone.utc)}
        for i in range(n_akus)
    ])
    storage.save_gku = AsyncMock(return_value='gku-0')
    storage.associate_akus_with_gku = AsyncMock()
    return storage


async def form_one_gku(storage, llm):
    """Run form_gkus_from_akus with clustering stubbed to put every AKU in one cluster."""
    n = len(await storage.get_akus_with_embeddings('ws'))
    with patch.object(organization, 'apply_adaptive_clustering',
                      return_value=(np.zeros(n, dtype=int), {})), \
         patch.object(organization, 'compute_attention_weighted_centroids',
                      return_value=(np.zeros((1, 8)), np.zeros(1))):
        ids = await organization.form_gkus_from_akus(storage, 'ws', min_cluster_size=2, llm=llm)
    assert ids == ['gku-0']
    return storage.save_gku.call_args.kwargs['pattern_signature']


@pytest.mark.asyncio
async def test_extract_text_pattern_uses_label_and_builds_signature():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=CHAIN_RESPONSE)

    signature = await QueryGraphExtractor(llm).extract_text_pattern("- fact", label="FACTS")

    assert "FACTS: - fact" in llm.generate.call_args.kwargs['user_message']
    # A 4-node chain: open wedges, no triangles
    assert signature['graphlets']['triangle'] == 0
    assert signature['graphlets']['wedge'] == 2
    assert signature['graph_properties'] == {'num_nodes': 4, 'num_edges': 3}


@pytest.mark.asyncio
async def test_extract_text_pattern_returns_none_without_graph():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="no JSON here")

    assert await QueryGraphExtractor(llm).extract_text_pattern("text") is None


@pytest.mark.asyncio
async def test_gku_signature_comes_from_fact_relations():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=CHAIN_RESPONSE)
    storage = make_storage()

    signature = await form_one_gku(storage, llm)

    assert signature['structure_source'] == 'entity_relations'
    assert signature['graphlets']['wedge'] == 2          # from the relation graph
    assert signature['cluster_size'] == 4                # embedding stats are kept
    sent = llm.generate.call_args.kwargs['user_message']
    assert all(f'- fact number {i}' in sent for i in range(4))


@pytest.mark.asyncio
async def test_gku_signature_falls_back_when_extraction_fails():
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="no JSON here")

    signature = await form_one_gku(make_storage(), llm)

    assert signature['structure_source'] == 'embedding_similarity'
    assert 'graphlets' in signature


@pytest.mark.asyncio
async def test_gku_signature_without_llm_uses_embedding_graph():
    signature = await form_one_gku(make_storage(), llm=None)

    assert signature['structure_source'] == 'embedding_similarity'


@pytest.mark.asyncio
async def test_gku_records_its_facts_time():
    storage = make_storage()

    await form_one_gku(storage, llm=None)

    fact_time = storage.save_gku.call_args.kwargs['cluster_metadata']['fact_time']
    jan = lambda d: datetime(2026, 1, d, tzinfo=timezone.utc).timestamp()
    assert fact_time['oldest'] == jan(1)
    assert fact_time['newest'] == jan(4)
    assert fact_time['median'] == (jan(2) + jan(3)) / 2
    assert fact_time['source'] == 'ingested_at'


def test_fact_time_prefers_event_time_and_parses_iso_strings():
    akus = [
        {'created_at': datetime(2026, 5, 1, tzinfo=timezone.utc),
         'aku_metadata': {'event_time': '1969-07-20T20:17:00Z'}},
        {'created_at': '2026-05-02T00:00:00+00:00', 'aku_metadata': {}},
        {'created_at': None},
    ]

    summary = organization.fact_time_summary(akus)

    assert summary['oldest'] == datetime(1969, 7, 20, 20, 17, tzinfo=timezone.utc).timestamp()
    assert summary['newest'] == datetime(2026, 5, 2, tzinfo=timezone.utc).timestamp()
    assert summary['source'] == 'event_time+ingested_at'
    assert organization.fact_time_summary([{'created_at': None}]) is None


def test_temporal_head_ranks_by_fact_time_not_consolidation_time():
    from ekm.core.attention import TemporalRetrievalHead
    from ekm.core.attention.mechanisms import _gku_reference_time

    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    consolidated = datetime(2026, 5, 31, tzinfo=timezone.utc)   # both GKUs rebuilt yesterday
    old_facts = {'created_at': consolidated,
                 'cluster_metadata': {'fact_time': {'median': datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()}}}
    new_facts = {'created_at': consolidated,
                 'cluster_metadata': {'fact_time': {'median': datetime(2026, 5, 20, tzinfo=timezone.utc).timestamp()}}}
    legacy = {'created_at': consolidated, 'cluster_metadata': {}}   # GKU saved before this change

    times = [{'created_at': _gku_reference_time(g)} for g in (old_facts, new_facts, legacy)]
    scores = TemporalRetrievalHead().compute_attention({'timestamp': now}, times)

    assert scores[1] > scores[0]            # newer knowledge scores higher
    assert _gku_reference_time(legacy) == consolidated


def test_facts_text_respects_character_budget():
    text = organization._facts_text(["a" * 40, "b" * 40, "c" * 40], max_chars=90)

    assert text.count("\n") == 1        # two facts fit, the third is cut
    assert "c" not in text
