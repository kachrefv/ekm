"""
Tests for: measured semantic preservation, event-time extraction, relation-label
matching, and the three heads on the default (hierarchical) retrieval path.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from ekm.core.consolidation import SleepConsolidator
from ekm.core.knowledge.extraction import KnowledgeExtractor, normalize_event_time
from ekm.core.patterns import calculate_pattern_similarity
from ekm.core.query_analysis import normalize_relation_label, relation_label_set


# --- Semantic preservation -------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_preservation_is_measured_from_merged_text():
    storage = MagicMock()
    storage.get_akus_by_ids = AsyncMock(return_value=[
        {'id': 'merged', 'content': 'merged text', 'embedding': [0.7, 0.7]},
        {'id': 'kept', 'content': 'a', 'embedding': [1.0, 0.0]},      # covered by the summary
        {'id': 'lost', 'content': 'b', 'embedding': [0.0, 1.0]},      # dropped by the summary
    ])
    embeddings = MagicMock()
    embeddings.embed_query = AsyncMock(return_value=[1.0, 0.0])      # the merged text's meaning
    consolidator = SleepConsolidator(storage, MagicMock(), embeddings)

    score = await consolidator._measure_semantic_preservation([('merged', ['kept', 'lost'])])

    assert score == pytest.approx(0.5)                # mean of 1.0 and 0.0
    embeddings.embed_query.assert_awaited_once_with('merged text')


@pytest.mark.asyncio
async def test_semantic_preservation_is_one_when_nothing_merged():
    consolidator = SleepConsolidator(MagicMock(), MagicMock(), MagicMock())
    assert await consolidator._measure_semantic_preservation([]) == 1.0


# --- Event times ----------------------------------------------------------------

def test_normalize_event_time():
    assert normalize_event_time('1969-07-20') == '1969-07-20'
    assert normalize_event_time('1969-07') == '1969-07-01'
    assert normalize_event_time('1969') == '1969-01-01'
    assert normalize_event_time('2026-05-01T10:00:00Z').startswith('2026-05-01T10:00:00')
    for bad in (None, '', 'null', 'last spring', '2026-13-01', 1969):
        assert normalize_event_time(bad) is None


def test_extraction_parses_dated_and_plain_facts():
    extractor = KnowledgeExtractor(llm=MagicMock(), embeddings=MagicMock())
    response = '''{"summary": "Apollo", "facts": [
        {"fact": "Apollo 11 landed on the Moon", "when": "1969-07-20"},
        {"fact": "The Moon has no atmosphere", "when": null},
        "Plain string facts still work"]}'''

    parsed = extractor._parse_extraction_response(response)

    assert parsed['facts'] == ['Apollo 11 landed on the Moon', 'The Moon has no atmosphere',
                               'Plain string facts still work']
    assert parsed['fact_times'] == ['1969-07-20', None, None]


@pytest.mark.asyncio
async def test_training_stores_event_time_on_dated_facts():
    from ekm.storage.memory import MemoryStorage
    from ekm.core.scalability import ScalableEKMBackend
    from ekm.core.training import TrainingService

    storage = MemoryStorage()
    embeddings = MagicMock()
    embeddings.embed_query = AsyncMock(return_value=[1.0, 0.0])
    embeddings.embed_documents = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])
    service = TrainingService(storage=storage, llm=MagicMock(), embeddings=embeddings,
                              scalable_backend=ScalableEKMBackend(storage, embeddings))
    service.text_processor = MagicMock()
    service.text_processor.chunk_text = AsyncMock(return_value=['chunk'])
    service.knowledge_extractor = MagicMock()
    service.knowledge_extractor.single_step_extraction = AsyncMock(return_value={
        'summary': 's', 'facts': ['dated fact', 'undated fact'], 'fact_times': ['1969-07-20', None]})

    await service.train('ws', 'text')

    metadata = {a['content']: a['aku_metadata'] for a in storage.akus.values()}
    assert metadata['dated fact'] == {'event_time': '1969-07-20'}
    assert metadata['undated fact'] == {}


# --- Relation labels ------------------------------------------------------------

def test_relation_labels_are_normalized():
    assert normalize_relation_label('Was Born In') == 'born'
    assert normalize_relation_label('designed by') == 'designed'
    assert normalize_relation_label('is a') == ''
    assert relation_label_set([['A', 'B', 'born in'], ['C', 'D', 'was born in'], ['E', 'F']]) == ['born']


def test_relation_overlap_drives_structural_similarity():
    query = {'relation_labels': ['born']}
    covers_query = {'relation_labels': ['born', 'lived', 'married', 'worked']}   # larger cluster, still full overlap
    unrelated = {'relation_labels': ['orbits', 'contains']}

    assert calculate_pattern_similarity(query, covers_query) == pytest.approx(1.0)
    assert calculate_pattern_similarity(query, unrelated) == pytest.approx(0.0)


# --- Heads on the default path -------------------------------------------------

def test_head_count_adapts_to_embedding_size():
    from ekm.core.attention.mechanisms import _compatible_heads

    assert _compatible_heads(768, 8) == 8
    assert _compatible_heads(1000, 8) == 8
    assert _compatible_heads(1001, 8) == 7
    assert _compatible_heads(2, 8) == 2
    assert _compatible_heads(3, 8) == 3
    assert _compatible_heads(7, 4) == 1


@pytest.mark.asyncio
async def test_default_path_uses_structure_to_rank_clusters():
    from ekm.core.attention import QKVAttentionRetriever

    def gku(gku_id, labels):
        return {'id': gku_id, 'centroid_embedding': [1.0, 0.0], 'cluster_metadata': {},
                'pattern_signature': {'relation_labels': labels}, 'created_at': 1000.0}

    storage = MagicMock()
    # Same centroid, so cosine alone cannot tell them apart
    storage.get_gkus_by_workspace = AsyncMock(return_value=[gku('other', ['orbits']), gku('match', ['born'])])
    storage.get_akus_in_gku = AsyncMock(side_effect=lambda gku_id: [
        {'id': f'{gku_id}-fact', 'content': 'x', 'embedding': [1.0, 0.0]}])

    retriever = QKVAttentionRetriever(embedding_dim=2, use_hierarchical_attention=True)
    result = await retriever.retrieve(storage, 'ws', [1.0, 0.0], query_pattern={'relation_labels': ['born']},
                                      query_time_context={'timestamp': 1000.0}, top_k=2)

    scores = result.interpretations['gku_scores']
    assert scores['match'] > scores['other']
    assert result.interpretations['selected_gku_ids'][0] == 'match'
