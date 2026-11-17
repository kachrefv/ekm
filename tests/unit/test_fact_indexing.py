"""
Training must store each fact once (linked to its episode), and the fact index
must be per workspace, rebuilt from storage, and report cosine similarity.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from ekm.storage.memory import MemoryStorage
from ekm.core.scalability import ScalableEKMBackend
from ekm.core.training import TrainingService

VECTORS = {
    'chunk one': [1.0, 0.0, 0.0],
    'chunk two': [0.0, 1.0, 0.0],
    'cats purr': [1.0, 0.1, 0.0],
    'cats meow': [1.0, 0.2, 0.0],
    'kittens purr too': [0.95, 0.15, 0.0],
    'stars are hot': [0.0, 0.0, 1.0],
}
FACTS = {'chunk one': ['cats purr', 'cats meow'], 'chunk two': ['kittens purr too', 'stars are hot']}


def make_embeddings():
    embeddings = MagicMock()
    embeddings.embed_query = AsyncMock(side_effect=lambda text: VECTORS[text])
    embeddings.embed_documents = AsyncMock(side_effect=lambda texts: [VECTORS[t] for t in texts])
    return embeddings


def make_training(storage, backend):
    service = TrainingService(storage=storage, llm=MagicMock(), embeddings=make_embeddings(),
                              scalable_backend=backend)
    service.text_processor = MagicMock()
    service.text_processor.chunk_text = AsyncMock(return_value=['chunk one', 'chunk two'])
    service.knowledge_extractor = MagicMock()
    service.knowledge_extractor.single_step_extraction = AsyncMock(
        side_effect=lambda chunk: {'summary': chunk, 'facts': FACTS[chunk]})
    return service


@pytest.mark.asyncio
async def test_training_stores_each_fact_once_with_its_episode():
    storage = MemoryStorage()
    backend = ScalableEKMBackend(storage=storage, embeddings=make_embeddings())

    await make_training(storage, backend).train('ws', 'text')

    contents = sorted(a['content'] for a in storage.akus.values())
    assert contents == sorted(sum(FACTS.values(), []))          # no duplicates
    assert all(a['episode_id'] for a in storage.akus.values())   # none orphaned
    assert len(backend._indexed_ids['ws']) == 4


@pytest.mark.asyncio
async def test_training_links_new_facts_to_earlier_ones_by_cosine():
    storage = MemoryStorage()
    backend = ScalableEKMBackend(storage=storage, embeddings=make_embeddings())

    await make_training(storage, backend).train('ws', 'text')

    by_content = {a['id']: a['content'] for a in storage.akus.values()}
    links = {(by_content[r['source_aku_id']], by_content[r['target_aku_id']]): r
             for r in storage.relationships.values()}
    # Cross-chunk link from the new fact to the similar earlier ones
    assert ('kittens purr too', 'cats purr') in links
    assert links[('kittens purr too', 'cats purr')]['temporal_proximity'] == 0.0
    assert links[('kittens purr too', 'cats purr')]['semantic_similarity'] > 0.99
    # Same-chunk link exists once, from the local loop
    same_chunk = [k for k in links if set(k) == {'cats purr', 'cats meow'}]
    assert len(same_chunk) == 1
    # Unrelated facts are not linked
    assert not any('stars are hot' in k for k in links)


@pytest.mark.asyncio
async def test_index_is_per_workspace_and_survives_restart():
    storage = MemoryStorage()
    ids = await storage.save_akus('ws1', 'ep', [{'content': 'cats purr', 'embedding': VECTORS['cats purr']}])

    # A fresh backend (as after a restart) rebuilds the index from storage
    backend = ScalableEKMBackend(storage=storage, embeddings=make_embeddings())
    found = await backend.search_similar_akus('ws1', [1.0, 0.0, 0.0], k=5)
    assert [aku_id for aku_id, _ in found] == ids

    # Another workspace never sees ws1's facts
    assert await backend.search_similar_akus('ws2', [1.0, 0.0, 0.0], k=5) == []


@pytest.mark.asyncio
async def test_search_returns_cosine_and_index_akus_skips_known_ids():
    storage = MemoryStorage()
    backend = ScalableEKMBackend(storage=storage, embeddings=make_embeddings())

    assert await backend.index_akus('ws', ['same', 'orth'], [[2.0, 0.0], [0.0, 3.0]]) == 2
    assert await backend.index_akus('ws', ['same'], [[2.0, 0.0]]) == 0      # already indexed

    sims = dict(await backend.search_similar_akus('ws', [5.0, 0.0], k=2))
    assert sims['same'] == pytest.approx(1.0, abs=1e-5)
    assert sims['orth'] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.asyncio
async def test_invalidate_rebuilds_without_archived_facts():
    storage = MemoryStorage()
    ids = await storage.save_akus('ws', 'ep', [{'content': 'cats purr', 'embedding': VECTORS['cats purr']}])
    backend = ScalableEKMBackend(storage=storage, embeddings=make_embeddings())
    assert await backend.search_similar_akus('ws', [1.0, 0.0, 0.0], k=5)

    await storage.archive_akus(ids)
    backend.invalidate_workspace_index('ws')

    assert await backend.search_similar_akus('ws', [1.0, 0.0, 0.0], k=5) == []
