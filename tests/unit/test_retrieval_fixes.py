"""
Regression tests for the retrieval review:
- episodes never leak across workspaces, and similarity is cosine everywhere
- GKU membership is read through the ORM (no REPLACE() join) and skips archived facts
- every result is scored on one scale; a GKU can only add to a fact's score
- the focus buffer boosts the GKUs that hold focused facts
- one source episode cannot fill the result list
- the query is embedded once per retrieval
- debug_mode returns debug info
- a small IVF batch gets an exact index instead of degenerate clusters
"""
import uuid

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ekm.core.retrieval.service import RetrievalService, GKU_CONTEXT_BOOST, MAX_RESULTS_PER_SOURCE
from ekm.core.state import FocusBuffer
from ekm.storage.memory import MemoryStorage


def unit(*v):
    a = np.asarray(v, dtype=float)
    return (a / np.linalg.norm(a)).tolist()


def cos(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- SQL storage -----------------------------------------------------------------

@pytest.fixture
async def sql_storage():
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from ekm.core.models import Base
    from ekm.storage.sql import SQLStorage

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session = async_sessionmaker(engine, expire_on_commit=False)()
    try:
        yield SQLStorage(session)
    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_episodes_stay_in_their_workspace(sql_storage):
    ws_a, ws_b = str(uuid.uuid4()), str(uuid.uuid4())
    await sql_storage.save_episode(ws_a, "A's secret", "a", unit(1, 0, 0), {})
    # Workspace A is searched first, which used to fill the shared index with A only
    assert [e['content'] for e in await sql_storage.find_similar_episodes(ws_a, unit(1, 0, 0), 0.0, 5)] == ["A's secret"]

    await sql_storage.save_episode(ws_b, "B's note", "b", unit(1, 0.1, 0), {})
    found = await sql_storage.find_similar_episodes(ws_b, unit(1, 0, 0), 0.0, 5)

    assert [e['content'] for e in found] == ["B's note"]


@pytest.mark.asyncio
async def test_episode_similarity_is_cosine_and_threshold_applies(sql_storage):
    ws = str(uuid.uuid4())
    await sql_storage.save_episode(ws, "close", "", [3.0, 4.0], {})
    await sql_storage.save_episode(ws, "orthogonal", "", [0.0, 1.0], {})

    found = await sql_storage.find_similar_episodes(ws, [1.0, 0.0], threshold=0.3, limit=5)

    assert [e['content'] for e in found] == ["close"]
    assert found[0]['similarity'] == pytest.approx(0.6, abs=1e-6)
    assert found[0]['created_at'] is not None


@pytest.mark.asyncio
async def test_gku_membership_skips_archived_facts(sql_storage):
    ws = str(uuid.uuid4())
    active, archived = await sql_storage.save_akus(ws, None, [
        {'content': 'active', 'embedding': [1.0, 0.0]},
        {'content': 'archived', 'embedding': [0.0, 1.0]},
    ])
    gku_id = await sql_storage.save_gku(ws, "cluster", centroid_embedding=[1.0, 0.0])
    await sql_storage.associate_akus_with_gku(gku_id, [active, archived])
    await sql_storage.associate_akus_with_gku(gku_id, [active])          # no duplicate row
    await sql_storage.archive_akus([archived])

    members = await sql_storage.get_akus_in_gku(gku_id)

    assert [m['content'] for m in members] == ['active']
    assert members[0]['embedding'] == pytest.approx([1.0, 0.0])
    assert await sql_storage.get_gku_ids_for_akus(ws, [active, archived]) == {
        active: [gku_id], archived: [gku_id]}


@pytest.mark.asyncio
async def test_gku_membership_reads_rows_written_by_old_code(sql_storage):
    """The old raw-SQL insert stored dashed UUID strings; they must still be found."""
    from sqlalchemy import text
    ws = str(uuid.uuid4())
    [aku_id] = await sql_storage.save_akus(ws, None, [{'content': 'legacy', 'embedding': [1.0, 0.0]}])
    gku_id = await sql_storage.save_gku(ws, "cluster", centroid_embedding=[1.0, 0.0])
    await sql_storage.db.execute(text("INSERT INTO ekm_gku_akus (gku_id, aku_id) VALUES (:g, :a)"),
                                 {'g': gku_id, 'a': aku_id})
    await sql_storage.db.commit()

    assert [m['content'] for m in await sql_storage.get_akus_in_gku(gku_id)] == ['legacy']
    assert await sql_storage.get_gku_ids_for_akus(ws, [aku_id]) == {aku_id: [gku_id]}


@pytest.mark.asyncio
async def test_fact_similarity_is_cosine(sql_storage):
    ws = str(uuid.uuid4())
    await sql_storage.save_akus(ws, None, [{'content': 'f', 'embedding': [3.0, 4.0]}])

    [match] = await sql_storage.find_similar_akus(ws, [1.0, 0.0], threshold=0.0, limit=5)

    assert match['similarity'] == pytest.approx(0.6, abs=1e-6)


# --- Scoring in the service -------------------------------------------------------

def make_service(storage, rl=None):
    llm = MagicMock()
    llm.generate = AsyncMock(return_value="no graph")
    embeddings = MagicMock()
    embeddings.embed_query = AsyncMock(return_value=[1.0, 0.0])
    rl = rl or MagicMock()
    return RetrievalService(storage=storage, llm=llm, embeddings=embeddings,
                            scalable_backend=MagicMock(), rl_feedback_system=rl)


async def memory_workspace():
    """Workspace 'ws': one GKU with two facts, a fact outside it, and two episodes."""
    storage = MemoryStorage()
    ep_strong = await storage.save_episode('ws', 'strong episode', '', unit(1, 0.2), {})
    await storage.save_episode('ws', 'weak episode', '', unit(1, 1.5), {})
    in_gku = await storage.save_akus('ws', ep_strong, [
        {'content': 'in-gku close', 'embedding': unit(1, 0.4)},
        {'content': 'in-gku far', 'embedding': unit(0.2, 1)},
    ])
    await storage.save_akus('ws', None, [{'content': 'loose fact', 'embedding': unit(1, 0.9)}])
    gku_id = await storage.save_gku('ws', 'Cluster', centroid_embedding=unit(1, 0.5))
    await storage.associate_akus_with_gku(gku_id, in_gku)
    return storage, gku_id, in_gku


@pytest.mark.asyncio
async def test_all_layers_share_the_cosine_scale():
    storage, gku_id, _ = await memory_workspace()
    service = make_service(storage)

    out = await service.retrieve('ws', 'q', top_k=10, use_rl_feedback=False)

    for r in out['results']:
        if r['item_type'] == 'episode':
            assert r['score'] == pytest.approx(r['semantic_similarity'])
        else:
            assert r['score'] == pytest.approx(r['semantic_similarity'] + r['gku_bonus'])
            assert 0.0 <= r['gku_bonus'] <= GKU_CONTEXT_BOOST
    scores = [r['score'] for r in out['results']]
    assert scores == sorted(scores, reverse=True)
    # The strong episode (cos ~0.98) beats the loose fact (cos ~0.74): they are comparable now
    order = [r['content'] for r in out['results']]
    assert order.index('strong episode') < order.index('loose fact')


@pytest.mark.asyncio
async def test_gku_membership_never_lowers_a_fact():
    storage, gku_id, _ = await memory_workspace()
    service = make_service(storage)

    out = await service.retrieve('ws', 'q', mode='facts', top_k=10, use_rl_feedback=False)

    by_content = {r['content']: r for r in out['results']}
    close = by_content['in-gku close']
    assert close['found_by'] == 'gku'
    assert close['score'] >= cos([1, 0], unit(1, 0.4))
    assert out['metadata']['layer_distribution']['gku'] == 1


@pytest.mark.asyncio
async def test_query_is_embedded_once():
    storage, _, _ = await memory_workspace()
    rl = MagicMock()
    rl.has_learned = False
    rl.get_ranked_responses = AsyncMock(side_effect=lambda query, candidate_responses, **kw:
                                        MagicMock(ranked_responses=candidate_responses,
                                                  confidence_scores=[], exploration_factor=0.0))
    rl.get_model_performance_metrics = MagicMock(return_value={})
    service = make_service(storage, rl=rl)

    await service.retrieve('ws', 'q', top_k=5)

    service.embeddings.embed_query.assert_awaited_once()
    kwargs = rl.get_ranked_responses.await_args.kwargs
    assert kwargs['query_embedding'] == [1.0, 0.0]
    assert kwargs['candidate_embeddings']          # fact embeddings handed over


@pytest.mark.asyncio
async def test_focus_buffer_boosts_its_gku():
    storage = MemoryStorage()
    a_facts = await storage.save_akus('ws', None, [{'content': 'a', 'embedding': unit(1, 0.30)}])
    b_facts = await storage.save_akus('ws', None, [{'content': 'b', 'embedding': unit(1, 0.31)}])
    gku_a = await storage.save_gku('ws', 'A', centroid_embedding=unit(1, 0.30))
    gku_b = await storage.save_gku('ws', 'B', centroid_embedding=unit(1, 0.31))
    await storage.associate_akus_with_gku(gku_a, a_facts)
    await storage.associate_akus_with_gku(gku_b, b_facts)
    service = make_service(storage)

    plain = await service._retrieve_with_qkv_attention('ws', 'q', top_k=1, query_pattern={})
    focus = FocusBuffer()
    focus.update(b_facts)
    focused = await service._retrieve_with_qkv_attention('ws', 'q', top_k=1, query_pattern={},
                                                         focus_buffer=focus)

    assert plain['metadata']['selected_gku_ids'] == [gku_a]
    assert focused['metadata']['selected_gku_ids'] == [gku_b]


def test_one_source_cannot_fill_the_list():
    ranked = ([{'id': f's{i}', 'source_episode_id': 'ep1'} for i in range(4)] +
              [{'id': 'other', 'source_episode_id': 'ep2'}, {'id': 'free', 'source_episode_id': None}])

    picked = RetrievalService._diversify(ranked, 4)

    assert [r['id'] for r in picked] == ['s0', 's1', 'other', 'free'][:4]
    assert sum(r['source_episode_id'] == 'ep1' for r in picked) == MAX_RESULTS_PER_SOURCE
    # With nothing else left, deferred items still fill the list
    assert len(RetrievalService._diversify(ranked[:4], 4)) == 4


@pytest.mark.asyncio
async def test_debug_mode_returns_debug_info():
    storage, _, _ = await memory_workspace()
    service = make_service(storage)

    with patch('ekm.core.explainability.generate_retrieval_explanation', return_value={'ok': True}):
        out = await service.retrieve('ws', 'q', top_k=3, use_rl_feedback=False, debug_mode=True)

    assert service.debugger is not None
    assert out['debug_info'].gku_names == ['Cluster']
    assert out['metadata']['explanation'] == {'ok': True}


@pytest.mark.asyncio
async def test_failed_layer_does_not_sink_the_other():
    storage, _, _ = await memory_workspace()
    service = make_service(storage)
    service._retrieve_with_qkv_attention = AsyncMock(side_effect=RuntimeError("boom"))

    out = await service.retrieve('ws', 'q', top_k=3, use_rl_feedback=False)

    assert out['results'] and all(r['layer'] == 'episodic' for r in out['results'])
    assert out['metadata']['layer_errors'] == {'facts': 'boom'}


# --- Vector index -----------------------------------------------------------------

def test_small_ivf_batch_uses_exact_index():
    faiss = pytest.importorskip("faiss")
    from ekm.core.scalability import VectorIndexManager

    index = VectorIndexManager(dimension=4, index_type="IVF")
    vectors = np.eye(4, dtype=np.float32)
    index.add_vectors(vectors, ['a', 'b', 'c', 'd'])

    assert isinstance(index.index, faiss.IndexFlatL2)
    _, ids = index.search(vectors[2], 1)
    assert ids == ['c']
