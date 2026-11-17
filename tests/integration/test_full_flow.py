"""
End-to-end run against the real OpenRouter API (Claude Haiku 5.5 + OpenRouter
embeddings). Skipped unless OPENROUTER_API_KEY is set; it makes paid API calls.
"""
import os

import pytest

from ekm.core.consolidation import SleepConsolidator
from ekm.core.mesh import EKM
from ekm.providers.openrouter import OpenRouterProvider
from ekm.storage.factory import create_storage


@pytest.mark.asyncio
async def test_ekm_full_flow():
    # 1. Setup Providers and Storage
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    provider = OpenRouterProvider()
    storage = create_storage("memory")

    # 2. Initialize EKM
    ekm = EKM(storage=storage, llm=provider, embeddings=provider)
    workspace_id = "test-integration-flow"

    knowledge_base = [
        "Quantum Neural Fabrics (QNF) are a hypothetical architecture for distributed artificial intelligence, where processing nodes are connected in a mesh-like structure.",
        "QNF uses spreading activation to simulate biological neural pathways, allowing for dynamic memory retrieval and self-organizing knowledge.",
        "The primary advantage of QNF is its ability to handle massive topological datasets by localized processing without global synchronization.",
        "Episodic Knowledge Mesh (EKM) is a concrete implementation of QNF principles, focusing on episodic and semantic memory layers."
    ]

    for i, text in enumerate(knowledge_base):
        await ekm.train(workspace_id, text, title=f"QNF Concept {i+1}")

    # 3. Initial Retrieval
    query = "What is the primary advantage of Quantum Neural Fabrics?"
    results = await ekm.retrieve(workspace_id, query)

    assert results['metadata']['total_results'] > 0
    assert results['results'][0]['score'] > 0.5  # Expect reasonable relevance

    # 4. Consolidation
    consolidator = SleepConsolidator(
        storage=storage,
        llm=provider,
        embeddings=provider,
        min_cluster_support=2
    )
    await consolidator.run_consolidation(workspace_id)
    ekm.scalable_backend.invalidate_workspace_index(workspace_id)

    # 5. Final Retrieval
    final_results = await ekm.retrieve(workspace_id, query)
    assert final_results['metadata']['total_results'] > 0
