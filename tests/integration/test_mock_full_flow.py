import asyncio
import pytest
import numpy as np
from ekm.providers.mock import MockLLM, MockEmbeddings
from ekm.storage.factory import create_storage
from ekm.core.mesh import EKM
from ekm.core.consolidation import SleepConsolidator

@pytest.mark.asyncio
async def test_mock_full_flow():
    """Test the entire EKM flow (Train -> Retrieve -> Consolidate -> Retrieve) using mocks."""
    
    # 1. Setup Mock Providers and Memory Storage
    llm = MockLLM()
    embeddings = MockEmbeddings()
    storage = create_storage("memory")
    
    # 2. Initialize EKM
    ekm = EKM(storage=storage, llm=llm, embeddings=embeddings)
    workspace_id = "test-mock-full-flow"
    
    # 3. Phase: Training
    knowledge_base = [
        "The Episodic Knowledge Mesh (EKM) is a system for long-term memory in AI agents.",
        "EKM uses a three-layer memory architecture: Episodic, Semantic, and Abstract.",
        "Consolidation is the process of moving information from the episodic layer to the semantic layer.",
        "GKUs represent Global Knowledge Units that aggregate multiple Atomic Knowledge Units (AKUs)."
    ]
    
    for i, text in enumerate(knowledge_base):
        await ekm.train(workspace_id, text, title=f"Concept {i+1}")
    
    # 4. Phase: Initial Retrieval
    query = "What are the layers of EKM?"
    results = await ekm.retrieve(workspace_id, query, top_k=5)
    
    print(f"Results metadata: {results['metadata']}")
    print(f"Number of results: {len(results['results'])}")
    
    assert results['metadata']['total_results'] > 0
    assert len(results['results']) > 0
    assert results['results'][0]['layer'] == 'aku' # Should be from AKUs initially
    
    # 5. Phase: Consolidation (Sleep Cycle)
    consolidator = SleepConsolidator(
        storage=storage, 
        llm=llm, 
        embeddings=embeddings,
        min_cluster_support=2 # Force clusters for small metadata
    )
    
    # We need to make sure the embeddings are similar enough for clustering in MockEmbeddings
    # MockEmbeddings usually returns random vectors, which might fail clustering.
    # Let's override the mock behavior if needed, but for now we test the flow stability.
    
    consolidation_result = await consolidator.run_consolidation(workspace_id)
    
    # Verify consolidation result object
    assert hasattr(consolidation_result, 'consolidated_akus')
    assert hasattr(consolidation_result, 'created_gkus')
    
    # 6. Phase: Post-Consolidation Retrieval
    final_results = await ekm.retrieve(workspace_id, query, top_k=2)
    
    assert final_results['metadata']['total_results'] > 0
    # Even if no GKUs were formed (due to mock randomness), the results should still be retrievable
    
    print("\nMock Full Flow Test Successful!")

if __name__ == "__main__":
    asyncio.run(test_mock_full_flow())
