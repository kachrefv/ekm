import pytest
import os
from ekm.providers.gemini import GeminiProvider
from ekm.providers.deepseek import DeepSeekProvider
from ekm.storage.factory import create_storage
from ekm.core.mesh import EKM
from ekm.core.consolidation import SleepConsolidator

@pytest.mark.asyncio
async def test_ekm_full_flow():
    # 1. Setup Providers and Storage
    # Using the verified API keys
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        pytest.skip("GEMINI_API_KEY not set")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not gemini_key or not deepseek_key:
        pytest.skip("API keys not available")

    gemini = GeminiProvider(api_key=gemini_key)
    deepseek = DeepSeekProvider(api_key=deepseek_key)
    storage = create_storage("memory")
    
    # 2. Initialize EKM
    ekm = EKM(storage=storage, llm=deepseek, embeddings=gemini)
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
        llm=deepseek, 
        embeddings=gemini,
        min_cluster_support=2
    )
    
    consolidation_result = await consolidator.run_consolidation(workspace_id)
    
    assert len(consolidation_result.consolidated_akus) > 0
    # GKUs might or might not be created depending on clustering, but we check the object
    
    # 5. Final Retrieval
    final_results = await ekm.retrieve(workspace_id, query)
    assert final_results['metadata']['total_results'] > 0
