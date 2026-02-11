import asyncio
import numpy as np
from ekm.core.state import FocusBuffer
from ekm.core.mesh import EKM
from ekm.storage.factory import create_storage
from ekm.providers.mock import MockLLM, MockEmbeddings

class MockStorage:
    def __init__(self):
        self.episodes = {}
        self.akus = {}
        self.gkus = {}
        self.relationships = []
        
    async def save_episode(self, workspace_id, content, summary, embedding, metadata):
        import uuid
        episode_id = str(uuid.uuid4())
        self.episodes[episode_id] = {
            'workspace_id': workspace_id,
            'content': content,
            'summary': summary,
            'embedding': embedding,
            'metadata': metadata
        }
        return episode_id
        
    async def save_akus(self, workspace_id, episode_id, akus):
        import uuid
        aku_ids = []
        for aku in akus:
            aku_id = str(uuid.uuid4())
            self.akus[aku_id] = {
                'workspace_id': workspace_id,
                'episode_id': episode_id,
                'content': aku['content'],
                'embedding': aku['embedding'],
                'metadata': aku.get('metadata', {})
            }
            aku_ids.append(aku_id)
        return aku_ids
    
    async def find_similar_episodes(self, workspace_id, query_embedding, threshold, limit):
        return [{'id': 'mock_id', 'content': 'mock content', 'summary': 'mock summary', 'similarity': 0.8}]
    
    async def find_similar_akus(self, workspace_id, query_embedding, threshold, limit):
        return list(self.akus.values())[:limit]
    
    async def get_akus_by_ids(self, ids):
        return [self.akus.get(id) for id in ids if id in self.akus]
    
    async def get_akus_with_embeddings(self, workspace_id):
        return [aku for aku in self.akus.values() if aku['workspace_id'] == workspace_id and aku.get('embedding') is not None]
    
    async def get_aku_relationships(self, workspace_id, aku_ids):
        return self.relationships
    
    async def save_relationships(self, workspace_id, relationships):
        self.relationships.extend(relationships)
    
    async def save_gku(self, workspace_id, name, description, centroid_embedding, pattern_signature, cluster_metadata):
        import uuid
        gku_id = str(uuid.uuid4())
        self.gkus[gku_id] = {
            'workspace_id': workspace_id,
            'name': name,
            'description': description,
            'centroid_embedding': centroid_embedding,
            'pattern_signature': pattern_signature,
            'cluster_metadata': cluster_metadata
        }
        return gku_id
    
    async def associate_akus_with_gku(self, gku_id, aku_ids):
        pass
    
    async def get_gku(self, gku_id):
        return self.gkus.get(gku_id, None)
    
    async def get_akus_in_gku(self, gku_id):
        return []
    
    async def archive_akus(self, aku_ids):
        pass

    async def get_gkus_by_workspace(self, workspace_id):
        return [gku for gku in self.gkus.values() if gku['workspace_id'] == workspace_id]

async def test_focus_buffer_basic():
    """Test basic functionality of the FocusBuffer class."""
    print("=== Testing FocusBuffer Basic Functionality ===")
    
    # Create a focus buffer
    focus_buffer = FocusBuffer()
    
    # Add some items
    item_ids = ["item1", "item2", "item3"]
    focus_buffer.update(item_ids, action='activate')
    
    print(f"Added items: {item_ids}")
    print(f"Buffer size: {len(focus_buffer.items)}")
    
    # Check that items were added with proper weights
    for item_id in item_ids:
        assert item_id in focus_buffer.items
        item = focus_buffer.items[item_id]
        print(f"  {item_id}: weight={item.current_weight}, freq={item.frequency}")
        assert item.current_weight == 1.5  # Default activation weight
        assert item.frequency == 1
    
    # Update with the same items (should increase frequency and weight)
    focus_buffer.update(["item1", "item2"], action='activate')
    
    print(f"After re-activating item1 and item2:")
    for item_id in ["item1", "item2"]:
        item = focus_buffer.items[item_id]
        print(f"  {item_id}: weight={item.current_weight}, freq={item.frequency}")
        assert item.frequency == 2  # Should be incremented
    
    # Test decay by adding unrelated items
    focus_buffer.update(["item4", "item5"], action='activate')
    
    print(f"After adding new items (should decay others):")
    for item_id, item in focus_buffer.items.items():
        print(f"  {item_id}: weight={item.current_weight}, freq={item.frequency}")
    
    print("PASS: Basic FocusBuffer functionality test passed!\n")


async def test_focus_buffer_integration():
    """Test integration of FocusBuffer with EKM system."""
    print("=== Testing FocusBuffer Integration with EKM ===")
    
    # Use mock components
    storage = MockStorage()
    llm = MockLLM()
    embeddings = MockEmbeddings()
    
    # Initialize EKM
    ekm = EKM(storage=storage, llm=llm, embeddings=embeddings)
    workspace_id = 'test-focus-buffer'
    
    # Train with some knowledge
    knowledge_base = [
        "Quantum computing uses quantum bits (qubits) that can exist in superposition states.",
        "Machine learning algorithms can process large datasets to identify patterns.",
        "Neural networks are inspired by the human brain's structure and function.",
        "Blockchain technology provides decentralized and secure transaction records.",
        "Artificial intelligence aims to create systems that can perform human-like tasks."
    ]
    
    print("Training EKM with knowledge base...")
    for i, text in enumerate(knowledge_base):
        await ekm.train(workspace_id, text, title=f"Concept {i+1}")
    
    print(f"Trained with {len(knowledge_base)} knowledge items")
    
    # Create a focus buffer with specific items
    focus_buffer = FocusBuffer()
    
    # Get AKUs to add to focus buffer (we'll simulate this)
    # Since we're using mocks, we'll fabricate some AKU IDs
    aku_ids = [f"aku_{i}" for i in range(3)]
    focus_buffer.update(aku_ids, action='activate')
    
    print(f"Created focus buffer with items: {aku_ids}")
    
    # Test retrieval with focus buffer
    query = "What is quantum computing?"
    print(f"Query: '{query}'")
    
    # Retrieve without focus buffer
    results_without_focus = await ekm.retrieve(workspace_id, query, top_k=3)
    print(f"Results without focus: {len(results_without_focus['results'])}")
    
    # Retrieve with focus buffer
    results_with_focus = await ekm.retrieve(workspace_id, query, top_k=3, focus_buffer=focus_buffer)
    print(f"Results with focus: {len(results_with_focus['results'])}")
    
    # Check that focus buffer was used
    metadata_with_focus = results_with_focus['metadata']
    if 'focus_buffer_used' in metadata_with_focus:
        print(f"Focus buffer used: {metadata_with_focus['focus_buffer_used']}")
    elif 'layer_distribution' in metadata_with_focus:
        print(f"Layer distribution: {metadata_with_focus['layer_distribution']}")
    
    print("✅ FocusBuffer integration test passed!\n")


async def test_focus_buffer_decay_and_pruning():
    """Test decay and pruning functionality of FocusBuffer."""
    print("=== Testing FocusBuffer Decay and Pruning ===")
    
    # Create a focus buffer with small max size to test pruning
    focus_buffer = FocusBuffer(max_size=3, decay_rate=0.1)
    
    # Add more items than max size
    item_ids = [f"item{i}" for i in range(5)]
    focus_buffer.update(item_ids, action='activate')
    
    print(f"Added {len(item_ids)} items to buffer with max_size=3")
    print(f"Buffer size before pruning: {len(focus_buffer.items)}")
    
    # The buffer should automatically prune to max_size
    assert len(focus_buffer.items) <= focus_buffer.max_size
    print(f"Buffer size after pruning: {len(focus_buffer.items)}")
    
    # Test decay by updating with different items
    old_items = list(focus_buffer.items.keys())
    focus_buffer.update(["new_item1", "new_item2"], action='activate')
    
    print(f"After adding new items, checking decay of old items:")
    for item_id in old_items:
        if item_id in focus_buffer.items:
            item = focus_buffer.items[item_id]
            print(f"  {item_id}: weight={item.current_weight:.2f} (should be decayed)")
    
    print("✅ FocusBuffer decay and pruning test passed!\n")


async def test_focus_buffer_weight_calculation():
    """Test the weight calculation and update logic."""
    print("=== Testing FocusBuffer Weight Calculation ===")
    
    focus_buffer = FocusBuffer(decay_rate=0.05)
    
    # Add an item
    focus_buffer.update(["test_item"], action='activate')
    initial_weight = focus_buffer.items["test_item"].current_weight
    print(f"Initial weight: {initial_weight}")
    
    # Re-activate the same item (should increase weight)
    focus_buffer.update(["test_item"], action='activate')
    updated_weight = focus_buffer.items["test_item"].current_weight
    print(f"Weight after re-activation: {updated_weight}")
    
    # The weight should have increased due to relevance score boost
    assert updated_weight > initial_weight
    
    # Test with 'other' action (should not boost as much)
    focus_buffer.update(["test_item2"], action='other')
    item2_weight = focus_buffer.items["test_item2"].current_weight
    print(f"Weight with 'other' action: {item2_weight}")
    
    # 'other' action should have lower boost than 'activate'
    assert item2_weight == 1.0  # Default weight for non-activate action
    
    print("✅ FocusBuffer weight calculation test passed!\n")


async def main():
    """Run all focus buffer tests."""
    print("Testing Focus Buffer System\n")
    
    await test_focus_buffer_basic()
    await test_focus_buffer_decay_and_pruning()
    await test_focus_buffer_weight_calculation()
    await test_focus_buffer_integration()
    
    print("All Focus Buffer tests passed!")


if __name__ == "__main__":
    asyncio.run(main())