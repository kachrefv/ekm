
import time
from ekm.core.scalability import CacheManager

def test_cache_size_tracking():
    """Verify cache size is tracked correctly."""
    # Create cache with small size
    cache = CacheManager(max_size=10, default_ttl=60)
    
    # 1. Add new item
    cache.set("key1", "value1")
    assert cache.size == 1
    assert cache.get("key1") == "value1"
    
    # 2. Update existing item - size should NOT change
    cache.set("key1", "updated_value")
    assert cache.size == 1
    assert cache.get("key1") == "updated_value"
    
    # 3. Add another new item - size should increase
    cache.set("key2", "value2")
    assert cache.size == 2
    assert cache.get("key2") == "value2"
    
    # 4. Remove item
    cache._remove(cache._hash_key("key1"))
    assert cache.size == 1
    
    # 5. Clear cache
    cache.clear()
    assert cache.size == 0
