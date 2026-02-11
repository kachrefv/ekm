from typing import List, Dict, Any, Optional
from .base import BaseStorage
import uuid
import copy
from datetime import datetime


class MemoryStorage(BaseStorage):
    """In-memory storage implementation for EKM."""
    
    def __init__(self):
        # Initialize in-memory data stores
        self.workspaces: Dict[str, Dict[str, Any]] = {}
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.akus: Dict[str, Dict[str, Any]] = {}
        self.gkus: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, Dict[str, Any]] = {}
        self.rl_states: Dict[str, Dict[str, Any]] = {}  # RL state persistence
        
    def _generate_id(self) -> str:
        """Generate a new UUID string."""
        return str(uuid.uuid4())
    
    def _serialize_embedding(self, embedding):
        """Serialize embedding to ensure consistent format."""
        if embedding is None:
            return None
        if isinstance(embedding, (list, tuple)):
            return [float(x) for x in embedding]
        if hasattr(embedding, 'tolist'):
            return [float(x) for x in embedding.tolist()]
        return embedding
    
    def _deserialize_embedding(self, embedding):
        """Deserialize embedding if needed."""
        return embedding
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    async def get_workspace(self, workspace_id: str) -> Dict[str, Any]:
        """Get workspace by ID. Creates it if it doesn't exist."""
        if workspace_id not in self.workspaces:
            self.workspaces[workspace_id] = {
                'id': workspace_id,
                'name': f"Workspace {workspace_id}",
                'created_at': datetime.now()
            }
        workspace = self.workspaces[workspace_id]
        return {'id': workspace['id'], 'name': workspace['name']}

    async def save_episode(self, workspace_id: str, content: str, summary: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        """Save an episode to memory."""
        episode_id = self._generate_id()
        episode = {
            'id': episode_id,
            'workspace_id': workspace_id,
            'content': content,
            'summary': summary,
            'embedding': self._serialize_embedding(embedding),
            'meta_data': metadata,
            'created_at': datetime.now()
        }
        self.episodes[episode_id] = episode
        return episode_id

    async def save_akus(self, workspace_id: str, episode_id: Optional[str], akus: List[Dict[str, Any]]) -> List[str]:
        """Save multiple AKUs to memory."""
        ids = []
        for aku_data in akus:
            aku_id = self._generate_id()
            aku = {
                'id': aku_id,
                'workspace_id': workspace_id,
                'episode_id': episode_id,
                'content': aku_data['content'],
                'embedding': self._serialize_embedding(aku_data.get('embedding')),
                'aku_metadata': aku_data.get('metadata') or aku_data.get('aku_metadata') or {},
                'is_archived': False,
                'created_at': datetime.now()
            }
            self.akus[aku_id] = aku
            ids.append(aku_id)
        return ids

    async def save_relationships(self, workspace_id: str, relationships: List[Dict[str, Any]]):
        """Save relationships to memory."""
        for rel_data in relationships:
            rel_id = self._generate_id()
            rel = {
                'id': rel_id,
                'workspace_id': workspace_id,
                'source_aku_id': rel_data['source_aku_id'],
                'target_aku_id': rel_data['target_aku_id'],
                'semantic_similarity': rel_data.get('semantic_similarity', 0.0),
                'temporal_proximity': rel_data.get('temporal_proximity', 0.0),
                'causal_weight': rel_data.get('causal_weight', 0.0),
                'edge_attributes': rel_data.get('edge_attributes', {}),
                'created_at': datetime.now()
            }
            self.relationships[rel_id] = rel

    async def find_similar_episodes(self, workspace_id: str, embedding: List[float], threshold: float = 0.5, limit: int = 5) -> List[Dict[str, Any]]:
        """Find episodes similar to the given embedding."""
        query_emb = self._serialize_embedding(embedding)
        results = []
        
        for episode in self.episodes.values():
            if episode['workspace_id'] == workspace_id and episode['embedding']:
                sim = self._cosine_similarity(query_emb, self._serialize_embedding(episode['embedding']))
                if sim >= threshold:
                    results.append({
                        'id': episode['id'],
                        'summary': episode['summary'],
                        'content': episode['content'],
                        'similarity': sim
                    })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    async def find_similar_akus(self, workspace_id: str, embedding: List[float], threshold: float, limit: int) -> List[Dict[str, Any]]:
        """Find AKUs similar to the given embedding."""
        query_emb = self._serialize_embedding(embedding)
        results = []
        
        for aku in self.akus.values():
            if (aku['workspace_id'] == workspace_id and 
                aku['embedding'] and 
                not aku['is_archived']):
                
                sim = self._cosine_similarity(query_emb, self._serialize_embedding(aku['embedding']))
                if sim >= threshold:
                    results.append({
                        'id': aku.get('id'),
                        'content': aku.get('content'),
                        'episode_id': aku.get('episode_id'),
                        'similarity': sim
                    })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    async def get_aku_relationships(self, workspace_id: str, aku_ids: List[str]) -> List[Dict[str, Any]]:
        """Get relationships for the specified AKUs."""
        results = []
        
        for rel in self.relationships.values():
            if (rel['workspace_id'] == workspace_id and 
                (rel['source_aku_id'] in aku_ids or rel['target_aku_id'] in aku_ids)):
                
                results.append({
                    'source_aku_id': rel['source_aku_id'],
                    'target_aku_id': rel['target_aku_id'],
                    'semantic_similarity': rel['semantic_similarity'],
                    'temporal_proximity': rel['temporal_proximity'],
                    'causal_weight': rel['causal_weight']
                })
        
        return results

    async def get_akus_by_ids(self, aku_ids: List[str]) -> List[Dict[str, Any]]:
        """Get AKUs by their IDs."""
        results = []
        
        for aku_id in aku_ids:
            aku = self.akus.get(aku_id)
            if aku:
                results.append({
                    'id': aku.get('id'),
                    'content': aku.get('content'),
                    'embedding': aku.get('embedding'),
                    'aku_metadata': aku.get('aku_metadata', {}),
                    'episode_id': aku.get('episode_id')
                })
        
        return results

    async def archive_akus(self, aku_ids: List[str]):
        """Mark AKUs as archived."""
        for aku_id in aku_ids:
            if aku_id in self.akus:
                self.akus[aku_id]['is_archived'] = True

    async def save_gku(self, workspace_id: str, name: str, description: str = "",
                      centroid_embedding: Optional[List[float]] = None,
                      pattern_signature: Optional[Dict[str, Any]] = None,
                      cluster_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a new Global Knowledge Unit to memory."""
        gku_id = self._generate_id()
        gku = {
            'id': gku_id,
            'workspace_id': workspace_id,
            'name': name,
            'description': description,
            'centroid_embedding': self._serialize_embedding(centroid_embedding) if centroid_embedding else None,
            'pattern_signature': pattern_signature,
            'cluster_metadata': cluster_metadata or {},
            'created_at': datetime.now()
        }
        self.gkus[gku_id] = gku
        return gku_id

    async def get_gku(self, gku_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific GKU by ID."""
        gku = self.gkus.get(gku_id)
        if not gku:
            return None
        
        return {
            'id': gku['id'],
            'workspace_id': gku['workspace_id'],
            'name': gku['name'],
            'description': gku['description'],
            'centroid_embedding': gku['centroid_embedding'],
            'pattern_signature': gku['pattern_signature'],
            'cluster_metadata': gku['cluster_metadata'],
            'created_at': gku['created_at']
        }

    async def get_gkus_by_workspace(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Get all GKUs for a specific workspace."""
        results = []
        
        for gku in self.gkus.values():
            if gku['workspace_id'] == workspace_id:
                results.append({
                    'id': gku['id'],
                    'workspace_id': gku['workspace_id'],
                    'name': gku['name'],
                    'description': gku['description'],
                    'centroid_embedding': gku['centroid_embedding'],
                    'pattern_signature': gku['pattern_signature'],
                    'cluster_metadata': gku['cluster_metadata'],
                    'created_at': gku['created_at']
                })
        
        return results

    async def associate_akus_with_gku(self, gku_id: str, aku_ids: List[str]):
        """Associate AKUs with a GKU."""
        # In memory implementation - we'll store associations separately
        # Since we don't have a direct relationship table in memory, we'll add a field to AKUs
        for aku_id in aku_ids:
            if aku_id in self.akus:
                aku = self.akus[aku_id]
                if 'associated_gkus' not in aku:
                    aku['associated_gkus'] = []
                if gku_id not in aku['associated_gkus']:
                    aku['associated_gkus'].append(gku_id)

    async def get_akus_in_gku(self, gku_id: str) -> List[Dict[str, Any]]:
        """Get all AKUs associated with a specific GKU."""
        results = []
        
        for aku in self.akus.values():
            if ('associated_gkus' in aku and 
                gku_id in aku['associated_gkus']):
                
                results.append({
                    'id': aku.get('id'),
                    'content': aku.get('content'),
                    'embedding': aku.get('embedding'),
                    'aku_metadata': aku.get('aku_metadata', {}),
                    'created_at': aku.get('created_at')
                })
        
        return results

    async def get_akus_with_embeddings(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Get all AKUs in a workspace that have embeddings."""
        results = []
        
        for aku in self.akus.values():
            if (aku['workspace_id'] == workspace_id and 
                aku['embedding'] is not None and 
                not aku['is_archived']):
                
                results.append({
                    'id': aku.get('id'),
                    'content': aku.get('content'),
                    'embedding': aku.get('embedding'),
                    'aku_metadata': aku.get('aku_metadata', {}),
                    'episode_id': aku.get('episode_id'),
                    'created_at': aku.get('created_at')
                })
        
        return results

    async def update_gku_pattern_signature(self, gku_id: str, pattern_signature: Dict[str, Any]):
        """Update the pattern signature of an existing GKU."""
        if gku_id in self.gkus:
            self.gkus[gku_id]['pattern_signature'] = pattern_signature

    async def save_rl_state(self, model_id: str, weights: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Save RL model weights and state for persistence."""
        self.rl_states[model_id] = {
            'weights': weights,
            'metadata': metadata or {},
            'updated_at': datetime.now()
        }

    async def load_rl_state(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load RL model weights and state."""
        return self.rl_states.get(model_id)
