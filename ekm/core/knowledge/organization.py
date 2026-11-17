import numpy as np
import logging
import warnings
from typing import List, Dict, Any, Optional, Tuple
from ...storage.base import BaseStorage
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False  # Approximate nearest neighbor library

logger = logging.getLogger(__name__)


def apply_approximate_nearest_neighbor_clustering(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Apply clustering using approximate nearest neighbors for efficiency.
    """
    try:
        if FAISS_AVAILABLE:
            n_samples, n_features = embeddings.shape
            
            # Use FAISS for approximate nearest neighbors if dataset is large enough
            if n_samples > 1000:  # Use ANN for larger datasets
                # Build FAISS index
                d = embeddings.shape[1]  # dimension
                index = faiss.IndexFlatIP(d)  # Inner product (cosine similarity after normalization)
                
                # Normalize embeddings for cosine similarity
                embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # Add vectors to index
                index.add(embeddings_normalized.astype('float32'))
                
                # Find approximate nearest neighbors
                k = min(100, n_samples)  # Number of neighbors to find
                _, indices = index.search(embeddings_normalized.astype('float32'), k)
                
                # Use the neighbor information for clustering
                # Create a similarity matrix based on neighbor relationships
                similarity_matrix = np.zeros((n_samples, n_samples))
                for i in range(n_samples):
                    similarity_matrix[i, indices[i]] = 1.0  # Set neighbors as similar
                
                # Apply spectral clustering with the similarity matrix
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42,
                    assign_labels='discretize'
                )
                labels = clustering.fit_predict(similarity_matrix)
                return labels
            else:
                # For smaller datasets, use traditional spectral clustering
                similarity_matrix = cosine_similarity(embeddings)
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42,
                    assign_labels='discretize'
                )
                return clustering.fit_predict(similarity_matrix)
        else:
             # Fallback if FAISS not available
            similarity_matrix = cosine_similarity(embeddings)
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42,
                assign_labels='discretize'
            )
            return clustering.fit_predict(similarity_matrix)

    except Exception as e:
        logger.warning(f"Error in ANN clustering: {e}, falling back to standard spectral")
        similarity_matrix = cosine_similarity(embeddings)
        # Using simple k-means as safe fallback if spectral fails
        try:
             kmeans = KMeans(n_clusters=n_clusters, random_state=42)
             return kmeans.fit_predict(embeddings)
        except:
             return np.zeros(embeddings.shape[0], dtype=int)



def apply_hierarchical_clustering(embeddings: np.ndarray, n_clusters: Optional[int] = None, 
                                 distance_threshold: Optional[float] = 0.5) -> np.ndarray:
    """
    Apply hierarchical clustering with flexible stopping criteria.
    """
    try:
        if n_clusters is None and distance_threshold is None:
            n_clusters = max(2, len(embeddings) // 10)  # Default: roughly 10 items per cluster
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage='ward'
        )
        
        labels = clustering.fit_predict(embeddings)
        return labels
    except Exception as e:
        logger.error(f"Error in hierarchical clustering: {e}")
        # Fallback: return all in one cluster
        return np.zeros(len(embeddings), dtype=int)


def apply_multi_similarity_clustering(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Apply clustering using multiple similarity measures combined.
    """
    try:
        n_samples = len(embeddings)
        
        # Compute multiple similarity matrices
        cosine_sim = cosine_similarity(embeddings)
        euclidean_dist = euclidean_distances(embeddings)
        # Normalize euclidean distances to [0,1] range
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Combine similarities (weighted average)
        # Combine similarities (weighted average)
        combined_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
        
        # Apply spectral clustering with combined similarity
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='discretize'
        )
        labels = clustering.fit_predict(combined_sim)
        return labels
    except Exception as e:
        logger.error(f"Error in multi-similarity clustering: {e}")
        # Fallback to cosine similarity clustering
        return apply_spectral_clustering(embeddings, n_clusters)


def compute_attention_weighted_centroids(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute attention-weighted centroids with uncertainty quantification.
    
    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        labels: Cluster labels for each sample
        
    Returns:
        Tuple of (centroids, uncertainties)
    """
    unique_labels = np.unique(labels)
    n_features = embeddings.shape[1]
    
    centroids = np.zeros((len(unique_labels), n_features))
    uncertainties = np.zeros(len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_embeddings = embeddings[mask]
        
        if len(cluster_embeddings) == 0:
            continue
            
        # Calculate attention weights based on similarity to cluster center
        cluster_center = np.mean(cluster_embeddings, axis=0)
        similarities = np.dot(cluster_embeddings, cluster_center) / (
            np.linalg.norm(cluster_embeddings, axis=1) * np.linalg.norm(cluster_center) + 1e-8
        )
        
        # Normalize similarities to get attention weights
        attention_weights = similarities / (np.sum(similarities) + 1e-8)
        
        # Compute attention-weighted centroid
        weighted_centroid = np.average(cluster_embeddings, axis=0, weights=attention_weights)
        centroids[i] = weighted_centroid
        
        # Calculate uncertainty as the average distance from the centroid
        distances = np.linalg.norm(cluster_embeddings - weighted_centroid, axis=1)
        uncertainties[i] = np.mean(distances)
    
    return centroids, uncertainties


def extract_enhanced_pattern_signature(embeddings: np.ndarray, labels: np.ndarray, 
                                      cluster_idx: int) -> Dict[str, Any]:
    """
    Extract enhanced pattern signature for a cluster.
    """
    mask = labels == cluster_idx
    cluster_embeddings = embeddings[mask]
    
    if len(cluster_embeddings) == 0:
        return {}
    
    # Basic statistics
    n_points = len(cluster_embeddings)
    
    # Structural features of the cluster's similarity graph. These use the same
    # functions (and therefore the same keys and definitions) as the query side
    # in query_analysis.py, so calculate_pattern_similarity compares like with like.
    from scipy.sparse import csr_matrix
    from ..patterns import count_graphlets, compute_random_walk_patterns, compute_degree_statistics

    similarity_matrix = cosine_similarity(cluster_embeddings)
    # Set diagonal to 0 to exclude self-loops
    np.fill_diagonal(similarity_matrix, 0)
    # Threshold to create adjacency matrix
    threshold = 0.5
    # Sparse, like the query graph, so both sides take the same code paths
    adj_matrix = csr_matrix((similarity_matrix > threshold).astype(np.float32))

    graphlets = count_graphlets(adj_matrix)
    degree_stats = compute_degree_statistics(adj_matrix)
    random_walk_stats = compute_random_walk_patterns(adj_matrix)
    
    # Embedding statistics
    embedding_stats = {
        'mean_embedding_norm': float(np.mean(np.linalg.norm(cluster_embeddings, axis=1))),
        'std_embedding_norm': float(np.std(np.linalg.norm(cluster_embeddings, axis=1))),
        'intra_cluster_similarity': float(np.mean(cosine_similarity(cluster_embeddings).flatten())) if n_points > 1 else 1.0
    }
    
    return {
        'graphlets': graphlets,
        'degree_stats': degree_stats,
        'random_walks': random_walk_stats,
        'embedding_stats': embedding_stats,
        'cluster_size': n_points
    }


def compute_cluster_quality_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute quality metrics for clustering results.
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    metrics = {}
    
    if n_clusters > 1 and len(embeddings) > n_clusters:
        try:
            # Silhouette score
            sil_score = silhouette_score(embeddings, labels)
            metrics['silhouette_score'] = sil_score
        except:
            metrics['silhouette_score'] = 0.0
        
        try:
            # Calinski-Harabasz score
            ch_score = calinski_harabasz_score(embeddings, labels)
            metrics['calinski_harabasz_score'] = ch_score
        except:
            metrics['calinski_harabasz_score'] = 0.0
        
        try:
            # Davies-Bouldin score
            db_score = davies_bouldin_score(embeddings, labels)
            metrics['davies_bouldin_score'] = db_score
        except:
            metrics['davies_bouldin_score'] = float('inf')
    
    # Inertia (within-cluster sum of squares)
    inertia = 0.0
    for label in unique_labels:
        mask = labels == label
        cluster_points = embeddings[mask]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            inertia += np.sum((cluster_points - centroid) ** 2)
    metrics['inertia'] = float(inertia)
    
    # Number of clusters
    metrics['n_clusters'] = n_clusters
    
    return metrics


def apply_adaptive_clustering(embeddings: np.ndarray, target_n_clusters: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply adaptive clustering with automatic parameter tuning.
    """
    n_samples, n_features = embeddings.shape

    # Determine optimal number of clusters if not provided
    if target_n_clusters is None:
        if n_samples < 10:
            # Very small datasets: use minimal clustering
            target_n_clusters = max(1, min(n_samples, 2))  # At most 2 clusters, or n_samples if smaller
        elif n_samples < 50:
            # Small datasets: fewer clusters to avoid overfitting
            target_n_clusters = max(2, min(n_samples // 3, 5))  # Max 5 clusters for small datasets, at least 3 samples per cluster
        else:
            # Heuristic: aim for 5-20 samples per cluster
            target_n_clusters = max(2, min(n_samples // 10, 50))  # Between 2 and 50 clusters

    # Ensure target_n_clusters is not greater than n_samples
    target_n_clusters = min(target_n_clusters, n_samples)

    # If we only have 1 cluster to make, assign all points to the same cluster
    if target_n_clusters <= 1:
        return np.zeros(n_samples, dtype=int), {'n_clusters': 1, 'silhouette_score': 0.0}

    # Choose clustering algorithm based on data characteristics
    if n_samples < 100:
        # Small dataset: use spectral clustering
        labels = apply_spectral_clustering(embeddings, target_n_clusters)
    elif n_samples < 1000:
        # Medium dataset: use multi-similarity clustering
        labels = apply_multi_similarity_clustering(embeddings, target_n_clusters)
    else:
        # Large dataset: use approximate nearest neighbor clustering
        labels = apply_approximate_nearest_neighbor_clustering(embeddings, target_n_clusters)

    # Compute quality metrics
    quality_metrics = compute_cluster_quality_metrics(embeddings, labels)

    # Adjust clustering if quality is poor and we have enough samples to adjust
    if quality_metrics.get('silhouette_score', 0) < 0.2 and n_samples > 10 and target_n_clusters > 1:
        # Try with fewer clusters if silhouette score is poor
        adjusted_n_clusters = max(1, min(target_n_clusters // 2, n_samples))  # Ensure it's not more than n_samples
        if adjusted_n_clusters != target_n_clusters and adjusted_n_clusters > 0:
            if n_samples < 1000:
                labels = apply_spectral_clustering(embeddings, adjusted_n_clusters)
            else:
                labels = apply_approximate_nearest_neighbor_clustering(embeddings, adjusted_n_clusters)
            quality_metrics = compute_cluster_quality_metrics(embeddings, labels)

    return labels, quality_metrics


def incremental_cluster_update(current_embeddings: np.ndarray, current_labels: np.ndarray,
                              new_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update clustering incrementally with new embeddings.
    """
    if len(new_embeddings) == 0:
        return current_labels, np.array([])
    
    # Combine current and new embeddings
    all_embeddings = np.vstack([current_embeddings, new_embeddings])
    
    # Determine number of clusters based on current clustering
    n_current_clusters = len(np.unique(current_labels))
    
    # Apply clustering to the combined dataset
    all_labels, _ = apply_adaptive_clustering(all_embeddings, n_current_clusters)
    
    # Return updated labels for all items and labels for new items only
    current_updated_labels = all_labels[:len(current_embeddings)]
    new_item_labels = all_labels[len(current_embeddings):]
    
    return current_updated_labels, new_item_labels


def apply_spectral_clustering(embeddings: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
    """Perform spectral clustering on embeddings."""
    try:
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics.pairwise import cosine_similarity

        n_samples = len(embeddings)
        
        if n_clusters is None:
            # For small datasets, use fewer clusters
            if n_samples < 10:
                n_clusters = max(1, n_samples)  # Each point may be its own cluster for very small sets
            elif n_samples < 50:
                n_clusters = max(2, n_samples // 4)  # Fewer clusters for small datasets
            else:
                n_clusters = max(2, min(int(np.sqrt(n_samples)), 10))  # Original logic for larger sets

        # Ensure n_clusters is not greater than n_samples
        n_clusters = min(n_clusters, n_samples)

        if n_clusters <= 1:
            # If only 1 cluster is needed, assign all points to the same cluster
            return np.zeros(n_samples, dtype=int)

        similarity_matrix = cosine_similarity(embeddings)
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='discretize'
        )
        return clustering.fit_predict(similarity_matrix)
    except Exception as e:
        logger.error(f"Spectral clustering failed: {e}")
        return np.zeros(len(embeddings), dtype=int)


def _to_epoch(value: Any) -> Optional[float]:
    """Seconds since the epoch from a datetime, ISO-8601 string or number; None if unusable."""
    if value is None or isinstance(value, bool):
        return None
    if hasattr(value, 'timestamp'):
        try:
            return float(value.timestamp())
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        from datetime import datetime
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except ValueError:
            return None
    return None


def fact_time_summary(akus: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    When the knowledge in a cluster dates from, as epoch seconds.

    Uses each AKU's `aku_metadata['event_time']` when present (the time the fact
    is about), otherwise its `created_at` (when it was ingested). A GKU's own
    `created_at` is not used: every GKU is recreated at each consolidation, so it
    only says when consolidation last ran.
    """
    times, sources = [], set()
    for aku in akus:
        event_time = _to_epoch((aku.get('aku_metadata') or {}).get('event_time'))
        if event_time is not None:
            times.append(event_time)
            sources.add('event_time')
            continue
        ingested = _to_epoch(aku.get('created_at'))
        if ingested is not None:
            times.append(ingested)
            sources.add('ingested_at')
    if not times:
        return None
    return {
        'median': float(np.median(times)),
        'oldest': float(np.min(times)),
        'newest': float(np.max(times)),
        'source': '+'.join(sorted(sources))
    }


def gku_reference_time(gku: Dict[str, Any]) -> Any:
    """Time the temporal head should use for a GKU: its facts' median time, else its creation time."""
    fact_time = (gku.get('cluster_metadata') or {}).get('fact_time')
    if isinstance(fact_time, dict) and fact_time.get('median') is not None:
        return fact_time['median']
    return gku.get('created_at')


def _facts_text(contents: List[str], max_chars: int) -> str:
    """Bullet list of a cluster's facts, cut at max_chars to bound LLM cost."""
    lines, used = [], 0
    for content in contents:
        if not content:
            continue
        line = f"- {content.strip()}"
        if used + len(line) > max_chars and lines:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


async def form_gkus_from_akus(
    storage: BaseStorage,
    workspace_id: str,
    min_cluster_size: int = 2,
    llm: Optional[Any] = None,
    max_fact_chars: int = 6000
) -> List[str]:
    """
    Form Global Knowledge Units (GKUs) from AKUs using enhanced clustering algorithms.

    Args:
        storage: Storage interface to access AKUs
        workspace_id: ID of the workspace to process
        min_cluster_size: Minimum number of AKUs required to form a cluster/GKU
        llm: If given, each GKU's structural signature is built from the
            entity-relation graph the LLM extracts from the GKU's facts, the same
            way query graphs are built. This costs one LLM call per GKU. Without
            it (or if extraction fails) the signature falls back to the graph of
            embedding similarities, which mostly repeats the semantic head.
        max_fact_chars: Maximum characters of fact text sent per GKU

    Returns:
        List of created GKU IDs
    """
    try:
        # Get all AKUs in the workspace with embeddings
        all_akus = await storage.get_akus_with_embeddings(workspace_id)

        if len(all_akus) < min_cluster_size:
            logger.info(f"Not enough AKUs with embeddings to form clusters: {len(all_akus)} < {min_cluster_size}")
            return []

        # Extract embeddings and corresponding AKU IDs
        embeddings = []
        aku_ids = []
        aku_contents = {}
        akus_by_id = {}
        for aku in all_akus:
            if aku.get('embedding') is not None:
                embeddings.append(aku['embedding'])
                aku_ids.append(aku['id'])
                aku_contents[aku['id']] = aku.get('content') or ""
                akus_by_id[aku['id']] = aku

        extractor = None
        if llm is not None:
            from ..query_analysis import QueryGraphExtractor
            extractor = QueryGraphExtractor(llm)

        if len(embeddings) < min_cluster_size:
            logger.info(f"Not enough AKUs with embeddings to form clusters: {len(embeddings)} < {min_cluster_size}")
            return []

        # Convert to numpy array
        embedding_array = np.array(embeddings)

        # Apply adaptive clustering with quality assessment
        cluster_labels, quality_metrics = apply_adaptive_clustering(embedding_array)

        # Compute attention-weighted centroids and uncertainties
        centroids, uncertainties = compute_attention_weighted_centroids(embedding_array, cluster_labels)

        # Group AKUs by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            label = int(label)  # Ensure it's a regular int, not numpy int
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(aku_ids[idx])

        # Create GKUs for clusters that meet minimum size requirements
        created_gku_ids = []
        for cluster_label, aku_ids_in_cluster in clusters.items():
            if len(aku_ids_in_cluster) >= min_cluster_size:
                # Get cluster embeddings and indices
                cluster_indices = [i for i, label in enumerate(cluster_labels) if int(label) == cluster_label]
                cluster_embeddings = embedding_array[cluster_indices]
                
                # Use attention-weighted centroid
                centroid_idx = cluster_label
                centroid_embedding = centroids[centroid_idx].tolist() if centroid_idx < len(centroids) else np.mean(cluster_embeddings, axis=0).tolist()
                
                # Calculate uncertainty for this cluster
                cluster_uncertainty = uncertainties[centroid_idx] if centroid_idx < len(uncertainties) else 0.0

                # Create GKU name based on cluster
                gku_name = f"Concept Cluster {cluster_label} ({len(aku_ids_in_cluster)} AKUs)"

                # Create GKU description
                gku_description = f"A cluster of {len(aku_ids_in_cluster)} related Atomic Knowledge Units formed through enhanced clustering."

                # Create enhanced pattern signature for the cluster
                pattern_signature = extract_enhanced_pattern_signature(embedding_array, cluster_labels, cluster_label)
                pattern_signature['structure_source'] = 'embedding_similarity'

                # Prefer the entity-relation graph of the cluster's facts: it is
                # the same kind of graph as the query graph, and its shape is not
                # derived from the embeddings the semantic head already uses.
                if extractor is not None:
                    facts = _facts_text([aku_contents.get(a, "") for a in aku_ids_in_cluster], max_fact_chars)
                    relational = await extractor.extract_text_pattern(facts, label="FACTS") if facts else None
                    if relational is not None:
                        pattern_signature.update(relational)
                        pattern_signature['structure_source'] = 'entity_relations'
                    else:
                        logger.warning(f"Relation extraction failed for cluster {cluster_label}; "
                                       f"using the embedding-similarity graph instead")

                # Add uncertainty information
                pattern_signature['uncertainty'] = float(cluster_uncertainty)
                pattern_signature['quality_metrics'] = quality_metrics

                # Create the GKU using storage
                gku_id = await storage.save_gku(
                    workspace_id=workspace_id,
                    name=gku_name,
                    description=gku_description,
                    centroid_embedding=centroid_embedding,
                    pattern_signature=pattern_signature,
                    cluster_metadata={
                        'algorithm': 'enhanced_clustering',
                        'cluster_size': len(aku_ids_in_cluster),
                        'created_from_aku_count': len(aku_ids_in_cluster),
                        'uncertainty': float(cluster_uncertainty),
                        'quality_metrics': quality_metrics,
                        # Used by the temporal head instead of the GKU's creation time
                        'fact_time': fact_time_summary([akus_by_id[a] for a in aku_ids_in_cluster])
                    }
                )

                # Associate AKUs with the GKU
                await storage.associate_akus_with_gku(gku_id, aku_ids_in_cluster)

                created_gku_ids.append(gku_id)

                logger.info(f"Created GKU {gku_id} with {len(aku_ids_in_cluster)} AKUs, uncertainty: {cluster_uncertainty:.3f}")

        logger.info(f"Successfully created {len(created_gku_ids)} GKUs from {len(all_akus)} AKUs with quality metrics: {quality_metrics}")
        return created_gku_ids

    except Exception as e:
        logger.error(f"Failed to form GKUs from AKUs: {e}")
        return []


def calculate_avg_similarity_within_cluster(cluster_embeddings: np.ndarray) -> float:
    """
    Calculate the average similarity within a cluster of embeddings.
    """
    if len(cluster_embeddings) < 2:
        return 1.0  # Perfect similarity for single item

    similarities = []
    for i in range(len(cluster_embeddings)):
        for j in range(i + 1, len(cluster_embeddings)):
            sim = cosine_similarity_single(cluster_embeddings[i], cluster_embeddings[j])
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.0


def cosine_similarity_single(vec1, vec2) -> float:
    """
    Calculate cosine similarity between two vectors.
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

