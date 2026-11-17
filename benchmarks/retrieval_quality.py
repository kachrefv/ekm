"""
Retrieval quality: recall@k and MRR over labelled queries, per configuration.

Use it to decide what the retrieval knobs are worth on *your* data, e.g.
whether the structural head justifies its per-query LLM call, or whether the
GKU bonus helps. Each variant runs the same queries through EKM.retrieve.

    from benchmarks.retrieval_quality import evaluate_variants
    report = await evaluate_variants(ekm, workspace_id, [
        {"query": "When were bees domesticated?", "relevant": ["...fact or chunk text..."]},
        ...
    ], k=5)

A result counts as relevant when its content equals, or contains, one of the
query's "relevant" strings (ids are not stable across re-ingestion; text is).

Run as a script for an offline smoke test on a synthetic corpus. Its
embeddings are made up, so its numbers only show the harness works; they say
nothing about real retrieval quality.

    python -m benchmarks.retrieval_quality [--k 5]
"""
import argparse
import asyncio
import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np


def _is_relevant(content: str, relevant: List[str]) -> bool:
    content = (content or "").strip().lower()
    return any(r.strip().lower() == content or r.strip().lower() in content for r in relevant if r.strip())


def score_ranking(contents: List[str], relevant: List[str], k: int) -> Dict[str, float]:
    """recall@k (share of relevant strings found in the top k) and reciprocal rank."""
    top = contents[:k]
    found = {r for r in relevant for c in top if _is_relevant(c, [r])}
    recall = len(found) / len(relevant) if relevant else 0.0
    rr = next((1.0 / (i + 1) for i, c in enumerate(top) if _is_relevant(c, relevant)), 0.0)
    return {"recall": recall, "rr": rr}


async def evaluate(ekm, workspace_id: str, labelled: List[Dict[str, Any]], k: int = 5,
                   **retrieve_kwargs) -> Dict[str, float]:
    """Mean recall@k and MRR of EKM.retrieve over the labelled queries."""
    recalls, rrs = [], []
    for item in labelled:
        out = await ekm.retrieve(workspace_id, item["query"], top_k=k, use_rl_feedback=False, **retrieve_kwargs)
        s = score_ranking([r.get("content", "") for r in out["results"]], item["relevant"], k)
        recalls.append(s["recall"])
        rrs.append(s["rr"])
    return {f"recall@{k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "mrr": round(float(np.mean(rrs)), 4) if rrs else 0.0,
            "queries": len(labelled)}


@contextmanager
def _head_weights(ekm, weights: Optional[Dict[str, float]]):
    """Temporarily replace the service's head weights."""
    weighting = ekm.retrieval_service.adaptive_weighting
    saved = dict(weighting.head_weights)
    if weights is not None:
        weighting.head_weights = dict(weights)
    try:
        yield
    finally:
        weighting.head_weights = saved


@contextmanager
def _gku_bonus(value: Optional[float]):
    import ekm.core.retrieval.service as service
    saved = service.GKU_CONTEXT_BOOST
    if value is not None:
        service.GKU_CONTEXT_BOOST = value
    try:
        yield
    finally:
        service.GKU_CONTEXT_BOOST = saved


VARIANTS: Dict[str, Dict[str, Any]] = {
    "hybrid (default)": {},
    "facts only": {"retrieve": {"mode": "facts"}},
    "episodes only": {"retrieve": {"mode": "episodic"}},
    # Structural weight 0 also skips the query-graph LLM call
    "no structural head": {"weights": {"semantic": 0.8, "structural": 0.0, "temporal": 0.2}},
    "semantic head only": {"weights": {"semantic": 1.0, "structural": 0.0, "temporal": 0.0}},
    "no GKU bonus": {"gku_bonus": 0.0},
}


async def evaluate_variants(ekm, workspace_id: str, labelled: List[Dict[str, Any]], k: int = 5,
                            variants: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, float]]:
    report = {}
    for name, spec in (variants or VARIANTS).items():
        with _head_weights(ekm, spec.get("weights")), _gku_bonus(spec.get("gku_bonus")):
            report[name] = await evaluate(ekm, workspace_id, labelled, k, **spec.get("retrieve", {}))
    return report


# --- Offline smoke test --------------------------------------------------------------

class _LookupEmbeddings:
    """Returns pre-assigned vectors for known texts (offline, deterministic)."""

    def __init__(self, vectors: Dict[str, List[float]], dim: int):
        self.vectors, self.dim = vectors, dim

    async def embed_query(self, text: str) -> List[float]:
        return self.vectors.get(text, [0.0] * self.dim)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed_query(t) for t in texts]


class _NoGraphLLM:
    async def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        return "{}"


async def _synthetic_demo(k: int) -> Dict[str, Dict[str, float]]:
    from ekm.core.mesh import EKM
    from ekm.storage.memory import MemoryStorage

    rng = np.random.default_rng(7)
    dim, topics, facts_per_topic = 64, 8, 6
    centres = rng.normal(size=(topics, dim))
    vectors, labelled, storage = {}, [], MemoryStorage()

    def vec(base, noise):
        v = base + rng.normal(scale=noise, size=dim)
        return (v / np.linalg.norm(v)).tolist()

    for t in range(topics):
        episode_text = f"topic {t} source chunk"
        episode_id = await storage.save_episode("demo", episode_text, "", vec(centres[t], 0.6), {})
        facts = [{"content": f"topic {t} fact {j}", "embedding": vec(centres[t], 0.5)} for j in range(facts_per_topic)]
        ids = await storage.save_akus("demo", episode_id, facts)
        centroid = np.mean([f["embedding"] for f in facts], axis=0)
        gku_id = await storage.save_gku("demo", f"Topic {t}", centroid_embedding=centroid.tolist())
        await storage.associate_akus_with_gku(gku_id, ids)
        query = f"question about topic {t}"
        vectors[query] = vec(centres[t], 0.7)
        labelled.append({"query": query, "relevant": [f"topic {t} fact {j}" for j in range(facts_per_topic)]
                         + [episode_text]})

    ekm = EKM(storage, _NoGraphLLM(), _LookupEmbeddings(vectors, dim))
    return await evaluate_variants(ekm, "demo", labelled, k)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    import logging
    logging.getLogger("ekm").setLevel(logging.ERROR)  # the scripted LLM returns no query graph
    report = asyncio.run(_synthetic_demo(args.k))
    print("Synthetic corpus (made-up embeddings: checks the harness, not real quality)")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
