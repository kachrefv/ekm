"""
Latency benchmarks for EKM components.

Runs the real code paths (training, consolidation, retrieval, attention heads)
against in-memory storage with offline stand-ins for the LLM and embeddings, so
it needs no API keys and makes no network calls. Every number is measured.

It measures speed only. Embeddings are hash-based and the LLM is scripted, so
retrieval *quality* means nothing here; benchmark that with real models and a
labelled dataset.

Usage:
    python -m benchmarks.performance [--docs 50] [--queries 20]
"""
import argparse
import asyncio
import hashlib
import json
import re
import statistics
import time
from typing import Any, Dict, List

import numpy as np

from ekm.core.attention import QKVAttentionRetriever
from ekm.core.consolidation import SleepConsolidator
from ekm.core.mesh import EKM
from ekm.providers.base import BaseEmbeddings, BaseLLM
from ekm.storage.memory import MemoryStorage

TOPICS = ["solar panels", "honey bees", "tidal energy", "sourdough bread", "coral reefs",
          "electric buses", "glaciers", "chess openings", "vaccines", "fiber optics"]


class ScriptedLLM(BaseLLM):
    """Answers the two prompts EKM sends with well-formed JSON and no network."""

    async def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        await asyncio.sleep(0)  # yield like a real async client
        if "graph" in system_prompt.lower():
            words = re.findall(r"[A-Za-z]{5,}", user_message)[-4:] or ["query"]
            edges = [[a, b, "related to"] for a, b in zip(words, words[1:])]
            return json.dumps({"nodes": words, "edges": edges})
        text = user_message.rsplit("TEXT:", 1)[-1]
        facts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20][:6]
        return json.dumps({"summary": text[:120], "facts": facts or [text[:200]]})


class TopicEmbeddings(BaseEmbeddings):
    """Deterministic offline embeddings: a shared vector per topic plus per-text
    noise, centred on zero. (ekm.providers.mock.MockEmbeddings maps hashes into
    [0, 1], so every pair has cosine ~0.85 and consolidation merges nearly all
    facts, leaving an almost empty workspace to time retrieval on.)"""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    def _vector(self, seed_text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little")
        return np.random.default_rng(seed).normal(size=self.embedding_dim)

    async def embed_query(self, text: str) -> List[float]:
        topic = next((t for t in TOPICS if t in text), "")
        vec = 0.8 * self._vector("topic:" + topic) + 0.6 * self._vector(text)
        return (vec / np.linalg.norm(vec)).tolist()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed_query(t) for t in texts]


def make_document(i: int) -> str:
    topic = TOPICS[i % len(TOPICS)]
    return " ".join(
        f"Fact {j} about {topic} number {i}: {topic} relate to measurement {i * 7 + j} in study {j}."
        for j in range(8)
    )


def summarize(samples: List[float]) -> Dict[str, float]:
    ms = sorted(s * 1000 for s in samples)
    return {
        "runs": len(ms),
        "mean_ms": round(statistics.mean(ms), 2),
        "p50_ms": round(ms[len(ms) // 2], 2),
        "p95_ms": round(ms[min(len(ms) - 1, int(len(ms) * 0.95))], 2),
    }


async def timed(coro_factory, runs: int) -> List[float]:
    samples = []
    for i in range(runs):
        start = time.perf_counter()
        await coro_factory(i)
        samples.append(time.perf_counter() - start)
    return samples


class EKMBenchmarkSuite:
    """Benchmark suite for EKM latency."""

    def __init__(self, num_docs: int = 50, num_queries: int = 20, embedding_dim: int = 384):
        self.num_docs = num_docs
        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.storage = MemoryStorage()
        self.llm = ScriptedLLM()
        self.embeddings = TopicEmbeddings(embedding_dim=embedding_dim)
        self.ekm = EKM(self.storage, self.llm, self.embeddings)
        self.workspace = "benchmark"

    async def benchmark_training(self) -> Dict[str, Any]:
        print(f"Training on {self.num_docs} documents...")
        samples = await timed(
            lambda i: self.ekm.train(self.workspace, make_document(i), title=f"doc {i}"), self.num_docs)
        return {**summarize(samples), "facts_stored": len(self.storage.akus)}

    async def benchmark_consolidation(self) -> Dict[str, Any]:
        print("Running consolidation...")
        consolidator = SleepConsolidator(self.storage, self.llm, self.embeddings)
        start = time.perf_counter()
        result = await consolidator.run_consolidation(self.workspace)
        elapsed = time.perf_counter() - start
        self.ekm.scalable_backend.invalidate_workspace_index(self.workspace)
        return {
            "seconds": round(elapsed, 3),
            "clusters_created": len(result.created_gkus),
            "facts_merged": len(result.archived_akus),
            "compression_ratio": round(result.compression_ratio, 3),
        }

    async def benchmark_retrieval(self) -> Dict[str, Any]:
        results = {}
        for label, flags in {
            "default_path": {},
            "flat_three_heads": {"use_hierarchical_attention": False},
            "episodic_mode": {"mode": "episodic"},
        }.items():
            print(f"Retrieval ({label}) x{self.num_queries}...")
            samples = await timed(
                lambda i: self.ekm.retrieve(
                    self.workspace, f"What is known about {TOPICS[i % len(TOPICS)]}?", top_k=5, **flags),
                self.num_queries)
            results[label] = summarize(samples)
        return results

    async def benchmark_attention_heads(self, num_gkus: int = 500) -> Dict[str, Any]:
        print(f"Attention heads over {num_gkus} clusters...")
        rng = np.random.default_rng(0)
        retriever = QKVAttentionRetriever(embedding_dim=self.embedding_dim)
        query = rng.normal(size=self.embedding_dim)
        centroids = [rng.normal(size=self.embedding_dim) for _ in range(num_gkus)]
        pattern = {"graphlets": {"triangle": 1, "wedge": 2}, "degree_stats": {"mean_degree": 2.0, "std_degree": 0.5}}
        times = [{"created_at": float(t)} for t in rng.uniform(0, 1e8, size=num_gkus)]

        def run(fn):
            start = time.perf_counter()
            fn()
            return round((time.perf_counter() - start) * 1000, 2)

        return {
            "semantic_ms": run(lambda: retriever.semantic_head.compute_attention(query, centroids)),
            "structural_ms": run(lambda: retriever.structural_head.compute_attention(pattern, [pattern] * num_gkus)),
            "temporal_ms": run(lambda: retriever.temporal_head.compute_attention({"timestamp": 1e8}, times)),
        }

    async def run_all_benchmarks(self) -> Dict[str, Any]:
        results = {"training": await self.benchmark_training()}
        results["consolidation"] = await self.benchmark_consolidation()
        results["retrieval"] = await self.benchmark_retrieval()
        results["attention_heads"] = await self.benchmark_attention_heads()
        return results


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--docs", type=int, default=50)
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args()

    results = await EKMBenchmarkSuite(args.docs, args.queries).run_all_benchmarks()
    print("\n=== EKM latency (offline stand-ins; speed only, not quality) ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
