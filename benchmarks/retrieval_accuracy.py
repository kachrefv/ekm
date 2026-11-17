"""
Retrieval accuracy on HotpotQA (multi-hop QA with labelled supporting paragraphs).

All paragraphs of the sampled questions (2 gold + 8 hard distractors each) go
into one shared corpus. For each question, every system returns a ranked list;
results are mapped back to their source paragraphs, and we score how many of the
question's 2 gold paragraphs appear near the top.

Systems (same embedding model everywhere, so only the method differs):
  dense_paragraphs   plain RAG: cosine between the query and each whole paragraph
  dense_facts        cosine between the query and EKM's extracted facts, nothing else
  ekm_hybrid         EKM retrieve() with defaults, before consolidation
  after consolidation (sleep):
  ekm_hybrid_slept   EKM retrieve() with defaults
  ekm_facts_slept    mode='facts'
  ekm_flat_slept     use_hierarchical_attention=False (flat three-head path)

Metrics per question (averaged): recall@2 and recall@5 of the gold paragraphs,
both_gold@5 (both supporting paragraphs in the top 5, what multi-hop needs), MRR.

Data: rows JSON from the Hugging Face datasets server, e.g.
  https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=validation&offset=0&length=40

Usage:
  OPENROUTER_API_KEY=... python -m benchmarks.retrieval_accuracy rows.json --questions 40
  python -m benchmarks.retrieval_accuracy rows.json --offline     # plumbing check, no API calls
"""
import argparse
import asyncio
import json
import statistics
import time
from typing import Any, Dict, List, Sequence

import numpy as np

from ekm.core.consolidation import SleepConsolidator
from ekm.core.mesh import EKM
from ekm.storage.memory import MemoryStorage

import functools

# Show progress immediately even when output is redirected to a file
print = functools.partial(print, flush=True)  # noqa: A001

WORKSPACE = "hotpot-eval"
RESULTS_PER_QUERY = 20


def load_questions(path: str, limit: int) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        rows = [r["row"] for r in json.load(f)["rows"]]
    return rows[:limit]


def build_corpus(questions: List[Dict[str, Any]]) -> Dict[str, str]:
    """title -> paragraph text, deduplicated across questions."""
    corpus: Dict[str, str] = {}
    for q in questions:
        for title, sentences in zip(q["context"]["title"], q["context"]["sentences"]):
            corpus.setdefault(title, f"{title}. " + "".join(sentences).strip())
    return corpus


def score(ranked_titles: Sequence[str], gold: Sequence[str]) -> Dict[str, float]:
    gold = set(gold)
    first_hit = next((i for i, t in enumerate(ranked_titles) if t in gold), None)
    return {
        "recall@2": len(gold & set(ranked_titles[:2])) / len(gold),
        "recall@5": len(gold & set(ranked_titles[:5])) / len(gold),
        "both_gold@5": float(gold <= set(ranked_titles[:5])),
        "mrr": 1.0 / (first_hit + 1) if first_hit is not None else 0.0,
    }


def dedupe(titles: List[str]) -> List[str]:
    seen, out = set(), []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


class Provenance:
    """Maps any retrieved id (episode, fact, merged fact) to its source paragraph titles."""

    def __init__(self, storage: MemoryStorage):
        self.storage = storage

    def episode_title(self, episode_id: Any) -> List[str]:
        episode = self.storage.episodes.get(episode_id)
        title = (episode or {}).get("meta_data", {}).get("title")
        return [title] if title else []

    def aku_titles(self, aku_id: Any, depth: int = 0) -> List[str]:
        aku = self.storage.akus.get(aku_id)
        if not aku or depth > 5:
            return []
        if aku.get("episode_id"):
            return self.episode_title(aku["episode_id"])
        sources = (aku.get("aku_metadata") or {}).get("consolidated_from") or []
        return dedupe([t for s in sources for t in self.aku_titles(s, depth + 1)])

    def result_titles(self, result: Dict[str, Any]) -> List[str]:
        if result.get("layer") == "episodic":
            return self.episode_title(result["id"])
        return self.aku_titles(result["id"])


def cosine_rank(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return np.argsort(-(matrix @ query) / np.where(norms == 0, 1, norms))


async def run(args) -> Dict[str, Any]:
    if args.offline:
        from benchmarks.performance import ScriptedLLM, TopicEmbeddings
        llm, embeddings = ScriptedLLM(), TopicEmbeddings()
        provider_label = "offline stand-ins (plumbing check only)"
    else:
        from ekm.providers.openrouter import OpenRouterProvider
        llm = embeddings = OpenRouterProvider()
        provider_label = f"OpenRouter: {llm.model_name} + {llm.embedding_model}"

    questions = load_questions(args.data, args.questions)
    corpus = build_corpus(questions)
    titles = list(corpus)
    print(f"{len(questions)} questions, {len(corpus)} paragraphs, {provider_label}")

    storage = MemoryStorage()
    ekm = EKM(storage, llm, embeddings)
    provenance = Provenance(storage)
    timings: Dict[str, float] = {}

    # --- Ingest -------------------------------------------------------------
    start = time.perf_counter()
    semaphore = asyncio.Semaphore(args.concurrency)
    failures: List[str] = []

    async def ingest(title: str):
        async with semaphore:
            try:
                await ekm.train(WORKSPACE, corpus[title], title=title)
            except Exception as e:  # keep going; report how many paragraphs failed
                failures.append(f"{title}: {e}")

    await asyncio.gather(*(ingest(t) for t in titles))
    timings["ingest_s"] = time.perf_counter() - start
    print(f"Ingested in {timings['ingest_s']:.0f}s: {len(storage.akus)} facts, "
          f"{len(storage.episodes)} episodes, {len(failures)} failed paragraphs")

    # --- Baseline query embeddings (shared by both dense baselines) ---------
    query_vecs = np.array(await embeddings.embed_documents([q["question"] for q in questions]))
    paragraph_vecs = np.array(await embeddings.embed_documents([corpus[t] for t in titles]))

    def fact_matrix():
        active = [a for a in storage.akus.values() if not a.get("is_archived") and a.get("embedding")]
        return active, np.array([a["embedding"] for a in active], dtype=float)

    rankings: Dict[str, List[List[str]]] = {}

    rankings["dense_paragraphs"] = [
        [titles[i] for i in cosine_rank(qv, paragraph_vecs)[:RESULTS_PER_QUERY]] for qv in query_vecs]

    facts, matrix = fact_matrix()
    rankings["dense_facts"] = [
        dedupe([t for i in cosine_rank(qv, matrix)[:RESULTS_PER_QUERY * 3] for t in provenance.aku_titles(facts[i]["id"])])
        for qv in query_vecs]

    async def ekm_rankings(label: str, **kwargs) -> List[List[str]]:
        start = time.perf_counter()
        out = []
        for q in questions:
            response = await ekm.retrieve(WORKSPACE, q["question"], top_k=RESULTS_PER_QUERY, **kwargs)
            out.append(dedupe([t for r in response["results"] for t in provenance.result_titles(r)]))
        timings[f"{label}_s_per_query"] = (time.perf_counter() - start) / len(questions)
        print(f"  {label}: {timings[f'{label}_s_per_query']:.1f}s per query")
        return out

    print("Retrieving before consolidation...")
    rankings["ekm_hybrid"] = await ekm_rankings("ekm_hybrid")

    print("Consolidating (sleep)...")
    start = time.perf_counter()
    result = await SleepConsolidator(storage, llm, embeddings).run_consolidation(WORKSPACE)
    ekm.scalable_backend.invalidate_workspace_index(WORKSPACE)
    timings["consolidation_s"] = time.perf_counter() - start
    consolidation = {
        "clusters": len(result.created_gkus), "facts_merged": len(result.archived_akus),
        "merged_into": len(result.consolidated_akus),
        "compression_ratio": round(result.compression_ratio, 3),
        "semantic_preservation": round(result.semantic_preservation, 3)
        if result.semantic_preservation == result.semantic_preservation else None,
    }
    print(f"  {consolidation}")

    print("Retrieving after consolidation...")
    rankings["ekm_hybrid_slept"] = await ekm_rankings("ekm_hybrid_slept")
    rankings["ekm_facts_slept"] = await ekm_rankings("ekm_facts_slept", mode="facts")
    rankings["ekm_flat_slept"] = await ekm_rankings("ekm_flat_slept", use_hierarchical_attention=False)

    # --- Score ----------------------------------------------------------------
    per_question = []
    summary: Dict[str, Dict[str, float]] = {}
    for system, ranked in rankings.items():
        scores = [score(r, q["supporting_facts"]["title"]) for r, q in zip(ranked, questions)]
        summary[system] = {m: round(statistics.mean(s[m] for s in scores), 3) for m in scores[0]}
        for q, r, s in zip(questions, ranked, scores):
            per_question.append({"system": system, "id": q["id"], "type": q["type"],
                                 "question": q["question"], "gold": q["supporting_facts"]["title"],
                                 "top5": r[:5], **s})

    by_type: Dict[str, Dict[str, float]] = {}
    for system in rankings:
        for qtype in ("bridge", "comparison"):
            rows = [p for p in per_question if p["system"] == system and p["type"] == qtype]
            if rows:
                by_type.setdefault(system, {})[qtype] = round(statistics.mean(p["recall@5"] for p in rows), 3)

    return {"setup": {"questions": len(questions), "paragraphs": len(corpus), "provider": provider_label,
                      "facts_extracted": sum(1 for a in storage.akus.values() if a.get("episode_id")),
                      "failed_paragraphs": failures[:20]},
            "summary": summary, "recall@5_by_type": by_type,
            "consolidation": consolidation, "timings": {k: round(v, 2) for k, v in timings.items()},
            "per_question": per_question}


def main():
    parser = argparse.ArgumentParser(description="EKM retrieval accuracy on HotpotQA")
    parser.add_argument("data", help="HotpotQA rows JSON (datasets-server format)")
    parser.add_argument("--questions", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8, help="paragraphs ingested in parallel")
    parser.add_argument("--offline", action="store_true", help="no API calls; checks the plumbing only")
    parser.add_argument("--out", help="write full results (incl. per-question) to this JSON file")
    args = parser.parse_args()

    results = asyncio.run(run(args))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== Retrieval accuracy: {results['setup']['questions']} HotpotQA questions, "
          f"{results['setup']['paragraphs']} paragraphs ===")
    print(f"{'system':<20}{'recall@2':>10}{'recall@5':>10}{'both@5':>9}{'MRR':>8}")
    for system, m in results["summary"].items():
        print(f"{system:<20}{m['recall@2']:>10.3f}{m['recall@5']:>10.3f}{m['both_gold@5']:>9.3f}{m['mrr']:>8.3f}")
    print("\nrecall@5 by question type:", json.dumps(results["recall@5_by_type"]))
    print("consolidation:", json.dumps(results["consolidation"]))
    print("timings:", json.dumps(results["timings"]))


if __name__ == "__main__":
    main()
