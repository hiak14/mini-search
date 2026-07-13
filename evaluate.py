"""
Relevance metrics and latency benchmarking.

  Precision@k : of the top-k returned, the fraction that are relevant.
  AP          : average of Precision@i at each rank i where a relevant doc
                appears, divided by the number of relevant docs. MAP is the
                mean of AP over all queries.
  nDCG@k      : DCG@k / IDCG@k, where DCG = sum of rel_i / log2(i+1).
                Normalised against the ideal ordering.

For the synthetic corpus, a returned doc is relevant to a query iff it belongs
to the query's topic. For hand-labelled data, pass explicit qrels.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Sequence, Tuple


def precision_at_k(ranked: Sequence[int], relevant: set, k: int = 10) -> float:
    if k == 0:
        return 0.0
    topk = ranked[:k]
    if not topk:
        return 0.0
    return sum(1 for d in topk if d in relevant) / min(k, len(topk))


def average_precision(ranked: Sequence[int], relevant: set) -> float:
    if not relevant:
        return 0.0
    hits = 0
    ap = 0.0
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            hits += 1
            ap += hits / i
    return ap / len(relevant)


def ndcg_at_k(ranked: Sequence[int], relevant: set, k: int = 10) -> float:
    def dcg(items: Sequence[int]) -> float:
        return sum((1.0 if d in relevant else 0.0) / math.log2(i + 1)
                   for i, d in enumerate(items[:k], 1))
    actual = dcg(ranked)
    ideal_items = [1] * min(len(relevant), k)  # all-relevant ideal ordering
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, len(ideal_items) + 1))
    return (actual / idcg) if idcg > 0 else 0.0


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
    return s[idx]


def evaluate_quality(
    queries: List[Tuple[str, set]],
    search_fn: Callable[[str], List[int]],
    k: int = 10,
) -> Dict[str, float]:
    """queries: list of (query_string, relevant_doc_id_set)."""
    ps, aps, ndcgs = [], [], []
    for q, rel in queries:
        ranked = search_fn(q)
        ps.append(precision_at_k(ranked, rel, k))
        aps.append(average_precision(ranked, rel))
        ndcgs.append(ndcg_at_k(ranked, rel, k))
    n = len(queries)
    return {
        f"P@{k}": sum(ps) / n if n else 0.0,
        "MAP": sum(aps) / n if n else 0.0,
        f"nDCG@{k}": sum(ndcgs) / n if n else 0.0,
        "n_queries": n,
    }


def benchmark_latency(
    queries: List[str],
    search_fn: Callable[[str], object],
    warmup: int = 5,
) -> Dict[str, float]:
    """Run queries, return p50/p95/mean latency in milliseconds."""
    for q in queries[:warmup]:
        search_fn(q)
    times_ms: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        search_fn(q)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": percentile(times_ms, 50),
        "p95_ms": percentile(times_ms, 95),
        "mean_ms": sum(times_ms) / len(times_ms) if times_ms else 0.0,
        "n": len(times_ms),
    }
