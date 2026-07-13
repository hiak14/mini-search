"""
Produces the numbers reported in the README.

Builds an index over the synthetic corpus, then measures:
  * index size on disk, compression ratio vs an uncompressed baseline,
    and index size as a % of the raw corpus
  * build time
  * query latency p50/p95 for ranked / boolean / phrase
  * retrieval quality (P@10, MAP, nDCG@10) for BM25 vs TF-IDF
  * a small k1/b tuning sweep
"""

from __future__ import annotations

import os
import pickle
import random
import shutil
import time

from corpus import make_synthetic_corpus, TOPICS
from indexer import build_index
from index import Index
from ranker import BM25Ranker, TfidfRanker
from query import BooleanQuery, phrase_search, ranked_search
from evaluate import evaluate_quality, benchmark_latency

ANALYZER_KW = dict(remove_stopwords=True, do_stem=True, do_fold_accents=True)
OUT = "bench_index"
N_DOCS = 30000


def dir_size(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))


def uncompressed_postings_size(idx: Index) -> int:
    """Baseline: 4 bytes each for doc_id, tf, n_positions, and every position."""
    total = 0
    for term in idx.dictionary:
        pl = idx.get(term)
        for p in pl.postings:
            total += 12 + 4 * len(p.positions)
    return total


def main():
    random.seed(0)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)

    print(f"Generating synthetic corpus: {N_DOCS} docs ...")
    docs, topic_of = make_synthetic_corpus(n_docs=N_DOCS)
    raw_bytes = sum(len(body.encode("utf-8")) for _, _, _, body in docs)

    print("Building index (SPIMI, gap+VByte) ...")
    t0 = time.time()
    meta = build_index(docs, OUT, memory_cap_postings=300_000,
                       analyzer_kwargs=ANALYZER_KW)
    build_s = time.time() - t0

    idx = Index(OUT)
    postings_size = os.path.getsize(os.path.join(OUT, "postings.bin"))
    dict_size = os.path.getsize(os.path.join(OUT, "dict.bin"))
    docstore_size = os.path.getsize(os.path.join(OUT, "docstore.json"))
    total_index = postings_size + dict_size
    uncompressed = uncompressed_postings_size(idx)

    print("\n=== CORPUS & INDEX ===")
    print(f"documents indexed     : {idx.n_docs}")
    print(f"unique terms (vocab)  : {idx.vocab_size()}")
    print(f"avg doc length        : {idx.avgdl:.1f} tokens")
    print(f"raw corpus size       : {raw_bytes/1e6:.2f} MB")
    print(f"postings.bin          : {postings_size/1e6:.2f} MB")
    print(f"dict.bin              : {dict_size/1e6:.2f} MB")
    print(f"index (postings+dict) : {total_index/1e6:.2f} MB")
    print(f"blocks flushed        : {meta['n_blocks']}  (forced SPIMI merge)")

    print("\n=== COMPRESSION ===")
    print(f"uncompressed postings : {uncompressed/1e6:.2f} MB (fixed 4-byte ints, no gaps)")
    print(f"compressed postings   : {postings_size/1e6:.2f} MB (gap + VByte)")
    print(f"compression ratio     : {postings_size/uncompressed*100:.1f}% of uncompressed "
          f"(saved {100-postings_size/uncompressed*100:.1f}%)")
    print(f"index as % of corpus  : {total_index/raw_bytes*100:.1f}%")

    print("\n=== BUILD ===")
    print(f"build wall-clock      : {build_s:.2f} s  ({idx.n_docs/build_s:.0f} docs/s)")

    # ---- quality: BM25 vs TF-IDF -------------------------------------------
    topic_docs = {}
    for d, t in topic_of.items():
        topic_docs.setdefault(t, set()).add(d)
    eval_queries = []
    for topic, vocab in TOPICS.items():
        q = " ".join(vocab[:3])  # 3-term query from the topic vocabulary
        eval_queries.append((q, topic_docs[topic]))

    bm = BM25Ranker(idx, k1=1.5, b=0.75)
    tf = TfidfRanker(idx)
    bm_fn = lambda q: [d for d, _ in ranked_search(idx, bm, q, k=10, analyzer_kwargs=ANALYZER_KW)]
    tf_fn = lambda q: [d for d, _ in ranked_search(idx, tf, q, k=10, analyzer_kwargs=ANALYZER_KW)]
    bm_q = evaluate_quality(eval_queries, bm_fn, k=10)
    tf_q = evaluate_quality(eval_queries, tf_fn, k=10)

    print("\n=== QUALITY (top-10, 12 topic queries) ===")
    print(f"{'ranker':8} {'P@10':>7} {'MAP':>7} {'nDCG@10':>9}")
    print(f"{'BM25':8} {bm_q['P@10']:7.3f} {bm_q['MAP']:7.3f} {bm_q['nDCG@10']:9.3f}")
    print(f"{'TF-IDF':8} {tf_q['P@10']:7.3f} {tf_q['MAP']:7.3f} {tf_q['nDCG@10']:9.3f}")

    # ---- latency ------------------------------------------------------------
    rng = random.Random(1)
    topics = list(TOPICS.items())
    ranked_qs = [" ".join(rng.sample(v, 3)) for _, v in topics for _ in range(30)]
    bool_qs = [f"{rng.choice(v)} AND {rng.choice(v)}" for _, v in topics for _ in range(30)]
    phrase_qs = [" ".join(rng.sample(v, 2)) for _, v in topics for _ in range(30)]

    bq = BooleanQuery(idx, ANALYZER_KW)
    lat_ranked = benchmark_latency(ranked_qs, lambda q: ranked_search(idx, bm, q, k=10, analyzer_kwargs=ANALYZER_KW))
    lat_bool = benchmark_latency(bool_qs, lambda q: bq.search(q))
    lat_phrase = benchmark_latency(phrase_qs, lambda q: phrase_search(idx, q, ANALYZER_KW))

    print("\n=== QUERY LATENCY (ms) ===")
    print(f"{'type':8} {'p50':>7} {'p95':>7} {'mean':>7}   (n)")
    for name, lat in [("ranked", lat_ranked), ("boolean", lat_bool), ("phrase", lat_phrase)]:
        print(f"{name:8} {lat['p50_ms']:7.2f} {lat['p95_ms']:7.2f} {lat['mean_ms']:7.2f}   ({lat['n']})")

    # ---- k1/b sweep ---------------------------------------------------------
    print("\n=== BM25 k1/b SWEEP (nDCG@10) ===")
    print(f"{'k1\\b':>6}" + "".join(f"{b:>8}" for b in (0.0, 0.5, 0.75, 1.0)))
    for k1 in (1.2, 1.5, 2.0):
        row = f"{k1:>6}"
        for b in (0.0, 0.5, 0.75, 1.0):
            r = BM25Ranker(idx, k1=k1, b=b)
            fn = lambda q, r=r: [d for d, _ in ranked_search(idx, r, q, k=10, analyzer_kwargs=ANALYZER_KW)]
            row += f"{evaluate_quality(eval_queries, fn, k=10)['nDCG@10']:8.3f}"
        print(row)


if __name__ == "__main__":
    main()
