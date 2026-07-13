"""
Command-line interface.

    # build an index from synthetic data (no download needed)
    python cli.py build --synthetic 20000 --out index_dir

    # build from Simple English Wikipedia (wikiextractor JSONL)
    python cli.py build --wiki wiki.jsonl --out index_dir

    # query
    python cli.py search "machine learning" --index index_dir
    python cli.py search --bool "python AND (search OR index) NOT java" --index index_dir
    python cli.py search --phrase "machine learning" --index index_dir
"""

from __future__ import annotations

import argparse
import sys

from corpus import load_wikipedia_jsonl, make_synthetic_corpus
from indexer import build_index
from index import Index
from ranker import BM25Ranker, TfidfRanker
from query import BooleanQuery, phrase_search, ranked_search

ANALYZER_KW = dict(remove_stopwords=True, do_stem=True, do_fold_accents=True)


def cmd_build(args: argparse.Namespace) -> None:
    if args.wiki:
        docs = load_wikipedia_jsonl(args.wiki, limit=args.limit)
        print(f"Building index from Wikipedia JSONL: {args.wiki}")
    else:
        n = args.synthetic or 20000
        docs, _ = make_synthetic_corpus(n_docs=n)
        print(f"Building index from synthetic corpus: {n} docs")
    meta = build_index(docs, args.out, memory_cap_postings=args.mem_cap,
                       analyzer_kwargs=ANALYZER_KW, progress_every=args.progress)
    print("Done. Index meta:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


def cmd_search(args: argparse.Namespace) -> None:
    idx = Index(args.index)
    if args.bool:
        ids = BooleanQuery(idx, ANALYZER_KW).search(args.query)
        _print_docs(idx, ids[: args.k])
    elif args.phrase:
        ids = phrase_search(idx, args.query, ANALYZER_KW)
        _print_docs(idx, ids[: args.k])
    else:
        ranker = TfidfRanker(idx) if args.tfidf else BM25Ranker(idx, k1=args.k1, b=args.b)
        results = ranked_search(idx, ranker, args.query, k=args.k,
                                analyzer_kwargs=ANALYZER_KW)
        for rank, (doc_id, score) in enumerate(results, 1):
            m = idx.doc_store.meta(doc_id)
            print(f"{rank:2d}. [{score:7.4f}] {m.title}  ({m.url})")


def _print_docs(idx: Index, ids) -> None:
    if not ids:
        print("(no matches)")
        return
    for rank, doc_id in enumerate(ids, 1):
        m = idx.doc_store.meta(doc_id)
        print(f"{rank:2d}. {m.title}  ({m.url})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="mini-search")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build an index")
    b.add_argument("--out", required=True)
    b.add_argument("--wiki", help="path to wikiextractor JSONL")
    b.add_argument("--synthetic", type=int, help="generate N synthetic docs")
    b.add_argument("--limit", type=int, default=None, help="cap wiki docs")
    b.add_argument("--mem-cap", type=int, default=1_000_000,
                   help="postings per SPIMI block before flush")
    b.add_argument("--progress", type=int, default=0)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("search", help="query an index")
    s.add_argument("query")
    s.add_argument("--index", required=True)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--bool", action="store_true", help="boolean query")
    s.add_argument("--phrase", action="store_true", help="phrase query")
    s.add_argument("--tfidf", action="store_true", help="use TF-IDF not BM25")
    s.add_argument("--k1", type=float, default=1.5)
    s.add_argument("--b", type=float, default=0.75)
    s.set_defaults(func=cmd_search)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
