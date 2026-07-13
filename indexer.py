"""
SPIMI (Single-Pass In-Memory Indexing) builder.

A single in-memory dict works for small collections but not once the full
postings map outgrows RAM. SPIMI instead:

  1. Streams documents, accumulating postings in a dict up to a memory cap.
  2. Flushes a sorted block to disk and starts a fresh dict.
  3. After all docs, k-way merges the sorted blocks (min-heap over block
     iterators) into one final index. A term's postings from different blocks
     arrive in doc_id order because blocks are processed in creation order and
     each block holds a disjoint, ascending doc_id range.

Final on-disk index (in <out_dir>):
  postings.bin   concatenated per-term posting blocks, gap+VByte encoded
  dict.bin       term -> (df, offset, byte_length) into postings.bin
  docstore.json  doc_id -> (title, url, length) + corpus stats
  meta.json      build parameters and stats

Postings.bin layout, per term (all VByte):
  df, then for each of df postings:
     gap(doc_id), tf, n_positions, gap(pos_1), gap(pos_2-pos_1), ...
"""

from __future__ import annotations

import heapq
import json
import os
import pickle
import shutil
import tempfile
import time
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Tuple

from analyzer import analyze
from encoding import gap_encode, vbyte_encode
from store import DocStore

# A "posting" while building: doc_id -> list of positions.
# term -> {doc_id: [positions]}
Block = Dict[str, Dict[int, List[int]]]

Document = Tuple[int, str, str, str]  # (doc_id, title, url, body)


def _encode_term_postings(postings: List[Tuple[int, List[int]]]) -> bytes:
    """
    postings: list of (doc_id, sorted_positions), ascending by doc_id.
    Returns the gap+VByte encoded byte block for one term.
    """
    nums: List[int] = [len(postings)]  # df
    doc_gaps = gap_encode([d for d, _ in postings])
    for (doc_id, positions), dgap in zip(postings, doc_gaps):
        tf = len(positions)
        nums.append(dgap)
        nums.append(tf)
        nums.append(len(positions))
        nums.extend(gap_encode(sorted(positions)))
    return vbyte_encode(nums)


class SpimiIndexer:
    def __init__(self, out_dir: str, memory_cap_postings: int = 1_000_000,
                 analyzer_kwargs: dict | None = None) -> None:
        """
        memory_cap_postings: flush a block once this many (term,doc) postings
        have accumulated. Tune for your RAM; small values exercise the merge.
        """
        self.out_dir = out_dir
        self.memory_cap = memory_cap_postings
        self.analyzer_kwargs = analyzer_kwargs or {}
        self.doc_store = DocStore()
        self._block_paths: List[str] = []
        self._tmp_dir = tempfile.mkdtemp(prefix="spimi_blocks_")
        self._postings_in_mem = 0
        self._block: Block = defaultdict(lambda: defaultdict(list))
        self._n_indexed = 0

    # ---- block accumulation -------------------------------------------------

    def add_document(self, doc_id: int, title: str, url: str, body: str) -> None:
        terms = analyze(body, **self.analyzer_kwargs)
        doc_len = len(terms)
        self.doc_store.add(doc_id, title, url, doc_len)

        seen_pairs = 0
        local = self._block
        for term, pos in terms:
            plist = local[term][doc_id]
            if not plist:
                seen_pairs += 1
            plist.append(pos)
        self._postings_in_mem += seen_pairs
        self._n_indexed += 1

        if self._postings_in_mem >= self.memory_cap:
            self._flush_block()

    def _flush_block(self) -> None:
        if not self._block:
            return
        path = os.path.join(self._tmp_dir, f"block_{len(self._block_paths):05d}.pkl")
        # Sort terms; within a term, postings already keyed by ascending doc_id
        # because we stream doc_ids in increasing order.
        with open(path, "wb") as f:
            for term in sorted(self._block.keys()):
                postings = [(d, self._block[term][d]) for d in sorted(self._block[term])]
                pickle.dump((term, postings), f)
        self._block_paths.append(path)
        self._block = defaultdict(lambda: defaultdict(list))
        self._postings_in_mem = 0

    # ---- k-way merge --------------------------------------------------------

    def _iter_block(self, path: str) -> Iterator[Tuple[str, List[Tuple[int, List[int]]]]]:
        with open(path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    return

    def _merge(self) -> Tuple[bytes, Dict[str, Tuple[int, int, int]]]:
        """
        k-way merge of sorted blocks. Returns (postings_bytes, dictionary).
        dictionary: term -> (df, offset, byte_length).
        """
        iterators = [self._iter_block(p) for p in self._block_paths]
        heap: List[Tuple[str, int, List[Tuple[int, List[int]]]]] = []
        for i, it in enumerate(iterators):
            try:
                term, postings = next(it)
                heap.append((term, i, postings))
            except StopIteration:
                pass
        heapq.heapify(heap)

        postings_buf = bytearray()
        dictionary: Dict[str, Tuple[int, int, int]] = {}

        while heap:
            term = heap[0][0]
            # Gather every block-fragment for this term (k-way merge step).
            merged: List[Tuple[int, List[int]]] = []
            while heap and heap[0][0] == term:
                _, i, postings = heapq.heappop(heap)
                merged.extend(postings)  # disjoint, ascending doc_id ranges
                try:
                    nterm, npostings = next(iterators[i])
                    heapq.heappush(heap, (nterm, i, npostings))
                except StopIteration:
                    pass
            merged.sort(key=lambda x: x[0])  # safety: ensure doc_id order
            encoded = _encode_term_postings(merged)
            offset = len(postings_buf)
            postings_buf += encoded
            dictionary[term] = (len(merged), offset, len(encoded))

        return bytes(postings_buf), dictionary

    # ---- finalise -----------------------------------------------------------

    def finalize(self) -> dict:
        t0 = time.time()
        self._flush_block()
        postings_bytes, dictionary = self._merge()

        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "postings.bin"), "wb") as f:
            f.write(postings_bytes)
        with open(os.path.join(self.out_dir, "dict.bin"), "wb") as f:
            pickle.dump(dictionary, f)
        self.doc_store.save(os.path.join(self.out_dir, "docstore.json"))

        meta = {
            "n_docs": self.doc_store.n_docs,
            "n_terms": len(dictionary),
            "avgdl": self.doc_store.avgdl,
            "postings_bytes": len(postings_bytes),
            "n_blocks": len(self._block_paths),
            "analyzer_kwargs": self.analyzer_kwargs,
            "build_seconds": round(time.time() - t0, 3),
        }
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        shutil.rmtree(self._tmp_dir, ignore_errors=True)
        return meta


def build_index(documents: Iterable[Document], out_dir: str,
                memory_cap_postings: int = 1_000_000,
                analyzer_kwargs: dict | None = None,
                progress_every: int = 0) -> dict:
    """Convenience driver: stream documents into a SpimiIndexer and finalise."""
    idx = SpimiIndexer(out_dir, memory_cap_postings, analyzer_kwargs)
    for n, (doc_id, title, url, body) in enumerate(documents, 1):
        idx.add_document(doc_id, title, url, body)
        if progress_every and n % progress_every == 0:
            print(f"  indexed {n} docs ({len(idx._block_paths)} blocks flushed)")
    return idx.finalize()
