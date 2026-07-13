"""
Read side of the inverted index.

Loads the dictionary and compressed postings produced by indexer.py and
decodes posting lists on demand. A decoded posting is:

    Posting(doc_id, tf, positions)

Posting lists come back sorted by doc_id, which is what makes intersection
and union linear. Long lists get skip pointers so AND queries can jump over
non-matching stretches.
"""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from encoding import gap_decode, vbyte_decode_stream
from store import DocStore


@dataclass
class Posting:
    doc_id: int
    tf: int
    positions: List[int]


class PostingList:
    """A decoded posting list with optional skip pointers."""

    def __init__(self, postings: List[Posting]) -> None:
        self.postings = postings
        self.skips: List[int] = []  # skip[i] = index to jump to from i
        if len(postings) >= 16:
            self._build_skips()

    def _build_skips(self) -> None:
        n = len(self.postings)
        step = max(1, int(math.sqrt(n)))
        self.skips = [-1] * n
        i = 0
        while i + step < n:
            self.skips[i] = i + step
            i += step

    @property
    def df(self) -> int:
        return len(self.postings)

    def doc_ids(self) -> List[int]:
        return [p.doc_id for p in self.postings]


class Index:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        with open(os.path.join(out_dir, "dict.bin"), "rb") as f:
            # term -> (df, offset, byte_length)
            self.dictionary: Dict[str, Tuple[int, int, int]] = pickle.load(f)
        with open(os.path.join(out_dir, "postings.bin"), "rb") as f:
            self.postings_data = f.read()
        self.doc_store = DocStore.load(os.path.join(out_dir, "docstore.json"))
        self._cache: Dict[str, PostingList] = {}

    # ---- stats --------------------------------------------------------------

    @property
    def n_docs(self) -> int:
        return self.doc_store.n_docs

    @property
    def avgdl(self) -> float:
        return self.doc_store.avgdl

    def df(self, term: str) -> int:
        entry = self.dictionary.get(term)
        return entry[0] if entry else 0

    # ---- posting access -----------------------------------------------------

    def _decode_term(self, term: str) -> PostingList:
        entry = self.dictionary.get(term)
        if entry is None:
            return PostingList([])
        df, offset, length = entry
        pos = offset
        (df_check,), pos = vbyte_decode_stream(self.postings_data, pos, 1)
        postings: List[Posting] = []
        prev_doc = 0
        for _ in range(df_check):
            (dgap, tf, npos), pos = vbyte_decode_stream(self.postings_data, pos, 3)
            prev_doc += dgap
            posgaps, pos = vbyte_decode_stream(self.postings_data, pos, npos)
            positions = gap_decode(posgaps)
            postings.append(Posting(prev_doc, tf, positions))
        return PostingList(postings)

    def get(self, term: str) -> PostingList:
        """Return the (cached) decoded posting list for a term."""
        pl = self._cache.get(term)
        if pl is None:
            pl = self._decode_term(term)
            self._cache[term] = pl
        return pl

    def vocab_size(self) -> int:
        return len(self.dictionary)
