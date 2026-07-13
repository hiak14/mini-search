"""
Scoring functions.

  TfidfRanker : log-tf weighting + idf, with document-length normalisation.
  BM25Ranker  : saturating tf and tunable length normalisation via (k1, b).

Both consume an Index and score per (term, posting), so they plug into the
document-at-a-time retrieval loop in query.py.
"""

from __future__ import annotations

import math
from typing import Dict

from index import Index, Posting


class BM25Ranker:
    """
    score(D,Q) = sum of_t idf(t) * f(t,D)(k1+1) / ( f(t,D) + k1(1 - b + b|D|/avgdl) )
    idf(t)     = log( (N - df + 0.5) / (df + 0.5) + 1 )   [BM25 idf variant]
    """

    def __init__(self, index: Index, k1: float = 1.5, b: float = 0.75) -> None:
        self.index = index
        self.k1 = k1
        self.b = b
        self.N = index.n_docs
        self.avgdl = index.avgdl or 1.0
        self._idf_cache: Dict[str, float] = {}

    def idf(self, term: str) -> float:
        if term not in self._idf_cache:
            df = self.index.df(term)
            self._idf_cache[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        return self._idf_cache[term]

    def term_score(self, term: str, posting: Posting) -> float:
        f = posting.tf
        dl = self.index.doc_store.length(posting.doc_id)
        denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        return self.idf(term) * (f * (self.k1 + 1)) / denom

    def max_term_score(self, term: str) -> float:
        """Upper bound on a term's contribution, for WAND-style pruning."""
        # tf -> inf saturates the tf factor to (k1+1); shortest doc maximises it.
        return self.idf(term) * (self.k1 + 1)


class TfidfRanker:
    """
    log-tf:   w(t,D) = 1 + log(f(t,D))
    idf:      log(N / df(t))
    score:    cosine similarity of query & doc tf-idf vectors, with the doc
              vector L2-normalised so long docs don't dominate.
    """

    def __init__(self, index: Index) -> None:
        self.index = index
        self.N = index.n_docs
        self._idf_cache: Dict[str, float] = {}

    def idf(self, term: str) -> float:
        if term not in self._idf_cache:
            df = self.index.df(term) or 1
            self._idf_cache[term] = math.log(self.N / df)
        return self._idf_cache[term]

    def term_score(self, term: str, posting: Posting) -> float:
        tf_w = 1.0 + math.log(posting.tf) if posting.tf > 0 else 0.0
        dl = self.index.doc_store.length(posting.doc_id) or 1
        # length normalisation: divide by sqrt(doc length)
        return (tf_w * self.idf(term)) / math.sqrt(dl)

    def max_term_score(self, term: str) -> float:
        return self.idf(term) * 10.0  # loose bound
