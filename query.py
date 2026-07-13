"""
Boolean, phrase, and ranked free-text query processing.

Boolean:  python AND (search OR index) NOT java
          - AND  = posting-list intersection (two-pointer, with skip pointers)
          - OR   = union (merge)
          - NOT  = difference vs the full doc set
Phrase:   positional intersection over adjacent terms
Ranked:   document-at-a-time (DAAT) scoring with a size-k min-heap for top-k
"""

from __future__ import annotations

import heapq
import re
from typing import List, Optional, Set, Tuple

from analyzer import analyze
from index import Index, PostingList, Posting


# ============================ posting-list algebra ===========================

def intersect(a: PostingList, b: PostingList) -> List[int]:
    """AND of two posting lists -> sorted doc_ids, using skip pointers."""
    pa = pb = 0
    A, B = a.postings, b.postings
    sa, sb = a.skips, b.skips
    out: List[int] = []
    while pa < len(A) and pb < len(B):
        da, db = A[pa].doc_id, B[pb].doc_id
        if da == db:
            out.append(da)
            pa += 1
            pb += 1
        elif da < db:
            # try to skip ahead in A
            if sa and pa < len(sa) and sa[pa] != -1 and A[sa[pa]].doc_id <= db:
                while sa[pa] != -1 and A[sa[pa]].doc_id <= db:
                    pa = sa[pa]
            else:
                pa += 1
        else:
            if sb and pb < len(sb) and sb[pb] != -1 and B[sb[pb]].doc_id <= da:
                while sb[pb] != -1 and B[sb[pb]].doc_id <= da:
                    pb = sb[pb]
            else:
                pb += 1
    return out


def union(a: PostingList, b: PostingList) -> List[int]:
    """OR of two posting lists -> sorted doc_ids."""
    pa = pb = 0
    A, B = a.postings, b.postings
    out: List[int] = []
    while pa < len(A) and pb < len(B):
        da, db = A[pa].doc_id, B[pb].doc_id
        if da == db:
            out.append(da); pa += 1; pb += 1
        elif da < db:
            out.append(da); pa += 1
        else:
            out.append(db); pb += 1
    out.extend(A[pa].doc_id for pa in range(pa, len(A)))
    out.extend(B[pb].doc_id for pb in range(pb, len(B)))
    return out


def difference(a: List[int], b: Set[int]) -> List[int]:
    """a NOT b: doc_ids in a but not in b."""
    return [d for d in a if d not in b]


# ============================ boolean parser =================================
# Grammar (precedence: NOT > AND > OR), parentheses supported:
#   expr   := orexpr
#   orexpr := andexpr (OR andexpr)*
#   andexpr:= notexpr (AND notexpr)*
#   notexpr:= NOT notexpr | atom
#   atom   := '(' expr ')' | TERM

_BOOL_TOKEN = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+")


class BooleanQuery:
    def __init__(self, index: Index, analyzer_kwargs: dict | None = None) -> None:
        self.index = index
        self.analyzer_kwargs = analyzer_kwargs or {}

    def _term_postings(self, raw: str) -> PostingList:
        terms = analyze(raw, **self.analyzer_kwargs)
        if not terms:
            return PostingList([])
        return self.index.get(terms[0][0])

    def search(self, query: str) -> List[int]:
        tokens = _BOOL_TOKEN.findall(query)
        self._toks = tokens
        self._i = 0
        result = self._parse_or()
        return sorted(result if isinstance(result, list) else list(result))

    def _peek(self) -> Optional[str]:
        return self._toks[self._i] if self._i < len(self._toks) else None

    def _next(self) -> str:
        t = self._toks[self._i]; self._i += 1; return t

    def _parse_or(self) -> List[int]:
        left = self._parse_and()
        while self._peek() == "OR":
            self._next()
            right = self._parse_and()
            left = union(_as_pl(left), _as_pl(right))
        return left

    def _parse_and(self) -> List[int]:
        # AND and infix NOT share this precedence level, left-associative.
        #   A AND B  -> intersection
        #   A NOT B  -> difference (A minus B), i.e. "A AND NOT B"
        left = self._parse_not()
        while self._peek() in ("AND", "NOT"):
            op = self._next()
            right = self._parse_not()
            if op == "AND":
                left = intersect(_as_pl(left), _as_pl(right))
            else:  # NOT as binary difference
                left = difference(left, set(right))
        return left

    def _parse_not(self) -> List[int]:
        # Leading (unary) NOT: complement against the full doc set.
        if self._peek() == "NOT":
            self._next()
            operand = self._parse_not()
            alld = set(self.index.doc_store.all_doc_ids())
            return sorted(alld - set(operand))
        return self._parse_atom()

    def _parse_atom(self) -> List[int]:
        tok = self._next()
        if tok == "(":
            inner = self._parse_or()
            if self._peek() == ")":
                self._next()
            return inner
        return self._term_postings(tok).doc_ids()


def _as_pl(doc_ids: List[int]) -> PostingList:
    """Wrap a bare doc_id list back into a PostingList for further algebra."""
    return PostingList([Posting(d, 0, []) for d in doc_ids])


# ============================ phrase queries =================================

def phrase_search(index: Index, phrase: str, analyzer_kwargs: dict | None = None) -> List[int]:
    """
    Positional intersection. For "machine learning": intersect on doc_id, then
    keep docs where some position p of 'machine' has p+1 a position of 'learning'.
    Generalises to k terms by chaining the offset check.
    """
    analyzer_kwargs = analyzer_kwargs or {}
    terms = [t for t, _ in analyze(phrase, **analyzer_kwargs)]
    if not terms:
        return []
    if len(terms) == 1:
        return index.get(terms[0]).doc_ids()

    plists = [index.get(t) for t in terms]
    if any(pl.df == 0 for pl in plists):
        return []

    # candidate docs = intersection of all term doc_id sets
    candidates = plists[0]
    cand_ids = candidates.doc_ids()
    for pl in plists[1:]:
        cand_ids = intersect(_as_pl(cand_ids), pl)
    cand_set = set(cand_ids)

    # position maps per term: doc_id -> positions
    pos_maps = []
    for pl in plists:
        pos_maps.append({p.doc_id: p.positions for p in pl.postings if p.doc_id in cand_set})

    out: List[int] = []
    for d in cand_ids:
        anchor = set(pos_maps[0][d])
        ok = False
        for p in anchor:
            if all((p + offset) in set(pos_maps[i][d]) for i, offset in enumerate(range(len(terms)))):
                ok = True
                break
        if ok:
            out.append(d)
    return out


# ============================ ranked retrieval (DAAT) ========================

def ranked_search(index: Index, ranker, query: str, k: int = 10,
                  analyzer_kwargs: dict | None = None) -> List[Tuple[int, float]]:
    """
    Document-at-a-time BM25/TF-IDF over the union of query-term posting lists.
    A size-k min-heap keeps the top-k in O(n log k) instead of sorting all n.
    Returns [(doc_id, score)] sorted by score desc.
    """
    analyzer_kwargs = analyzer_kwargs or {}
    terms = [t for t, _ in analyze(query, **analyzer_kwargs)]
    # dedupe but keep one posting list per distinct term
    seen = set()
    qterms = [t for t in terms if not (t in seen or seen.add(t))]

    cursors = []
    for t in qterms:
        pl = index.get(t)
        if pl.df:
            cursors.append((t, pl.postings, 0))
    if not cursors:
        return []

    heap: List[Tuple[float, int]] = []  # (score, doc_id) min-heap of size k

    # DAAT: repeatedly find the smallest current doc_id across cursors, score it.
    import heapq as _h
    # priority queue over (current_doc_id, cursor_index)
    pq: List[Tuple[int, int]] = []
    cursors = [list(c) for c in cursors]
    for ci, (t, postings, p) in enumerate(cursors):
        pq.append((postings[0].doc_id, ci))
    _h.heapify(pq)

    while pq:
        doc_id = pq[0][0]
        score = 0.0
        # consume all cursors currently at doc_id
        while pq and pq[0][0] == doc_id:
            _, ci = _h.heappop(pq)
            t, postings, p = cursors[ci]
            score += ranker.term_score(t, postings[p])
            p += 1
            cursors[ci][2] = p
            if p < len(postings):
                _h.heappush(pq, (postings[p].doc_id, ci))
        # push into top-k heap
        if len(heap) < k:
            _h.heappush(heap, (score, doc_id))
        elif score > heap[0][0]:
            _h.heapreplace(heap, (score, doc_id))

    return [(doc_id, score) for score, doc_id in sorted(heap, key=lambda x: (-x[0], x[1]))]
