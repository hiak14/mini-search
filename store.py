"""
Document metadata store.

The inverted index itself only holds integers. Titles, URLs, per-doc lengths,
and the corpus stats needed for BM25 live here, keyed by integer doc_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class DocMeta:
    title: str
    url: str
    length: int  # number of (post-analysis) tokens in the document body


class DocStore:
    def __init__(self) -> None:
        self.docs: Dict[int, DocMeta] = {}
        self._total_length = 0

    def add(self, doc_id: int, title: str, url: str, length: int) -> None:
        self.docs[doc_id] = DocMeta(title, url, length)
        self._total_length += length

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    @property
    def avgdl(self) -> float:
        return (self._total_length / self.n_docs) if self.docs else 0.0

    def length(self, doc_id: int) -> int:
        return self.docs[doc_id].length

    def meta(self, doc_id: int) -> DocMeta:
        return self.docs[doc_id]

    def all_doc_ids(self) -> List[int]:
        return list(self.docs.keys())

    # ---- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {
            "total_length": self._total_length,
            "docs": {str(k): asdict(v) for k, v in self.docs.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "DocStore":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        s = cls()
        s._total_length = payload["total_length"]
        s.docs = {
            int(k): DocMeta(**v) for k, v in payload["docs"].items()
        }
        return s
