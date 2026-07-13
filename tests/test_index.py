"""
Builds an index over a tiny hand-written corpus whose exact postings are known,
then asserts them.

memory_cap_postings=2 forces the SPIMI builder to flush several blocks and
perform a real k-way merge, which is the code path that corrupts postings if
the merge or gap re-encoding is wrong.

Analysis runs with stemming/stopwords/accent-folding off so each token maps
1:1 to a term and the expected postings are unambiguous.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexer import build_index
from index import Index

RAW = dict(remove_stopwords=False, do_stem=False, do_fold_accents=False)

CORPUS = [
    (0, "t0", "u0", "alpha beta alpha"),
    (1, "t1", "u1", "beta gamma"),
    (2, "t2", "u2", "alpha gamma gamma"),
]

# (doc_id, tf, positions) per term, in doc_id order.
EXPECTED = {
    "alpha": [(0, 2, [0, 2]), (2, 1, [0])],
    "beta":  [(0, 1, [1]),    (1, 1, [0])],
    "gamma": [(1, 1, [1]),    (2, 2, [1, 2])],
}


class TestIndexCorrectness(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="idxtest_")
        # cap of 2 postings/block forces multiple blocks + k-way merge
        self.meta = build_index(CORPUS, self.dir, memory_cap_postings=2,
                                analyzer_kwargs=RAW)
        self.idx = Index(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_multiple_blocks_merged(self):
        # The build must not have fit in a single in-memory block.
        self.assertGreater(self.meta["n_blocks"], 1)

    def test_corpus_stats(self):
        self.assertEqual(self.idx.n_docs, 3)
        self.assertEqual(self.idx.vocab_size(), 3)
        # lengths 3, 2, 3 -> avg 8/3
        self.assertAlmostEqual(self.idx.avgdl, 8 / 3, places=6)

    def test_exact_postings(self):
        for term, expected in EXPECTED.items():
            pl = self.idx.get(term)
            got = [(p.doc_id, p.tf, list(p.positions)) for p in pl.postings]
            self.assertEqual(got, expected, f"postings mismatch for '{term}'")

    def test_document_frequencies(self):
        for term in EXPECTED:
            self.assertEqual(self.idx.df(term), 2)

    def test_missing_term(self):
        self.assertEqual(self.idx.df("nonexistent"), 0)
        self.assertEqual(self.idx.get("nonexistent").postings, [])


if __name__ == "__main__":
    unittest.main()
