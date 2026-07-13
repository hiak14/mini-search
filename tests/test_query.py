"""
Query layer tests.

1. The skip-pointer intersect()/union() must agree with naive set
   intersection/union on thousands of random inputs.
2. Boolean parser: precedence (NOT > AND > OR), parentheses, and the infix
   difference semantics of NOT, on a small corpus with known answers.
3. Phrase search: positional adjacency, including a doc that contains both
   terms but not adjacently (must not match).
"""

import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexer import build_index
from index import Index, Posting, PostingList
from query import BooleanQuery, intersect, union, phrase_search

RAW = dict(remove_stopwords=False, do_stem=False, do_fold_accents=False)


def make_pl(doc_ids):
    return PostingList([Posting(d, 1, []) for d in sorted(set(doc_ids))])


class TestPostingAlgebra(unittest.TestCase):
    def test_intersect_matches_set(self):
        rng = random.Random(1)
        for _ in range(3000):
            a = rng.sample(range(0, 300), rng.randint(0, 60))
            b = rng.sample(range(0, 300), rng.randint(0, 60))
            got = intersect(make_pl(a), make_pl(b))
            self.assertEqual(got, sorted(set(a) & set(b)))

    def test_union_matches_set(self):
        rng = random.Random(2)
        for _ in range(3000):
            a = rng.sample(range(0, 300), rng.randint(0, 60))
            b = rng.sample(range(0, 300), rng.randint(0, 60))
            got = union(make_pl(a), make_pl(b))
            self.assertEqual(got, sorted(set(a) | set(b)))

    def test_intersect_with_skips(self):
        # Lists long enough (>=16) to actually build and traverse skip pointers.
        a = list(range(0, 1000, 2))    # evens
        b = list(range(0, 1000, 3))    # multiples of 3
        got = intersect(make_pl(a), make_pl(b))
        self.assertEqual(got, sorted(set(a) & set(b)))  # multiples of 6


class TestBooleanAndPhrase(unittest.TestCase):
    CORPUS = [
        (0, "t0", "u0", "python search engine"),
        (1, "t1", "u1", "java index structure"),
        (2, "t2", "u2", "python java hybrid"),
        (3, "t3", "u3", "search index ranking"),
        (4, "t4", "u4", "python search index fast"),
        (5, "t5", "u5", "python java search index"),
    ]

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="qtest_")
        build_index(cls.CORPUS, cls.dir, memory_cap_postings=3, analyzer_kwargs=RAW)
        cls.idx = Index(cls.dir)
        cls.bq = BooleanQuery(cls.idx, RAW)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_and(self):
        self.assertEqual(self.bq.search("python AND java"), [2, 5])

    def test_or(self):
        self.assertEqual(self.bq.search("search OR java"), [0, 1, 2, 3, 4, 5])

    def test_not_infix(self):
        self.assertEqual(self.bq.search("python NOT java"), [0, 4])

    def test_precedence_and_parens(self):
        # python AND (search OR index) NOT java
        #   search OR index = {0,1,3,4,5}; AND python({0,2,4,5}) = {0,4,5};
        #   NOT java({1,2,5}) = {0,4}
        self.assertEqual(
            self.bq.search("python AND (search OR index) NOT java"), [0, 4]
        )

    def test_phrase_adjacent(self):
        # "python search": adjacent only in docs 0 and 4 (in doc 5 a 'java'
        # sits between them, so it must be excluded).
        self.assertEqual(phrase_search(self.idx, "python search", RAW), [0, 4])

    def test_phrase_two_term(self):
        self.assertEqual(phrase_search(self.idx, "search index", RAW), [3, 4, 5])

    def test_phrase_no_match(self):
        self.assertEqual(phrase_search(self.idx, "engine python", RAW), [])


if __name__ == "__main__":
    unittest.main()
