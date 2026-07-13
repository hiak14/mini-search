"""
Property tests for the compression primitives: decode(encode(x)) == x for
arbitrary inputs, over thousands of random integer lists plus the values
around each 7-bit byte boundary (127/128, 16383/16384, ...) where the VByte
continuation bit flips.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoding import (
    gap_decode,
    gap_encode,
    vbyte_decode,
    vbyte_decode_stream,
    vbyte_encode,
    vbyte_encode_number,
)


class TestVByte(unittest.TestCase):
    def test_single_number_roundtrip(self):
        for n in [0, 1, 127, 128, 129, 16383, 16384, 2097151, 2097152, 10**9]:
            with self.subTest(n=n):
                self.assertEqual(vbyte_decode(vbyte_encode_number(n)), [n])

    def test_boundary_values(self):
        # Values straddling each 7-bit byte boundary.
        boundaries = []
        for shift in (7, 14, 21, 28):
            base = 1 << shift
            boundaries += [base - 1, base, base + 1]
        self.assertEqual(vbyte_decode(vbyte_encode(boundaries)), boundaries)

    def test_random_lists_roundtrip(self):
        rng = random.Random(42)
        for _ in range(2000):
            nums = [rng.randint(0, 5_000_000) for _ in range(rng.randint(0, 40))]
            self.assertEqual(vbyte_decode(vbyte_encode(nums)), nums)

    def test_stream_decode_partial(self):
        # vbyte_decode_stream must read exactly `count` numbers and report the
        # new byte offset, so several lists can be packed back-to-back.
        a = [3, 130, 9001]
        b = [0, 1, 2, 7000000]
        blob = vbyte_encode(a) + vbyte_encode(b)
        got_a, pos = vbyte_decode_stream(blob, 0, len(a))
        got_b, pos2 = vbyte_decode_stream(blob, pos, len(b))
        self.assertEqual(got_a, a)
        self.assertEqual(got_b, b)
        self.assertEqual(pos2, len(blob))

    def test_empty(self):
        self.assertEqual(vbyte_encode([]), b"")
        self.assertEqual(vbyte_decode(b""), [])


class TestGap(unittest.TestCase):
    def test_gap_roundtrip(self):
        rng = random.Random(7)
        for _ in range(2000):
            n = rng.randint(0, 50)
            vals = sorted(rng.sample(range(0, 200000), min(n, 200000)))
            self.assertEqual(gap_decode(gap_encode(vals)), vals)

    def test_gaps_are_smaller(self):
        ids = [5, 12, 13, 40, 999]
        gaps = gap_encode(ids)
        self.assertEqual(gaps, [5, 7, 1, 27, 959])
        self.assertEqual(gap_decode(gaps), ids)

    def test_gap_then_vbyte_pipeline(self):
        # The real index path: sorted doc_ids -> gaps -> VByte -> back.
        rng = random.Random(11)
        for _ in range(500):
            ids = sorted(set(rng.randint(0, 1_000_000) for _ in range(rng.randint(1, 60))))
            restored = gap_decode(vbyte_decode(vbyte_encode(gap_encode(ids))))
            self.assertEqual(restored, ids)


if __name__ == "__main__":
    unittest.main()
