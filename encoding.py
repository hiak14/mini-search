"""
Gap and variable-byte (VByte) integer compression for posting lists.

Doc ids and positions are sorted ascending, so delta-encoding makes the
numbers small. VByte then stores each number in as few bytes as possible:
7 payload bits per byte, with the high bit set on the final byte of a number.

Round-trip tested in tests/test_encoding.py.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple


# ---- variable-byte ----------------------------------------------------------

def vbyte_encode_number(n: int) -> bytes:
    """Encode a single non-negative int. Last byte has high bit set."""
    if n < 0:
        raise ValueError("VByte encodes non-negative integers only")
    out = bytearray()
    while True:
        out.insert(0, n & 0x7F)
        if n < 128:
            break
        n >>= 7
    out[-1] |= 0x80  # mark terminal byte
    return bytes(out)


def vbyte_encode(numbers: Iterable[int]) -> bytes:
    out = bytearray()
    for n in numbers:
        out += vbyte_encode_number(n)
    return bytes(out)


def vbyte_decode(data: bytes) -> List[int]:
    """Decode a whole buffer of concatenated VByte numbers."""
    numbers: List[int] = []
    n = 0
    for byte in data:
        if byte & 0x80:
            n = (n << 7) | (byte & 0x7F)
            numbers.append(n)
            n = 0
        else:
            n = (n << 7) | byte
    return numbers


def vbyte_decode_stream(data: bytes, pos: int, count: int) -> Tuple[List[int], int]:
    """
    Decode exactly `count` numbers starting at byte offset `pos`.
    Returns (numbers, new_pos). Lets us walk a packed buffer field by field.
    """
    numbers: List[int] = []
    n = 0
    while len(numbers) < count:
        byte = data[pos]
        pos += 1
        if byte & 0x80:
            n = (n << 7) | (byte & 0x7F)
            numbers.append(n)
            n = 0
        else:
            n = (n << 7) | byte
    return numbers, pos


# ---- gap (delta) encoding ---------------------------------------------------

def gap_encode(sorted_ints: List[int]) -> List[int]:
    """[5, 12, 13, 40] -> [5, 7, 1, 27]. Input must be ascending."""
    out: List[int] = []
    prev = 0
    for x in sorted_ints:
        out.append(x - prev)
        prev = x
    return out


def gap_decode(gaps: List[int]) -> List[int]:
    """[5, 7, 1, 27] -> [5, 12, 13, 40]."""
    out: List[int] = []
    acc = 0
    for g in gaps:
        acc += g
        out.append(acc)
    return out
