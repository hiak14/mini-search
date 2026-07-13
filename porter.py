"""
Porter stemmer.

The Porter (1980) algorithm: a fixed sequence of suffix-stripping rules driven
by the "measure" m of a word, where m counts vowel-consonant transitions in
[C](VC){m}[V]. Tested against Porter's published vocabulary in
tests/test_porter.py.

Reference: M.F. Porter, "An algorithm for suffix stripping", Program 14(3), 1980.
"""

from __future__ import annotations


class PorterStemmer:
    def __init__(self) -> None:
        self.b = ""   # buffer holding the word
        self.k = 0    # index of last char of the part being considered
        self.j = 0    # general offset into the string

    # ---- primitive predicates ------------------------------------------------

    def _cons(self, i: int) -> bool:
        """True if b[i] is a consonant. 'y' is a consonant unless preceded by one."""
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return True if i == 0 else (not self._cons(i - 1))
        return True

    def _m(self) -> int:
        """measure of b[0..j]: number of VC sequences."""
        n = 0
        i = 0
        # skip leading consonants
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            # count each VC
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self) -> bool:
        """True if b[0..j] contains a vowel."""
        for i in range(self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, i: int) -> bool:
        """True if b[i-1..i] is a double consonant."""
        if i < 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i: int) -> bool:
        """
        True if b[i-2..i] is consonant-vowel-consonant and the final consonant
        is not w, x or y. Used to restore a final 'e' (e.g. -hop -> -hope).
        """
        if i < 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        return self.b[i] not in "wxy"

    def _ends(self, s: str) -> bool:
        """True if b[0..k] ends with s; sets j to just before the matched suffix."""
        length = len(s)
        if length > self.k + 1:
            return False
        if self.b[self.k - length + 1 : self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s: str) -> None:
        """Replace b[j+1..k] with s, adjusting k."""
        self.b = self.b[: self.j + 1] + s + self.b[self.k + 1 :]
        self.k = self.j + len(s)

    def _r(self, s: str) -> None:
        """setto(s) only if m(b[0..j]) > 0."""
        if self._m() > 0:
            self._setto(s)

    # ---- the five steps ------------------------------------------------------

    def _step1ab(self) -> None:
        # Step 1a: plurals and -ed/-ing prep
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        # Step 1b
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                if self.b[self.k] not in "lsz":
                    self.k -= 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step1c(self) -> None:
        """Turn terminal y into i when there's another vowel in the stem."""
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[: self.k] + "i" + self.b[self.k + 1 :]

    def _step2(self) -> None:
        ch = self.b[self.k - 1] if self.k >= 1 else ""
        rules = {
            "a": [("ational", "ate"), ("tional", "tion")],
            "c": [("enci", "ence"), ("anci", "ance")],
            "e": [("izer", "ize")],
            "l": [("bli", "ble"), ("alli", "al"), ("entli", "ent"),
                  ("eli", "e"), ("ousli", "ous")],
            "o": [("ization", "ize"), ("ation", "ate"), ("ator", "ate")],
            "s": [("alism", "al"), ("iveness", "ive"),
                  ("fulness", "ful"), ("ousness", "ous")],
            "t": [("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")],
            "g": [("logi", "log")],
        }
        for suffix, repl in rules.get(ch, []):
            if self._ends(suffix):
                self._r(repl)
                return

    def _step3(self) -> None:
        ch = self.b[self.k] if self.k >= 0 else ""
        rules = {
            "e": [("icate", "ic"), ("ative", ""), ("alize", "al")],
            "i": [("iciti", "ic")],
            "l": [("ical", "ic"), ("ful", "")],
            "s": [("ness", "")],
        }
        for suffix, repl in rules.get(ch, []):
            if self._ends(suffix):
                self._r(repl)
                return

    def _step4(self) -> None:
        ch = self.b[self.k - 1] if self.k >= 1 else ""
        suffixes = {
            "a": ["al"],
            "c": ["ance", "ence"],
            "e": ["er"],
            "i": ["ic"],
            "l": ["able", "ible"],
            "n": ["ant", "ement", "ment", "ent"],
            "o": ["ion", "ou"],
            "s": ["ism"],
            "t": ["ate", "iti"],
            "u": ["ous"],
            "v": ["ive"],
            "z": ["ize"],
        }
        for suffix in suffixes.get(ch, []):
            if self._ends(suffix):
                if suffix == "ion":
                    # -ion only drops after s or t
                    if self.j >= 0 and self.b[self.j] in "st" and self._m() > 1:
                        self.k = self.j
                    return
                if self._m() > 1:
                    self.k = self.j
                return

    def _step5(self) -> None:
        # Step 5a: remove a final -e
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        # Step 5b: -ll -> -l when m > 1
        if self.b[self.k] == "l" and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    # ---- public API ----------------------------------------------------------

    def stem(self, word: str) -> str:
        # Words of length <= 2 are left alone (Porter's original convention).
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[: self.k + 1]


def stem(word: str) -> str:
    return PorterStemmer().stem(word)
