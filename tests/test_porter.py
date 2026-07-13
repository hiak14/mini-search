"""
Validates the Porter stemmer against reference pairs drawn from Porter's
published voc.txt / output.txt fixtures. The pairs cover all five rule steps
(plurals, -ed/-ing, y->i, the step-4 suffix list, step-5 final-e and
double-consonant handling).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from porter import PorterStemmer

# (input, expected stem)
REFERENCE_PAIRS = [
    ("caresses", "caress"), ("ponies", "poni"), ("ties", "ti"),
    ("caress", "caress"), ("cats", "cat"), ("feed", "feed"),
    ("agreed", "agre"), ("plastered", "plaster"), ("bled", "bled"),
    ("motoring", "motor"), ("sing", "sing"), ("conflated", "conflat"),
    ("troubled", "troubl"), ("sized", "size"), ("hopping", "hop"),
    ("tanned", "tan"), ("falling", "fall"), ("hissing", "hiss"),
    ("fizzed", "fizz"), ("failing", "fail"), ("filing", "file"),
    ("happy", "happi"), ("sky", "sky"), ("relational", "relat"),
    ("conditional", "condit"), ("rational", "ration"), ("valenci", "valenc"),
    ("hesitanci", "hesit"), ("digitizer", "digit"), ("conformabli", "conform"),
    ("radicalli", "radic"), ("differentli", "differ"), ("vileli", "vile"),
    ("analogousli", "analog"), ("vietnamization", "vietnam"),
    ("predication", "predic"), ("operator", "oper"), ("feudalism", "feudal"),
    ("decisiveness", "decis"), ("hopefulness", "hope"), ("callousness", "callous"),
    ("formaliti", "formal"), ("sensitiviti", "sensit"), ("sensibiliti", "sensibl"),
    ("triplicate", "triplic"), ("formative", "form"), ("formalize", "formal"),
    ("electriciti", "electr"), ("electrical", "electr"), ("hopeful", "hope"),
    ("goodness", "good"), ("revival", "reviv"), ("allowance", "allow"),
    ("inference", "infer"), ("airliner", "airlin"), ("gyroscopic", "gyroscop"),
    ("adjustable", "adjust"), ("defensible", "defens"), ("irritant", "irrit"),
    ("replacement", "replac"), ("adjustment", "adjust"), ("dependent", "depend"),
    ("adoption", "adopt"), ("homologou", "homolog"), ("communism", "commun"),
    ("activate", "activ"), ("angularity", "angular"), ("homologous", "homolog"),
    ("effective", "effect"), ("bowdlerize", "bowdler"), ("probate", "probat"),
    ("rate", "rate"), ("cease", "ceas"), ("controll", "control"),
    ("roll", "roll"),
]


class TestPorterStemmer(unittest.TestCase):
    def setUp(self):
        self.stemmer = PorterStemmer()

    def test_reference_pairs(self):
        for word, expected in REFERENCE_PAIRS:
            with self.subTest(word=word):
                self.assertEqual(self.stemmer.stem(word), expected)

    def test_short_words_unchanged(self):
        for w in ("a", "be", "the", "of", "is"):
            self.assertEqual(self.stemmer.stem(w), w)


if __name__ == "__main__":
    unittest.main()
