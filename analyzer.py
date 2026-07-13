"""
Text analysis: turns raw text into index terms.

The same analyzer must run at index time and at query time. If a term is
processed differently on the two paths it can never match, so everything that
touches text goes through analyze().

Pipeline:  lowercase -> tokenise -> (optional accent fold) -> stopword drop
           -> Porter stem -> (term, position) pairs

Positions are 0-indexed token offsets in the original token stream, before
stopword removal is applied to the emitted list. Positions are kept even for
dropped tokens so that phrase queries stay correct.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

from porter import PorterStemmer

# A compact, standard English stop list (~180 words). Kept here so the project
# has no data dependencies. Toggle via analyze(..., remove_stopwords=False).
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves
""".split())

# Digits are kept ("bm25", "covid19"); apostrophes and hyphens act as token
# boundaries, so "don't" -> ["don", "t"].
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_stemmer = PorterStemmer()


def fold_accents(text: str) -> str:
    """café -> cafe. NFKD decompose, drop combining marks."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def analyze(
    text: str,
    remove_stopwords: bool = True,
    do_stem: bool = True,
    do_fold_accents: bool = True,
) -> List[Tuple[str, int]]:
    """
    Return a list of (term, position) pairs.

    `position` is the index of the token in the raw token stream, so positions
    are stable regardless of whether stopwords are removed. This is what makes
    phrase matching robust: two surviving terms keep their true distance.
    """
    text = text.lower()
    if do_fold_accents:
        text = fold_accents(text)

    out: List[Tuple[str, int]] = []
    for pos, tok in enumerate(_TOKEN_RE.findall(text)):
        if remove_stopwords and tok in STOPWORDS:
            continue
        term = _stemmer.stem(tok) if do_stem else tok
        if term:
            out.append((term, pos))
    return out


def analyze_terms(text: str, **kw) -> List[str]:
    """Convenience: just the terms, dropping positions."""
    return [t for t, _ in analyze(text, **kw)]
