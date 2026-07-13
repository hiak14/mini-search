"""
Document sources. Both produce (doc_id, title, url, body) tuples.

  * load_wikipedia_jsonl(path): reads the JSON-lines that wikiextractor
    produces from simplewiki-latest-pages-articles.xml.bz2, one
    {"id","url","title","text"} object per line.
  * make_synthetic_corpus(...): a topic-structured generator so the repo runs
    and benchmarks without the multi-GB Wikipedia download. Documents mix
    English topic vocabularies with a Zipfian background, which gives
    meaningful ranking behaviour and well-defined relevance labels.

To benchmark on real data:
    python cli.py build --wiki path/to/wiki.jsonl --out index_dir
"""

from __future__ import annotations

import itertools
import json
import random
from typing import Dict, Iterator, List, Tuple

Document = Tuple[int, str, str, str]


def load_wikipedia_jsonl(path: str, limit: int | None = None) -> Iterator[Document]:
    """Stream (doc_id, title, url, body) from wikiextractor JSONL output."""
    doc_id = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text.strip():
                continue
            yield (doc_id, obj.get("title", ""), obj.get("url", ""), text)
            doc_id += 1
            if limit and doc_id >= limit:
                return


# --------------------------------------------------------------------------- #
#  Synthetic corpus
# --------------------------------------------------------------------------- #

# Each topic is a real-English vocabulary. A document about a topic samples
# heavily from its own vocab, lightly from neighbours (noise), and from a shared
# function-word background. Relevance for evaluation = same-topic membership.
TOPICS: Dict[str, List[str]] = {
    "machine_learning": """machine learning model training data neural network deep
        gradient descent algorithm classifier regression supervised feature vector
        embedding tensor inference accuracy loss optimisation backpropagation dataset
        overfitting bayesian clustering reinforcement transformer attention""".split(),
    "astronomy": """star galaxy planet orbit telescope nebula cosmic universe solar
        lunar eclipse comet asteroid gravity supernova black hole radiation spectrum
        constellation astronomer observatory light year redshift cosmology""".split(),
    "cooking": """recipe ingredient flour butter sugar oven bake roast simmer sauce
        garlic onion pepper salt dough knead whisk marinade flavour cuisine dish
        saute boil grill seasoning vegetable""".split(),
    "music": """melody harmony rhythm chord scale tempo orchestra symphony composer
        instrument guitar piano violin note octave key signature concert song
        recording album genre jazz classical""".split(),
    "medicine": """patient disease diagnosis treatment symptom infection vaccine
        antibiotic surgery therapy clinical immune virus bacteria dosage prescription
        cardiac neurological chronic acute recovery hospital physician""".split(),
    "finance": """market stock investment portfolio dividend interest rate bond equity
        currency inflation revenue profit asset liability trading exchange capital
        risk return hedge liquidity valuation""".split(),
    "history": """empire dynasty revolution war treaty monarch ancient medieval
        century conquest civilisation kingdom battle reign colonial independence
        archaeology artefact chronicle settlement migration ruler""".split(),
    "geography": """mountain river valley desert climate continent ocean island
        peninsula plateau glacier volcano coastline latitude terrain ecosystem
        rainfall tropical temperate basin delta region""".split(),
    "sports": """team player match score goal championship tournament league coach
        athlete training stadium referee season victory defeat compete fitness
        olympic medal record final""".split(),
    "law": """court judge jury verdict statute constitution rights contract liability
        plaintiff defendant evidence appeal legislation jurisdiction precedent
        criminal civil tribunal ruling testimony""".split(),
    "chemistry": """molecule atom reaction compound element bond acid base catalyst
        solution concentration oxidation electron proton ion polymer organic
        synthesis crystal solvent isotope valence""".split(),
    "computing": """algorithm software program memory processor compiler database
        network protocol encryption cache index pointer recursion function variable
        thread concurrency operating system kernel storage""".split(),
}

_CONNECTORS = ("the of and to in is was for with that as on by an this from at "
               "which has are it its been their").split()


def _make_background_vocab(n: int = 4000, seed: int = 11) -> List[str]:
    """
    Generate ~n pronounceable pseudo-words for the vocabulary tail. The 12
    topic lists alone give only ~280 terms; these fillers restore a Zipfian
    long tail so posting-list lengths and dictionary size look realistic.
    """
    rng = random.Random(seed)
    cons = "bcdfghklmnprstvw"
    vow = "aeiou"
    words = set()
    while len(words) < n:
        syl = rng.randint(2, 3)
        w = "".join(rng.choice(cons) + rng.choice(vow) for _ in range(syl))
        words.add(w)
    return sorted(words)


_BACKGROUND = _make_background_vocab()
# Zipfian weights: a few background words are common, most are rare.
_BG_WEIGHTS = [1.0 / (i + 1) for i in range(len(_BACKGROUND))]
# Precomputed cumulative distribution. Passing cum_weights to random.choices
# avoids rebuilding the O(n) cumulative sum on every draw.
_BG_CUM = list(itertools.accumulate(_BG_WEIGHTS))


def make_synthetic_corpus(
    n_docs: int = 20000,
    seed: int = 7,
    min_len: int = 60,
    max_len: int = 260,
    n_spam: int = 250,
    spam_len: int = 1800,
) -> Tuple[List[Document], Dict[int, str]]:
    """
    Returns (documents, topic_of_doc). topic_of_doc maps doc_id -> topic name
    (or "spam" for distractor docs) and defines relevance for evaluation.

    Each document mixes connectors (~40%), its topic's vocabulary (~28%),
    cross-topic noise (~7%), and a Zipfian-sampled background tail (~25%).

    n_spam long keyword-stuffed documents are appended. Each contains every
    topic's vocabulary repeated heavily, so it matches every query while being
    relevant to none; length normalisation in the ranker is what keeps them
    out of the top-k.
    """
    rng = random.Random(seed)
    topic_names = list(TOPICS.keys())
    documents: List[Document] = []
    topic_of: Dict[int, str] = {}

    for doc_id in range(n_docs):
        topic = rng.choice(topic_names)
        vocab = TOPICS[topic]
        noise_topic = rng.choice([t for t in topic_names if t != topic])
        noise_vocab = TOPICS[noise_topic]

        length = rng.randint(min_len, max_len)
        words: List[str] = []
        for _ in range(length):
            r = rng.random()
            if r < 0.40:
                words.append(rng.choice(_CONNECTORS))
            elif r < 0.68:
                words.append(rng.choice(vocab))
            elif r < 0.75:
                words.append(rng.choice(noise_vocab))
            else:
                words.append(rng.choices(_BACKGROUND, cum_weights=_BG_CUM, k=1)[0])
        title = f"{topic.replace('_',' ').title()} article {doc_id}"
        url = f"https://example.org/doc/{doc_id}"
        documents.append((doc_id, title, url, " ".join(words)))
        topic_of[doc_id] = topic

    all_vocab = [w for v in TOPICS.values() for w in v]
    for s in range(n_spam):
        doc_id = n_docs + s
        words = [rng.choice(all_vocab) if rng.random() < 0.7 else rng.choice(_CONNECTORS)
                 for _ in range(spam_len)]
        documents.append((doc_id, f"Spam page {doc_id}",
                          f"https://example.org/spam/{doc_id}", " ".join(words)))
        topic_of[doc_id] = "spam"

    return documents, topic_of
