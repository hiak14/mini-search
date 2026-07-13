# mini-search

A small information retrieval engine in pure Python, no external IR libraries.
It builds a compressed inverted index over a document collection and answers
boolean, phrase, and ranked free-text queries with BM25 or TF-IDF.

On a 30,250-document benchmark corpus it produces a 9.6 MB index (30.3% of the
raw corpus), compresses postings by 72.7% versus a fixed-width baseline, and
serves ranked queries at p50 = 4.65 ms / p95 = 6.27 ms.

## Quick start

```bash
# build an index over a generated corpus (no download needed)
python cli.py build --synthetic 30000 --out index_dir

# query it
python cli.py search "machine learning model" --index index_dir
python cli.py search --bool "python AND (search OR index) NOT java" --index index_dir
python cli.py search --phrase "neural network" --index index_dir

# tests
python -m unittest discover -s tests -v

# reproduce the benchmark numbers below
python benchmark.py
```

Search flags: `--k N` (top-k), `--tfidf` (use TF-IDF instead of BM25),
`--k1` / `--b` (BM25 parameters).

## How it works

```
text --> analyzer --> indexer (SPIMI) --> postings.bin + dict.bin + docstore.json
                                              |
query --> analyzer --> query processing <-- index reader
                          |
                        ranker (BM25 / TF-IDF)
```

| Module        | Responsibility |
|---------------|----------------|
| `analyzer.py` | tokenise, lowercase, accent-fold, stopword filter, Porter stem. Emits (term, position) pairs; positions are raw token offsets so phrase queries survive stopword removal |
| `porter.py`   | Porter (1980) stemmer, all five rule steps |
| `encoding.py` | variable-byte integer coding + gap coding |
| `store.py`    | doc_id -> title, url, length; corpus stats for BM25 |
| `indexer.py`  | SPIMI: stream docs, flush sorted blocks at a memory cap, k-way merge with a heap |
| `index.py`    | loads the dictionary, decodes posting lists on demand, builds sqrt(n) skip pointers for long lists |
| `ranker.py`   | BM25 and TF-IDF |
| `query.py`    | intersect/union/difference, recursive-descent boolean parser, positional phrase search, document-at-a-time top-k |
| `evaluate.py` | P@10, MAP, nDCG@10, latency percentiles |
| `corpus.py`   | Wikipedia JSONL loader + synthetic corpus generator |
| `cli.py`      | build / search commands |

### On-disk format

Three files, written by `indexer.py`:

- `postings.bin`: for each term, in dictionary order: the document frequency,
  then per posting the gap-from-previous doc_id, term frequency, position
  count, and gap-encoded positions, all VByte-coded.
- `dict.bin`: pickled map of term -> (df, byte offset, byte length), so any
  posting list is one seek and one decode away.
- `docstore.json`: per-document title, URL, and length.

### Design notes

SPIMI rather than a single in-memory dict: peak memory is bounded by the block
size instead of the corpus size, at the cost of a merge pass. The index test
forces a two-posting block cap to exercise the merge path on a corpus whose
postings are known by hand.

Gap + VByte rather than raw integers or a bit-level coder: doc ids in a
posting list are strictly increasing, so gaps are small, and VByte encodes
small numbers in one byte. It gets most of the compression of Elias-gamma or
PForDelta while staying byte-aligned and fast to decode. Measured here it
removes 72.7% of the postings bytes.

Posting lists with df >= 16 get sqrt(n) skip pointers, so AND queries can jump
over non-matching stretches. This is why boolean latency (p50 = 1.32 ms) stays
below ranked latency over the same data.

Stopword removal is a toggle, on by default. It shrinks the index and helps
ranked search, but makes pure-stopword phrases ("to be or not to be")
unsearchable. The analyzer keeps raw token positions rather than post-filter
positions, so phrase queries over content words stay correct either way.

Ranked retrieval is document-at-a-time with a bounded min-heap for the top-k.
DAAT pairs naturally with doc_id-ordered posting lists: one forward pass, no
random access, and per-document length normalisation is trivial to apply.

## Benchmark results

From `python benchmark.py`: 30,000 topic documents plus 250 keyword-stuffed
spam distractors, 30,250 docs total.

Corpus and index:

| Metric | Value |
|---|---|
| Documents indexed | 30,250 |
| Vocabulary | 4,271 terms |
| Average document length | 105.2 tokens |
| Raw corpus size | 31.65 MB |
| postings.bin | 9.52 MB |
| dict.bin | 0.09 MB |
| Index total | 9.60 MB (30.3% of corpus) |
| SPIMI blocks merged | 7 |
| Build time | 14.16 s (~2,136 docs/s) |

Compression: 34.90 MB of uncompressed postings (fixed 4-byte ints) vs 9.52 MB
with gap + VByte, a 72.7% saving.

Query latency (ms, 360 queries per type):

| Query type | p50 | p95 | mean |
|---|---|---|---|
| ranked (BM25) | 4.65 | 6.27 | 4.88 |
| boolean | 1.32 | 2.05 | 3.01 |
| phrase | 4.37 | 5.81 | 4.52 |

Retrieval quality (top-10, 12 topic queries):

| Ranker | P@10 | MAP | nDCG@10 |
|---|---|---|---|
| BM25 | 1.000 | 0.004 | 1.000 |
| TF-IDF | 1.000 | 0.004 | 1.000 |

Two caveats on the quality numbers. BM25 and TF-IDF tie because the synthetic
topics are cleanly separable; any reasonable ranker puts the right documents
in the top 10. BM25's advantage shows on graded relevance, which this
generator does not produce. And MAP is low by construction, not failure: each
topic has ~2,500 relevant documents but only 10 are retrieved, so average
precision is recall-bounded at roughly k / |relevant|. P@10 and nDCG@10 are
the meaningful signals at this k.

BM25 k1/b sweep, nDCG@10:

|  k1 \ b | 0.0 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| 1.2 | 0.042 | 1.000 | 1.000 | 1.000 |
| 1.5 | 0.042 | 1.000 | 1.000 | 1.000 |
| 2.0 | 0.042 | 1.000 | 1.000 | 1.000 |

This is what the spam documents are for. With length normalisation off
(b = 0), the long keyword-stuffed distractors, which contain every topic's
vocabulary many times over, flood the top 10 and nDCG collapses to 0.042. With
b >= 0.5, BM25 divides out the document-length advantage and the spam drops
away. This is the failure mode the b parameter exists to fix, and why plain
term-frequency ranking is not enough.

## Running on real data

The synthetic corpus exists so the repo runs without a large download. To try
mini-search on real text, index Simple English Wikipedia (240k+ articles).

The README previously suggested pulling the raw XML dump and running it
through [wikiextractor](https://github.com/attardi/wikiextractor). On Windows
that path currently breaks in three separate ways: wikiextractor's regex
module uses inline `(?i)` flags that Python 3.11+ rejects as a hard error, and
its multiprocessing is hardcoded to `fork`, which does not exist on Windows
(only Linux/macOS have it) — switching to `spawn` then fails too, because the
reducer process tries to pickle an open file handle across the process
boundary, which `spawn` cannot do. None of this is a mini-search bug; it is
wikiextractor's Unix-only process model colliding with Windows. If you're on
Linux/macOS, or run this inside WSL, the original wikiextractor command
should work unmodified.

The simpler, cross-platform fix: skip wikiextractor and pull an already-parsed
copy of Simple English Wikipedia from Hugging Face's `datasets` library. It
returns the same fields mini-search expects (`id`, `title`, `url`, `text`),
with no XML parsing, no extraction step, and no multiprocessing at all.

```bash
pip install datasets
```

```python
# fetch_wiki.py
import json
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")

with open("wiki.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps({
            "id": str(row["id"]),
            "title": row["title"],
            "url": row["url"],
            "text": row["text"],
        }) + "\n")
```

```bash
python fetch_wiki.py

python cli.py build --wiki wiki.jsonl --out wiki_index --progress 10000
python cli.py search "solar eclipse" --index wiki_index
```

`load_wikipedia_jsonl` streams the file, so the indexer never holds the whole
corpus in memory.

### Real-data results

Indexing the full Simple English Wikipedia dump (20231101 snapshot):

| Metric | Value |
|---|---|
| Documents indexed | 241,787 |
| Vocabulary | 680,151 terms |
| Average document length | 122.4 tokens |
| postings.bin | ~95.1 MB |
| SPIMI blocks merged | 17 |
| Build time | 32.3 s |

`python cli.py search "solar eclipse" --index wiki_index` returns real
Wikipedia articles ranked sensibly by BM25 — the "Solar eclipse" article
itself first, followed by closely related topics (lunar eclipse, Saros cycle,
specific eclipse events, a disambiguation page), with scores decaying
smoothly rather than jumping around:

```
 1. [26.6729] Solar eclipse
 2. [26.4797] Solar eclipse of December 14, 2020
 3. [25.5158] Lunar eclipse
 4. [25.0890] Solar eclipse of June 10, 2021
 5. [24.6159] Eclipse
 6. [24.5437] Eclipse (disambiguation)
 7. [24.2249] Solar eclipse of June 21, 2020
 8. [24.1994] Solar eclipse of July 16, 2186
 9. [23.9650] Saros cycle
10. [22.4137] Solar eclipse of August 21, 2017
```

Vocabulary size jumps from 4,271 terms on the synthetic corpus to 680,151 on
real Wikipedia — real language carries far more lexical diversity (proper
nouns, rare terms, foreign words) than a generated topic corpus, and this is
reflected directly in dictionary size.

## Tests

```bash
python -m unittest discover -s tests -v
```

| File | Covers |
|---|---|
| `test_porter.py` | 75 Porter reference pairs across all five rule steps |
| `test_encoding.py` | VByte and gap round-trips over random lists, including the byte-boundary values where the continuation bit flips |
| `test_index.py` | tiny corpus with a 2-posting block cap, forcing a real k-way merge; asserts exact postings and positions |
| `test_query.py` | intersect/union cross-checked against set operations on 6,000 random inputs; boolean precedence and parentheses; phrase adjacency |

## Limitations

- The headline benchmark numbers are on a synthetic corpus. It is structured
  to give meaningful ranking behaviour (topic vocabularies, a Zipfian
  background tail, adversarial spam) but it is not natural text. See "Running
  on real data" above for results on real Wikipedia articles.
- The dictionary loads fully into memory (0.09 MB on the synthetic corpus;
  larger on real Wikipedia). A very large vocabulary would want an on-disk
  term dictionary.
- Ranked retrieval is exhaustive DAAT. WAND/MaxScore dynamic pruning would cut
  tail latency on large corpora; the `max_term_score` hook in `ranker.py` is
  in place for it.

