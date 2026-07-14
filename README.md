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

SPIMI keeps peak memory in check by limiting it to the block size, not the size 
of the whole corpus. The tradeoff is that you have to merge blocks at the end, 
but that's manageable. The index test deliberately splits postings into just 
two blocks—this way, I can see the merge in action on a small, hand-checked set.

For compressing postings, I use gap encoding with VByte instead of storing raw 
integers or squeezing everything down to bits. Since doc IDs always go up, the 
differences (gaps) between them are small. VByte packs small numbers into one byte 
fast, and you don’t need weird bit-level tricks. You get almost as much compression
as things like Elias-gamma or PForDelta, but it's simpler and stays byte-aligned,
making it quick to decode. In my tests, this got rid of about 72.7% of the bytes in postings.

If a term shows up in at least 16 documents, I add sqrt(n) skip pointers to its posting 
list. This lets AND queries leap over long stretches of irrelevant docs, so Boolean 
queries run fast (the median Boolean time is 1.32 ms-always snappier than ranked search 
on the same data).

Stopword removal is a toggle, on by default. It shrinks the index and helps
ranked search, but makes pure-stopword phrases ("to be or not to be")
unsearchable. The analyzer keeps raw token positions rather than post-filter
positions, so phrase queries over content words stay correct either way.

Ranked retrieval is a straight doc-at-a-time loop with a bounded min-heap to stash 
the top k docs. Posting lists are sorted by doc_id, so you just walk forward; no random 
jumping around, and normalizing for document length is simple.

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

There are a couple things to watch out for with the quality metrics. BM25 and TF-IDF “tie” 
because the test topics are intentionally easy—any solid ranker will find the right docs. 
BM25 really pulls ahead when you have graded relevance, but this test setup doesn’t generate 
that. Also, MAP (Mean Average Precision) shows up low—not because the system can’t rank, but
because there are a ton of relevant docs (~2,500 per topic), and only 10 are retrieved per 
query. So, average precision can’t get higher than about k divided by the number of relevant
docs. For this scenario, P@10 and nDCG@10 actually matter most.

BM25 k1/b sweep, nDCG@10:

|  k1 \ b | 0.0 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| 1.2 | 0.042 | 1.000 | 1.000 | 1.000 |
| 1.5 | 0.042 | 1.000 | 1.000 | 1.000 |
| 2.0 | 0.042 | 1.000 | 1.000 | 1.000 |

Spam documents are in the test set to check length normalization. 
If you turn off normalization (b = 0), those long, keyword-stuffed spam docs 
blast into the top 10, and nDCG sinks to 0.042. Once b is 0.5 or higher, 
BM25 cancels out the length advantage and the spam falls off the list. 
That's exactly why BM25 has this b parameter—and why you can’t rely on plain term-frequency ranking.

## Running on real data

The synthetic corpus exists so the repo runs without a large download. To try
mini-search on real text, index Simple English Wikipedia (240k+ articles).

Pull an already-parsed copy of Simple English Wikipedia from Hugging Face's `datasets` library. 
It returns the same fields mini-search expects (`id`, `title`, `url`, `text`),
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

-The main benchmark uses a synthetic dataset, designed to test ranking (with topic vocab, a Zipf-like tail, and tricky spam examples), but it’s not real natural language. For results with -real Wikipedia, see “Running on real data” above.
-The term dictionary loads fully into memory. That’s tiny for the toy set (0.09 MB), but gets bigger for real Wikipedia. If the vocab gets huge, you’d want to swap to an on-disk dictionary.
-Ranked retrieval does a full DAAT pass. On big corpora, smarter dynamic pruning (WAND/MaxScore) would speed up long-tail queries. There's a hook—max_term_score—in ranker.py ready for this.

