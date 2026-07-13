"""
Fetch Simple English Wikipedia from Hugging Face and write it to wiki.jsonl
in the format mini-search's cli.py --wiki expects (id, title, url, text).

Bypasses wikiextractor entirely, so none of the Windows fork/spawn issues apply.
"""

import json
from datasets import load_dataset

print("Downloading Simple English Wikipedia (this may take a few minutes)...")
ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
print(f"Loaded {len(ds)} articles. Writing to wiki.jsonl ...")

with open("wiki.jsonl", "w", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        f.write(json.dumps({
            "id": str(row["id"]),
            "title": row["title"],
            "url": row["url"],
            "text": row["text"],
        }) + "\n")
        if (i + 1) % 20000 == 0:
            print(f"  wrote {i + 1} articles...")

print("Done. wiki.jsonl is ready.")