"""
silver_label.py — Auto-label sumbee social media data with the current NER model.

Produces two CoNLL files:
  data/silver_high.conll   — sentences where ALL entities scored >= CONF_THRESHOLD
                             Safe to add to training directly (still review a sample)
  data/silver_review.conll — sentences with at least one low-confidence entity
                             Must be manually corrected before using for training

Run from NLP-intelligence/:
    python scripts/silver_label.py
    python scripts/silver_label.py --limit 500   # quick test on first 500 rows
"""

import argparse
import csv
import os
import re
import sys
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_core.ner_engine import NEREngine
from nlp_core.preprocessing import Preprocessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUMBEE_CSV   = os.path.join("..", "preprocessing", "sumbee_master_dataset.csv")
OUT_HIGH     = os.path.join("data", "silver_high.conll")
OUT_REVIEW   = os.path.join("data", "silver_review.conll")
CONF_THRESHOLD = 0.85        # entities below this trigger "review" bucket
MN_PATTERN   = re.compile(r"[А-Яа-яӨөҮүЁё]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_mongolian(text: str) -> bool:
    return bool(MN_PATTERN.search(text))


def word_offsets(text: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, word) for each whitespace-separated token."""
    result = []
    pos = 0
    for word in text.split():
        start = text.find(word, pos)
        end = start + len(word)
        result.append((start, end, word))
        pos = end
    return result


def align_to_conll(preprocessed: str, entities) -> List[Tuple[str, str]]:
    """
    Map NER entity spans (char offsets) back to individual tokens.
    Returns list of (word, BIO-label) pairs.
    """
    offsets = word_offsets(preprocessed)
    labels = ["O"] * len(offsets)

    for ent in entities:
        e_start, e_end, e_type = ent.start, ent.end, ent.entity_group
        first = True
        for i, (ws, we, _) in enumerate(offsets):
            # token overlaps with entity span
            if ws < e_end and we > e_start:
                labels[i] = f"B-{e_type}" if first else f"I-{e_type}"
                first = False

    return [(word, lbl) for (_, _, word), lbl in zip(offsets, labels)]


def to_conll_block(pairs: List[Tuple[str, str]]) -> str:
    """Format (word, label) pairs as a CoNLL block (blank-line separated)."""
    lines = [f"{word} O O {label}" for word, label in pairs]
    return "\n".join(lines)


def min_entity_score(entities) -> float:
    if not entities:
        return 1.0
    return min(e.score for e in entities)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(limit: int = None):
    preprocessor = Preprocessor()
    ner = NEREngine()

    csv_path = os.path.join(os.path.dirname(__file__), SUMBEE_CSV)
    if not os.path.exists(csv_path):
        # try relative from project root
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "..", "preprocessing", "sumbee_master_dataset.csv")

    print(f"Reading sumbee data from {csv_path}")
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if is_mongolian(row["Text"]):
                rows.append(row["Text"])
            if limit and len(rows) >= limit:
                break

    print(f"Mongolian rows to label: {len(rows)}")

    high_blocks = []
    review_blocks = []
    skipped = 0

    for i, raw in enumerate(rows):
        if i % 100 == 0:
            print(f"  {i}/{len(rows)} ...", end="\r")

        preprocessed = preprocessor.preprocess_nlp(raw)
        if not preprocessed.strip():
            skipped += 1
            continue

        try:
            entities = ner.recognize(preprocessed)
        except Exception as e:
            skipped += 1
            continue

        pairs = align_to_conll(preprocessed, entities)
        if not pairs:
            skipped += 1
            continue

        block = to_conll_block(pairs)
        min_score = min_entity_score(entities)

        if min_score >= CONF_THRESHOLD:
            high_blocks.append(block)
        else:
            # Add a comment line so reviewer knows which entities to check
            low_ents = [f"{e.word}({e.entity_group},{e.score:.2f})"
                        for e in entities if e.score < CONF_THRESHOLD]
            review_blocks.append(f"# REVIEW: {', '.join(low_ents)}\n{block}")

    print(f"\nDone. High-confidence: {len(high_blocks)} | "
          f"Needs review: {len(review_blocks)} | Skipped: {skipped}")

    # Write outputs (relative to project root, so run from NLP-intelligence/)
    base = os.path.dirname(os.path.dirname(__file__))
    high_path   = os.path.join(base, "data", "silver_high.conll")
    review_path = os.path.join(base, "data", "silver_review.conll")

    with open(high_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(high_blocks))
    print(f"Saved: {high_path}")

    with open(review_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(review_blocks))
    print(f"Saved: {review_path}")
    print(f"\nNext step: review {review_path} manually, then run scripts/merge_train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N Mongolian rows (default: all)")
    args = parser.parse_args()
    main(args.limit)
