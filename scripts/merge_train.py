"""
merge_train.py — Merge original train.txt with silver-labeled sumbee data.

Run AFTER reviewing data/silver_review.conll manually.

Usage:
    python scripts/merge_train.py                    # merges high + original
    python scripts/merge_train.py --include-review   # also includes review file
                                                     # (only after manual correction)

Output: data/train_merged.conll  (use this for Colab fine-tuning)
"""

import argparse
import os
import random


def read_conll_blocks(path: str):
    """Read a CoNLL file and return list of sentence blocks (skip # comments)."""
    blocks = []
    with open(path, encoding="utf-8") as f:
        current = []
        for line in f:
            line = line.rstrip()
            if line.startswith("#"):       # strip comment lines from review file
                continue
            if line == "":
                if current:
                    blocks.append("\n".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))
    return [b for b in blocks if b.strip()]


def main(include_review: bool = False, seed: int = 42):
    base = os.path.dirname(os.path.dirname(__file__))

    original   = os.path.join(base, "data", "train.txt")
    silver_high   = os.path.join(base, "data", "silver_high.conll")
    silver_review = os.path.join(base, "data", "silver_review_done.conll")
    output     = os.path.join(base, "data", "train_merged.conll")

    if not os.path.exists(original):
        print(f"ERROR: {original} not found"); return
    if not os.path.exists(silver_high):
        print(f"ERROR: {silver_high} not found — run silver_label.py first"); return

    print("Reading original train.txt ...")
    orig_blocks = read_conll_blocks(original)
    print(f"  {len(orig_blocks)} sentences")

    print("Reading silver_high.conll ...")
    high_blocks = read_conll_blocks(silver_high)
    print(f"  {len(high_blocks)} sentences")

    all_blocks = orig_blocks + high_blocks

    if include_review:
        if not os.path.exists(silver_review):
            print(f"WARNING: {silver_review} not found, skipping")
        else:
            print("Reading silver_review.conll (assuming manually corrected) ...")
            review_blocks = read_conll_blocks(silver_review)
            print(f"  {len(review_blocks)} sentences")
            all_blocks += review_blocks

    random.seed(seed)
    random.shuffle(all_blocks)

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_blocks))
        f.write("\n")

    print(f"\nMerged {len(all_blocks)} sentences → {output}")
    print("Upload this file to Google Drive for Colab fine-tuning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-review", action="store_true",
                        help="Also include silver_review.conll (only after manual correction)")
    args = parser.parse_args()
    main(include_review=args.include_review)
