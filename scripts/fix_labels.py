"""
fix_labels.py — Auto-correct known labeling errors in train_merged.conll.

Fixes applied:
  1. Sequence errors: I-X without preceding B-X or I-X → convert to B-X
  2. Definite wrong labels: common words incorrectly tagged as entities
  3. Systematic silver-label error: томилолт I-PER → O

Run from NLP-intelligence/:
    python scripts/fix_labels.py
Output: data/train_final.conll
"""

import os
import sys

# Words that are NEVER entities in any context
ALWAYS_O = {
    # Verbs wrongly tagged as entities
    "байна":  {"B-PER"},           # "is/are"
    "байгаа": {"I-MISC"},          # "being"
    "хийж":   {"B-PER", "B-LOC"}, # verb "doing"
    # Particles / pronouns wrongly tagged
    "юм":     {"I-MISC"},          # particle
    "бол":    {"I-MISC"},          # copula "is"
    "нэг":    {"I-MISC"},          # "one"
    "би":     {"I-MISC"},          # pronoun "I"
    "ямар":   {"B-PER"},           # interrogative "what kind"
    "та":     {"B-PER"},           # pronoun "you"
    "сарын":  {"B-PER"},           # "of the month"
    "мөн":    {"B-LOC"},           # adverb "also"
    "манай":  {"B-LOC"},           # possessive "our" — not a location
    # Number
    "2":      {"I-PER"},
    # Systematic silver error: "assignment/delegation" ≠ person
    "томилолт": {"I-PER"},
}


def fix_block(tokens):
    """
    tokens: list of (word, label)
    Returns fixed list of (word, label).
    """
    result = []
    prev_label = "O"
    prev_type = None

    for word, label in tokens:
        fixed = label

        # Fix 1: wrong labels for specific words
        key = word.lower()
        if key in ALWAYS_O and label in ALWAYS_O[key]:
            fixed = "O"

        # Fix 2: I-X without matching B-X or I-X before it → B-X
        if fixed.startswith("I-"):
            etype = fixed[2:]
            if prev_label == "O" or (
                prev_label.startswith("B-") and prev_label[2:] != etype
            ) or (
                prev_label.startswith("I-") and prev_label[2:] != etype
            ):
                fixed = f"B-{etype}"

        result.append((word, fixed))
        prev_label = fixed
        prev_type = fixed[2:] if "-" in fixed else None

    return result


def main():
    base = os.path.dirname(os.path.dirname(__file__))
    src  = os.path.join(base, "data", "train_merged.conll")
    dst  = os.path.join(base, "data", "train_final.conll")

    if not os.path.exists(src):
        print(f"ERROR: {src} not found"); sys.exit(1)

    fixed_count = 0
    seq_fixed   = 0
    out_blocks  = []

    with open(src, encoding="utf-8") as f:
        current_raw = []
        for line in f:
            line = line.rstrip()
            if line == "" or line.startswith("#"):
                if current_raw:
                    tokens = current_raw
                    fixed  = fix_block(tokens)
                    # Count changes
                    for (_, ol), (_, nl) in zip(tokens, fixed):
                        if ol != nl:
                            if ol.startswith("I-") and nl.startswith("B-"):
                                seq_fixed += 1
                            else:
                                fixed_count += 1
                    out_blocks.append(fixed)
                    current_raw = []
            else:
                parts = line.split()
                if len(parts) >= 4:
                    current_raw.append((parts[0], parts[-1]))

        if current_raw:
            out_blocks.append(fix_block(current_raw))

    with open(dst, "w", encoding="utf-8") as f:
        for block in out_blocks:
            for word, label in block:
                f.write(f"{word} O O {label}\n")
            f.write("\n")

    print(f"Wrong-label fixes:    {fixed_count}")
    print(f"Sequence fixes (I→B): {seq_fixed}")
    print(f"Sentences written:    {len(out_blocks)}")
    print(f"Saved → {dst}")
    print(f"\nUse data/train_final.conll for Colab fine-tuning.")


if __name__ == "__main__":
    main()
