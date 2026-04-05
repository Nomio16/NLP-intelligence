"""
evaluate_tokens.py — Token-level seqeval evaluation matching the Colab training metric.

Unlike evaluate.py (which reconstructs text and runs the full NLP pipeline),
this script feeds pre-tokenized CoNLL words directly to the model, ensuring
the evaluation is identical to what Colab measured during training.

Run from NLP-intelligence/:
    python eval/evaluate_tokens.py
    python eval/evaluate_tokens.py --limit 500
"""

import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_LABELS = {"PER", "LOC", "ORG"}   # MISC excluded — not in fine-tuned model


def parse_conll(path, limit=None):
    sentences, labels = [], []
    cur_w, cur_l = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line == "" or line.startswith("#"):
                if cur_w:
                    sentences.append(cur_w)
                    labels.append(cur_l)
                    cur_w, cur_l = [], []
                    if limit and len(sentences) >= limit:
                        break
            else:
                parts = line.split()
                if len(parts) >= 4:
                    cur_w.append(parts[0])
                    raw = parts[-1]
                    # Remap MISC → O so evaluation is PER/LOC/ORG only
                    cur_l.append("O" if "MISC" in raw else raw)
    if cur_w:
        sentences.append(cur_w)
        labels.append(cur_l)
    return sentences, labels


def predict_tokens(words_list, tokenizer, model, device, batch_size=32):
    """
    Run token classification on pre-tokenized word lists.
    Returns list of per-sentence label sequences aligned to original words.
    """
    import torch
    from torch.nn.functional import softmax

    all_preds = []

    for i in range(0, len(words_list), batch_size):
        if i % 200 == 0:
            print(f"  {i}/{len(words_list)} sentences...", end="\r")

        batch_words = words_list[i: i + batch_size]
        enc = tokenizer(
            batch_words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        # keep BatchEncoding for word_ids() before moving tensors to device
        word_ids_per_sent = [enc.word_ids(batch_index=b) for b in range(len(batch_words))]
        model_input = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**model_input).logits  # (batch, seq, num_labels)

        preds_ids = logits.argmax(-1).cpu().tolist()

        for b_idx, words in enumerate(batch_words):
            word_ids = word_ids_per_sent[b_idx]
            word_preds = {}
            for pos, wid in enumerate(word_ids):
                if wid is None or wid in word_preds:
                    continue          # skip [CLS]/[SEP]/padding and non-first subwords
                word_preds[wid] = model.config.id2label[preds_ids[b_idx][pos]]
            sent_preds = [word_preds.get(j, "O") for j in range(len(words))]
            all_preds.append(sent_preds)

    print()
    return all_preds


def main(limit=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from seqeval.metrics import (classification_report, f1_score,
                                  precision_score, recall_score)

    base       = os.path.dirname(os.path.dirname(__file__))
    test_path  = os.path.join(base, "Data", "data", "test.txt")
    model_path = os.path.join(base, "adapters", "ner_mongolian")

    if not os.path.exists(model_path):
        print(f"ERROR: Fine-tuned model not found at {model_path}")
        print("Run fine-tuning first and place model at adapters/ner_mongolian/")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()
    print(f"Model loaded on {device}")

    print(f"Parsing {test_path}...")
    sentences, true_labels = parse_conll(test_path, limit=limit)
    print(f"Sentences: {len(sentences)}")

    print("Running token-level prediction...")
    pred_labels = predict_tokens(sentences, tokenizer, model, device)

    print("\n" + "=" * 50)
    print("NER EVALUATION RESULTS (Token-Level, seqeval)")
    print("=" * 50)
    print(classification_report(true_labels, pred_labels))
    print(f"Overall F1:        {f1_score(true_labels, pred_labels):.4f}")
    print(f"Overall Precision: {precision_score(true_labels, pred_labels):.4f}")
    print(f"Overall Recall:    {recall_score(true_labels, pred_labels):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate on first N sentences only")
    args = parser.parse_args()
    main(args.limit)
