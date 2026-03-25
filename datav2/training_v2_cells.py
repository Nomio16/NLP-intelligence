"""
Training V2 — Cells for Colab Training.ipynb
Uses character-offset JSON format (datav2/train_v2_merged.jsonl)
Trains from previously fine-tuned model (checkpoint-3351)

Copy each section into a separate Colab cell.
"""

# ============================================================
# CELL 1: Setup
# ============================================================
"""
from google.colab import drive
drive.mount('/content/drive')
!pip install transformers datasets seqeval -q

# Upload datav2/train_v2_merged.jsonl to Drive first, OR
# upload directly to Colab:
# from google.colab import files
# uploaded = files.upload()  # select train_v2_merged.jsonl
"""

# ============================================================
# CELL 2: Load JSON data with character-offset labels
# ============================================================
"""
import json
import os

def load_jsonl(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get('text') and entry.get('labels'):
                entries.append(entry)
    return entries

# Load new training data (character-offset format)
train_data = load_jsonl('/content/drive/MyDrive/NER_finetune/train_v2_merged.jsonl')

# Also load existing CoNLL validation data (keep same eval set for fair comparison)
def parse_conll(path):
    sentences, labels = [], []
    cur_words, cur_labels = [], []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line == '' or line.startswith('#'):
                if cur_words:
                    sentences.append(cur_words)
                    labels.append(cur_labels)
                    cur_words, cur_labels = [], []
            else:
                parts = line.split()
                if len(parts) >= 4:
                    cur_words.append(parts[0])
                    cur_labels.append(parts[-1])
    if cur_words:
        sentences.append(cur_words)
        labels.append(cur_labels)
    return sentences, labels

valid_words, valid_labels = parse_conll('/content/drive/MyDrive/NER_finetune/valid.txt')

print(f"Train (JSON): {len(train_data)} sentences")
print(f"Valid (CoNLL): {len(valid_words)} sentences")
"""

# ============================================================
# CELL 3: Label setup (same labels as before for compatibility)
# ============================================================
"""
LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for i, l in enumerate(LABEL_LIST)}

def safe_label(l):
    return l if l in label2id else "O"

print(f"Labels: {LABEL_LIST}")
print(f"Mapping: {label2id}")
"""

# ============================================================
# CELL 4: Tokenize JSON data with CHARACTER-OFFSET alignment
#          THIS IS THE KEY DIFFERENCE FROM V1
# ============================================================
"""
from transformers import AutoTokenizer
import numpy as np

MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_json_entry(entry):
    '''
    Convert character-offset labels to token-level BIO labels.

    Input:  {"text": "Сайд Д.Батулга ...", "labels": [[5, 15, "PER"]]}
    Output: tokens with aligned BIO labels

    KEY: Every sub-token within a label span gets B-/I- label,
    INCLUDING dots and hyphens. No -100 for sub-tokens inside entities.
    '''
    text = entry['text']
    label_spans = entry['labels']  # [[start, end, label], ...]

    # Tokenize with offset mapping so we know where each token maps in original text
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors=None,
    )

    offset_mapping = encoding['offset_mapping']  # [(start, end), ...] for each token
    token_labels = []

    for idx, (tok_start, tok_end) in enumerate(offset_mapping):
        # Special tokens [CLS], [SEP], [PAD] have offset (0,0)
        if tok_start == 0 and tok_end == 0:
            token_labels.append(-100)
            continue

        # Check if this token falls within any entity span
        label = "O"
        for span_start, span_end, span_label in label_spans:
            # Token overlaps with entity span
            if tok_start >= span_start and tok_end <= span_end:
                # Is this the FIRST token in this entity span?
                is_first = True
                if idx > 0:
                    prev_start, prev_end = offset_mapping[idx - 1]
                    if prev_start >= span_start and prev_end <= span_end:
                        is_first = False

                if is_first:
                    label = f"B-{span_label}"
                else:
                    label = f"I-{span_label}"
                break

        # Map MISC to O (we don't train on MISC)
        if "MISC" in label:
            label = "O"

        token_labels.append(label2id.get(label, 0))

    encoding['labels'] = token_labels
    # Remove offset_mapping (not needed for training)
    del encoding['offset_mapping']
    return encoding


def tokenize_conll_data(words_list, labels_list):
    '''
    Tokenize CoNLL word-level data (for validation set).
    Uses word_ids mapping — first sub-token gets label, rest get -100.
    This matches the original training approach for fair eval comparison.
    '''
    tokenized = tokenizer(
        words_list,
        truncation=True,
        max_length=512,
        is_split_into_words=True,
        padding='max_length',
        return_tensors=None,
    )

    all_label_ids = []
    for i, labels in enumerate(labels_list):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                lbl = safe_label(labels[word_id])
                # Map MISC to O
                if "MISC" in lbl:
                    lbl = "O"
                label_ids.append(label2id[lbl])
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        all_label_ids.append(label_ids)
    tokenized["labels"] = all_label_ids
    return tokenized

# Tokenize training data (JSON with character offsets)
print("Tokenizing train (character-offset alignment)...")
train_encodings = {'input_ids': [], 'attention_mask': [], 'labels': []}

skipped = 0
for entry in train_data:
    try:
        enc = tokenize_json_entry(entry)
        train_encodings['input_ids'].append(enc['input_ids'])
        train_encodings['attention_mask'].append(enc['attention_mask'])
        train_encodings['labels'].append(enc['labels'])
    except Exception as e:
        skipped += 1
        continue

print(f"Train tokenized: {len(train_encodings['input_ids'])} sentences ({skipped} skipped)")

# Pad training to same length
max_len = 512
for i in range(len(train_encodings['input_ids'])):
    ids = train_encodings['input_ids'][i]
    mask = train_encodings['attention_mask'][i]
    labs = train_encodings['labels'][i]
    pad_len = max_len - len(ids)
    if pad_len > 0:
        train_encodings['input_ids'][i] = ids + [0] * pad_len
        train_encodings['attention_mask'][i] = mask + [0] * pad_len
        train_encodings['labels'][i] = labs + [-100] * pad_len

# Tokenize validation data (CoNLL word-level)
print("Tokenizing valid (word-level alignment)...")
valid_enc = tokenize_conll_data(valid_words, valid_labels)
print("Done")
"""

# ============================================================
# CELL 5: Verify tokenization — show example alignment
# ============================================================
"""
# Show how a name with dot is tokenized and labeled
sample_entry = {"text": "Сайд Д.Батулга мэдэгдэл хийлээ", "labels": [[5, 15, "PER"]]}
enc = tokenize_json_entry(sample_entry)

tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
labels = enc['labels']

print("Token alignment verification:")
print(f"Text: {sample_entry['text']}")
print(f"Entity span: [{5}:{15}] = '{sample_entry['text'][5:15]}'")
print()
print(f"{'Token':<15} {'Label':<10} {'Label Name'}")
print("-" * 40)
for tok, lab in zip(tokens, labels):
    if tok in ['[CLS]', '[SEP]', '[PAD]']:
        continue
    if lab == -100:
        lab_name = "(special)"
    else:
        lab_name = id2label[lab]
    print(f"{tok:<15} {lab:<10} {lab_name}")
"""

# ============================================================
# CELL 6: Create PyTorch datasets
# ============================================================
"""
import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

train_dataset = NERDataset(train_encodings)
valid_dataset = NERDataset(valid_enc)
print(f"Train: {len(train_dataset)} | Valid: {len(valid_dataset)}")
"""

# ============================================================
# CELL 7: Load model — FROM PREVIOUS FINE-TUNED CHECKPOINT
# ============================================================
"""
from transformers import AutoModelForTokenClassification

# Load from your PREVIOUSLY fine-tuned model (not base Davlan)
PREV_MODEL = "/content/drive/MyDrive/NER_finetune/checkpoints/checkpoint-3351"

model = AutoModelForTokenClassification.from_pretrained(
    PREV_MODEL,
    num_labels=len(LABEL_LIST),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # safe: same label set, so sizes match
)
print(f"Loaded from {PREV_MODEL}")
print(f"Labels: {model.config.num_labels}")
"""

# ============================================================
# CELL 8: Training config & train
# ============================================================
"""
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import f1_score, precision_score, recall_score

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_seqs, pred_seqs = [], []
    for pred_seq, label_seq in zip(preds, labels):
        true_s, pred_s = [], []
        for p_id, l_id in zip(pred_seq, label_seq):
            if l_id != -100:
                true_s.append(id2label[l_id])
                pred_s.append(id2label[p_id])
        true_seqs.append(true_s)
        pred_seqs.append(pred_s)
    return {
        "f1":        f1_score(true_seqs, pred_seqs),
        "precision": precision_score(true_seqs, pred_seqs),
        "recall":    recall_score(true_seqs, pred_seqs),
    }

data_collator = DataCollatorForTokenClassification(tokenizer)

args = TrainingArguments(
    output_dir="/content/drive/MyDrive/NER_finetune_v2/checkpoints",
    num_train_epochs=5,              # slightly more epochs for new data
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=300,                # more warmup for larger dataset
    weight_decay=0.01,
    learning_rate=2e-5,              # slightly lower LR for continued fine-tuning
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=3,
    fp16=True,
    report_to="none",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
"""

# ============================================================
# CELL 9: Save best model & check results
# ============================================================
"""
import json, os

# Save final model
save_path = "/content/drive/MyDrive/NER_finetune_v2/model_final"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved to {save_path}")
print(f"Best F1: {trainer.state.best_metric:.4f}")

# Compare checkpoints
base = "/content/drive/MyDrive/NER_finetune_v2/checkpoints"
for ckpt in sorted(os.listdir(base)):
    state_path = os.path.join(base, ckpt, "trainer_state.json")
    if os.path.exists(state_path):
        state = json.load(open(state_path))
        best = state.get("best_metric")
        best_ckpt = state.get("best_model_checkpoint")
        print(f"{ckpt}: f1={best}  best_checkpoint={best_ckpt}")
"""

# ============================================================
# CELL 10: Test the new model immediately
# ============================================================
"""
from transformers import pipeline

# Load the best model for inference
ner_pipe = pipeline(
    "ner",
    model=save_path,
    tokenizer=save_path,
    aggregation_strategy="simple",
    device=0
)

# Test cases — these should all work correctly now
test_cases = [
    "Монгол улсын ерөнхий сайд Л.Оюун-Эрдэнэ УИХ-ын чуулганд оролцлоо",
    "Д.Цогтбаатар хэвлэлийн хурал хийлээ",
    "Пурэвдоржийн Номин-Эрдэнэ шагнал хүртлээ",
    "МАН-ын дарга Батулгыг шүүхэд дуудсан",
    "Архангай аймагт бороо орно",
    "АНУ-ын ерөнхийлөгч Ж.Байден Монголд зочиллоо",
    "Баянгол дүүрэгт шинэ сургууль баригдана",
    "НИТХ-ын хурал болж байна",
    "Х.Баттулгын асуудлыг хэлэлцэнэ",
    "УИХ-ын гишүүн С.Ганбаатартай уулзав",
]

print("=" * 60)
print("MODEL V2 — NER Test Results")
print("=" * 60)
for text in test_cases:
    entities = ner_pipe(text)
    print(f"\\nInput: {text}")
    if entities:
        for e in entities:
            print(f'  {e["word"]:<25s} {e["entity_group"]:<5s} {e["score"]:.3f}')
    else:
        print("  (no entities found)")
"""

# ============================================================
# CELL 11: Upload new model to HuggingFace Hub
# ============================================================
"""
!pip install huggingface_hub -q
from huggingface_hub import login, HfApi

login("hf_YOUR_TOKEN")  # your -nlp token

api = HfApi()
api.upload_folder(
    folder_path=save_path,
    repo_id="Nomio4640/ner-mongolian",
    repo_type="model"
)
print("Uploaded to HuggingFace Hub!")
"""
