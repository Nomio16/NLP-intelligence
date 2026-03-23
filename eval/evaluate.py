import os
import sys
import logging

# Add the project root to the python path so we can import nlp_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_core.ner_engine import NEREngine
from nlp_core.preprocessing import Preprocessor

def extract_entities_from_conll(lines):
    """
    Extracts entities from a list of CoNLL-formatted lines for a single sentence.
    Returns the reconstructed text and a list of entities: (type, string).
    """
    words = []
    entities = []
    current_entity_type = None
    current_entity_words = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        word = parts[0]
        tag = parts[-1]
        
        words.append(word)

        if tag.startswith("B-"):
            if current_entity_type:
                entities.append((current_entity_type, " ".join(current_entity_words)))
            current_entity_type = tag[2:]
            current_entity_words = [word]
        elif tag.startswith("I-"):
            if current_entity_type == tag[2:]:
                current_entity_words.append(word)
            else:
                if current_entity_type:
                    entities.append((current_entity_type, " ".join(current_entity_words)))
                current_entity_type = tag[2:]
                current_entity_words = [word]
        else:
            if current_entity_type:
                entities.append((current_entity_type, " ".join(current_entity_words)))
                current_entity_type = None
                current_entity_words = []

    if current_entity_type:
        entities.append((current_entity_type, " ".join(current_entity_words)))

    text = " ".join(words)
    return text, entities

def evaluate_ner(test_file_path, limit=None):
    print(f"Loading test data from {test_file_path}...")
    
    with open(test_file_path, "r", encoding="utf-8") as f:
        blocks = f.read().split("\n\n")

    sentences = []
    for block in blocks:
        if not block.strip():
            continue
        text, true_ents = extract_entities_from_conll(block.split("\n"))
        if text:
            sentences.append((text, true_ents))

    if limit:
        sentences = sentences[:limit]

    print(f"Loaded {len(sentences)} test sentences.")
    
    preprocessor = Preprocessor()
    ner = NEREngine()
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    print("Running NER evaluation (this may take a while)...")
    for i, (text, true_ents) in enumerate(sentences):
        if i > 0 and i % 50 == 0:
            print(f"Processed {i}/{len(sentences)} sentences...")
            
        # Clean text specifically for NER
        clean_text = preprocessor.preprocess_nlp(text)
        
        predicted_results = ner.recognize(clean_text)
        
        # Format predictions into (type, string) lowercased for fair comparison
        # Strip dots so Д.Гантулга and Д. Гантулга both normalize to дгантулга
        pred_ents = [(res.entity_group, res.word.replace(" ", "").replace(".", "").lower())
                     for res in predicted_results]

        # Format true entities similarly — skip MISC since the fine-tuned model
        # does not produce MISC labels (removed from training set)
        true_ents_formatted = [
            (t, w.replace(" ", "").replace(".", "").lower())
            for t, w in true_ents
            if t != "MISC"
        ]
        
        # Calculate overlaps
        for true_e in true_ents_formatted:
            if true_e in pred_ents:
                true_positives += 1
                pred_ents.remove(true_e)
            else:
                false_negatives += 1
                
        # Whatever is left in pred_ents are false positives
        false_positives += len(pred_ents)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print("NER EVALUATION RESULTS (Entity-Level Exact Match)")
    print("="*40)
    print(f"Sentences Evaluated: {len(sentences)}")
    print(f"True Positives:      {true_positives}")
    print(f"False Positives:     {false_positives}")
    print(f"False Negatives:     {false_negatives}")
    print("-" * 40)
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test.txt")
    if not os.path.exists(test_path):
        print(f"Error: Could not find CoNLL test file at {test_path}")
    else:
        # Run on the first 500 sentences to get a quick estimate. 
        # Change limit=None to run on the entire test set.
        evaluate_ner(test_path, limit=500)
