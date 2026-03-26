import pandas as pd
from nlp_core.topic_modeler import TopicModeler

def main():
    csv_path = "NER-dataset/mongolian_news_demo.csv"
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path).head(50)
    
    texts = df["Text"].tolist()
    print(f"Loaded {len(texts)} texts.")
    
    print("Fitting topic model...")
    modeler = TopicModeler()
    results, summary = modeler.fit_transform(texts)
    
    print("\n--- Topic Summary ---")
    for s in summary:
        print(s)

    print("\n--- Sample results ---")
    for i, res in enumerate(results[:5]):
        print(f"Doc {i}: Topic {res.topic_id} ({res.topic_label}) - Probs: {res.probability:.4f}")
        print(f"Keywords: {res.keywords}")
        print(f"Text: {texts[i][:100]}...\n")

if __name__ == "__main__":
    main()
