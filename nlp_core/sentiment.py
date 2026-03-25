"""
Sentiment Analysis service using HuggingFace XLM-RoBERTa.
Wraps cardiffnlp/twitter-xlm-roberta-base-sentiment model.
"""

from typing import List, Optional
from .models import SentimentResult


# Map model labels to human-readable labels.
# Keys include both original-case and .lower() forms because we call
# result["label"].lower() before the lookup — the uppercase forms would
# never match after lowercasing.
LABEL_MAP = {
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    # Original-case (kept for safety if .lower() is ever removed)
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    # Lowercased forms — these are what actually get looked up
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}


class SentimentAnalyzer:
    """Sentiment analysis service using XLM-RoBERTa."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        self.model_name = model_name
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the sentiment pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                max_length=512,
            )
        return self._pipeline

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        if not text or not text.strip():
            return SentimentResult(label="neutral", score=0.0)
        pipe = self._load_pipeline()
        try:
            result = pipe(text)[0]
            raw_label = result.get("label", "neutral").lower()
            label = LABEL_MAP.get(raw_label, raw_label)
            return SentimentResult(
                label=label,
                score=float(result.get("score", 0.0)),
            )
        except Exception:
            return SentimentResult(label="neutral", score=0.0)

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[SentimentResult]:
        """Analyze sentiment of a batch of texts."""
        if not texts:
            return []
        pipe = self._load_pipeline()
        try:
            results = pipe(texts, batch_size=batch_size)
            out = []
            for result in results:
                raw_label = result.get("label", "neutral").lower()
                label = LABEL_MAP.get(raw_label, raw_label)
                out.append(SentimentResult(
                    label=label,
                    score=float(result.get("score", 0.0)),
                ))
            return out
        except Exception:
            return [SentimentResult(label="neutral", score=0.0) for _ in texts]
