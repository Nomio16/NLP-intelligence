"""
NER Engine — Named Entity Recognition using HuggingFace Transformers.
Wraps the Davlan/bert-base-multilingual-cased-ner-hrl model.
"""

from typing import List, Optional
from .models import EntityResult


class NEREngine:
    """Named Entity Recognition service using HuggingFace pipeline."""

    def __init__(self, model_name: str = "Davlan/bert-base-multilingual-cased-ner-hrl"):
        self.model_name = model_name
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the NER pipeline (heavy model, load only when needed)."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
            )
        return self._pipeline

    def _clean_entities(self, raw_entities: List[dict]) -> List[dict]:
        """Merge subword tokens (## prefixed) back together."""
        cleaned = []
        for ent in raw_entities:
            word = ent.get("word", "")
            if word.startswith("##") and len(cleaned) > 0:
                cleaned[-1]["word"] += word.replace("##", "")
            else:
                cleaned.append(dict(ent))
        return cleaned

    def recognize(self, text: str) -> List[EntityResult]:
        """Run NER on a single text and return cleaned entities."""
        if not text or not text.strip():
            return []
        pipe = self._load_pipeline()
        try:
            raw = pipe(text)
        except Exception:
            return []

        cleaned = self._clean_entities(raw)
        results = []
        for ent in cleaned:
            results.append(EntityResult(
                word=ent.get("word", ""),
                entity_group=ent.get("entity_group", "MISC"),
                score=float(ent.get("score", 0.0)),
                start=int(ent.get("start", 0)),
                end=int(ent.get("end", 0)),
            ))
        return results

    def recognize_batch(self, texts: List[str]) -> List[List[EntityResult]]:
        """Run NER on a batch of texts."""
        return [self.recognize(t) for t in texts]
