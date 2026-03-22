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
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
                device=device,
            )
            print(f"[NEREngine] Loaded on {'GPU' if device == 0 else 'CPU'}")
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

    def recognize_batch(self, texts: List[str], batch_size: int = 16) -> List[List[EntityResult]]:
        """Run NER on a batch of texts utilizing Hugging Face pipeline batching."""
        if not texts:
            return []
        
        # Filter empty texts to avoid pipeline errors
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
                
        # Preallocate empty results for all texts
        out: List[List[EntityResult]] = [[] for _ in texts]
        
        if not valid_texts:
            return out
            
        pipe = self._load_pipeline()
        try:
            # Send batch directly to pipeline
            raw_results = pipe(valid_texts, batch_size=batch_size)
            
            for idx, raw in zip(valid_indices, raw_results):
                cleaned = self._clean_entities(raw)
                entity_results = []
                for ent in cleaned:
                    entity_results.append(EntityResult(
                        word=ent.get("word", ""),
                        entity_group=ent.get("entity_group", "MISC"),
                        score=float(ent.get("score", 0.0)),
                        start=int(ent.get("start", 0)),
                        end=int(ent.get("end", 0)),
                    ))
                out[idx] = entity_results
        except Exception as e:
            print(f"[NEREngine] Batch processing error: {e}")
            # Fallback to single text processing if pipeline batch fails
            for idx, text in zip(valid_indices, valid_texts):
                out[idx] = self.recognize(text)
                
        return out
