"""
NER Engine — Named Entity Recognition using HuggingFace Transformers.
Wraps the Nomio4640/ner-mongolian fine-tuned model.

Long-text handling:
  BERT has a 512-token hard limit. Long social-media posts (especially
  Google reviews, long Facebook posts) are silently truncated, causing
  entities in the second half to be completely missed.

  Fix: texts longer than MAX_CHUNK_CHARS are split at sentence boundaries
  into overlapping chunks. Each chunk is processed independently and the
  character offsets from each chunk are corrected before merging. Duplicate
  entities at chunk boundaries are deduplicated by (word, start) key.
"""

from typing import List, Tuple
from .models import EntityResult


HF_MODEL_ID = "Nomio4640/ner-mongolian"

# ~400-450 Mongolian Cyrillic tokens ≈ 1 200-1 500 characters.
# Keeping well below 512 BERT tokens leaves room for tokenizer overhead.
MAX_CHUNK_CHARS = 1_300


class NEREngine:
    """Named Entity Recognition service using HuggingFace pipeline."""

    def __init__(self, model_name: str = None):
        import os
        # Use local model if it exists, otherwise fall back to HuggingFace Hub
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "adapters", "ner_mongolian")
        if model_name:
            self.model_name = model_name
        elif os.path.exists(os.path.join(local_path, "model.safetensors")):
            self.model_name = local_path
        else:
            self.model_name = HF_MODEL_ID
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

    # ------------------------------------------------------------------
    # Long-text chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[Tuple[str, int]]:
        """
        Split *text* into chunks of at most *max_chars* characters, breaking
        at sentence boundaries where possible.  Returns a list of
        (chunk_text, start_char_offset_in_original) tuples.
        """
        chunks: List[Tuple[str, int]] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + max_chars, n)
            if end < n:
                # Try to break at a sentence boundary within the window
                for sep in (". ", "! ", "? ", "\n", " "):
                    pos = text.rfind(sep, start + max_chars // 2, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, start))
            start = end
        return chunks or [(text, 0)]

    def _recognize_chunked(self, text: str) -> List[EntityResult]:
        """
        Run NER on *text* by splitting it into chunks, correcting entity
        character offsets back to the original text's coordinate space,
        and deduplicating entities that appear at chunk boundaries.
        """
        pipe = self._load_pipeline()
        chunks = self._chunk_text(text)
        all_results: List[EntityResult] = []
        seen: set = set()          # (word_lower, abs_start) dedup key

        for chunk_text, chunk_offset in chunks:
            if not chunk_text.strip():
                continue
            try:
                raw = pipe(chunk_text)
            except Exception:
                continue
            for ent in self._clean_entities(raw):
                word = ent.get("word", "")
                abs_start = chunk_offset + int(ent.get("start", 0))
                abs_end   = chunk_offset + int(ent.get("end", 0))
                key = (word.lower(), abs_start)
                if key in seen:
                    continue
                seen.add(key)
                all_results.append(EntityResult(
                    word=word,
                    entity_group=ent.get("entity_group", "MISC"),
                    score=float(ent.get("score", 0.0)),
                    start=abs_start,
                    end=abs_end,
                ))

        return all_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, text: str) -> List[EntityResult]:
        """
        Run NER on a single text and return cleaned entities.
        Automatically chunks texts longer than MAX_CHUNK_CHARS so that
        entities in the second half of long documents are not silently
        dropped by BERT's 512-token truncation.
        """
        if not text or not text.strip():
            return []

        # Long text → chunk-and-merge instead of letting BERT truncate
        if len(text) > MAX_CHUNK_CHARS:
            return self._recognize_chunked(text)

        pipe = self._load_pipeline()
        try:
            raw = pipe(text)
        except Exception:
            return []

        results = []
        for ent in self._clean_entities(raw):
            results.append(EntityResult(
                word=ent.get("word", ""),
                entity_group=ent.get("entity_group", "MISC"),
                score=float(ent.get("score", 0.0)),
                start=int(ent.get("start", 0)),
                end=int(ent.get("end", 0)),
            ))
        return results

    def recognize_batch(self, texts: List[str], batch_size: int = 16) -> List[List[EntityResult]]:
        """
        Run NER on a batch of texts.

        Short texts (≤ MAX_CHUNK_CHARS) are processed together via HuggingFace
        pipeline batching for GPU efficiency.  Long texts are handled
        individually with chunk-and-merge so that no entities are missed.
        """
        if not texts:
            return []

        out: List[List[EntityResult]] = [[] for _ in texts]

        # Separate short and long texts
        short_texts:  List[str] = []
        short_indices: List[int] = []
        long_indices:  List[int] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            if len(text) > MAX_CHUNK_CHARS:
                long_indices.append(i)
            else:
                short_texts.append(text)
                short_indices.append(i)

        # --- Batch-process short texts ---
        if short_texts:
            pipe = self._load_pipeline()
            try:
                raw_results = pipe(short_texts, batch_size=batch_size)
                for idx, raw in zip(short_indices, raw_results):
                    entity_results = []
                    for ent in self._clean_entities(raw):
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
                # Fallback to per-text processing
                for idx, text in zip(short_indices, short_texts):
                    out[idx] = self.recognize(text)

        # --- Chunk-and-merge long texts (sequential, no truncation) ---
        for idx in long_indices:
            out[idx] = self._recognize_chunked(texts[idx])

        return out
