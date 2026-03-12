"""
Topic Modeler — BERTopic wrapper for topic discovery.
Uses sentence-transformers for embeddings + HDBSCAN clustering.
"""

from typing import List, Dict, Tuple, Optional
from .models import TopicResult


class TopicModeler:
    """Topic modeling service using BERTopic."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        language: str = "multilingual",
    ):
        self.embedding_model_name = embedding_model
        self.language = language
        self._model = None
        self._embedding_model = None

    def _load_models(self):
        """Lazy-load BERTopic and the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from bertopic import BERTopic

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self._model = BERTopic(
                language=self.language,
                embedding_model=self._embedding_model,
            )
        return self._model, self._embedding_model

    def fit_transform(self, texts: List[str]) -> Tuple[List[TopicResult], Dict]:
        """
        Fit topic model on texts and return topic assignments.
        Returns: (list of TopicResult per document, topic_info dict)
        """
        if not texts:
            return [], {}

        model, emb_model = self._load_models()

        # Generate embeddings
        embeddings = emb_model.encode(texts, show_progress_bar=True)

        # Fit BERTopic
        topics, probs = model.fit_transform(texts, embeddings)

        # Get topic info
        topic_info = model.get_topic_info()

        # Build results
        results = []
        for i, (topic_id, prob) in enumerate(zip(topics, probs)):
            topic_words = model.get_topic(topic_id)
            keywords = [w for w, _ in topic_words[:5]] if topic_words else []

            # Try to get a readable label
            topic_row = topic_info[topic_info["Topic"] == topic_id]
            if not topic_row.empty and "Name" in topic_row.columns:
                label = str(topic_row.iloc[0]["Name"])
            else:
                label = f"Topic {topic_id}"

            results.append(TopicResult(
                topic_id=int(topic_id),
                topic_label=label,
                probability=float(prob) if isinstance(prob, (int, float)) else 0.0,
                keywords=keywords,
            ))

        # Build summary
        summary = []
        for _, row in topic_info.iterrows():
            summary.append({
                "topic_id": int(row["Topic"]),
                "name": str(row.get("Name", f"Topic {row['Topic']}")),
                "count": int(row["Count"]),
            })

        return results, summary

    def get_topic_info(self) -> List[Dict]:
        """Get topic summary if model has been fitted."""
        if self._model is None:
            return []
        info = self._model.get_topic_info()
        return [
            {
                "topic_id": int(row["Topic"]),
                "name": str(row.get("Name", f"Topic {row['Topic']}")),
                "count": int(row["Count"]),
            }
            for _, row in info.iterrows()
        ]
