"""
topic_modeler.py — BERTopic wrapper with small-dataset fallback.

Problem with the original:
  BERTopic uses HDBSCAN for clustering. HDBSCAN has a min_cluster_size
  parameter that defaults to 10 in BERTopic — meaning it needs at least
  10 documents just to form a single cluster. Below that, EVERYTHING gets
  assigned topic -1 (outlier) and the result is completely empty.

Solution:
  For small datasets (MIN_BERTOPIC_DOCS = 10+): use BERTopic as normal.
  For tiny datasets (MIN_TINY_DOCS = 3+): use KMeans clustering inside
  BERTopic. KMeans always assigns every point to a cluster (no outliers),
  and it works with as few as 2-3 documents.

  The number of clusters (topics) is automatically chosen as:
    n_clusters = max(2, min(n_docs // 2, MAX_TINY_TOPICS))
  So 3 docs → 2 topics, 6 docs → 3 topics, 8 docs → 4 topics.

Below MIN_TINY_DOCS (3): return empty — can't cluster 1-2 texts meaningfully.
"""

from typing import List, Dict, Tuple
from .models import TopicResult

# ---------------------------------------------------------------------------
# Mongolian suffix stripping for c-TF-IDF keyword extraction
# ---------------------------------------------------------------------------
# BERTopic uses CountVectorizer + c-TF-IDF to label each topic cluster.
# Without this, agglutinated forms fragment a single concept into many
# low-frequency tokens:  монголын / монголд / монголаас → 3 keywords
# With this tokenizer they all reduce to монгол → 1 keyword, higher weight.
#
# Rules are ordered longest-first so a longer suffix is tried before a
# shorter one that is a suffix of it (e.g. "аас" before "ас").
# Root must be ≥ 3 characters after stripping to avoid destroying short words.

_MN_SUFFIXES = [
    # Ablative (longest first to avoid partial matches)
    "аас", "ээс", "оос", "өөс",
    # Genitive
    "ийн", "ын", "ний",
    # Comitative
    "тай", "тэй", "той",
    # Directive
    "руу", "рүү",
    # Plural
    "ууд", "үүд",
    # Accusative
    "ийг", "ыг",
    # Dative (single char — checked last so longer suffixes win)
    "д", "т",
]
_MIN_ROOT = 3  # don't strip if remaining root would be shorter than this


def _mn_stem(word: str) -> str:
    for sfx in _MN_SUFFIXES:
        if word.endswith(sfx) and len(word) - len(sfx) >= _MIN_ROOT:
            return word[: -len(sfx)]
    return word


def _mongolian_tokenizer(text: str) -> List[str]:
    """Tokenize and stem Mongolian text for BERTopic's c-TF-IDF step."""
    return [_mn_stem(w) for w in text.split() if w]

# Thresholds
MIN_TINY_DOCS = 3       # minimum to attempt topic modeling at all
MIN_BERTOPIC_DOCS = 10  # minimum to use BERTopic's default HDBSCAN
MAX_TINY_TOPICS = 5     # cap for KMeans cluster count on small datasets


class TopicModeler:
    """Topic modeling service using BERTopic with small-dataset fallback."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        language: str = "multilingual",
    ):
        self.embedding_model_name = embedding_model
        self.language = language
        self._embedding_model = None
        self._model = None             # last fitted BERTopic model

    def _load_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _make_bertopic(self, n_docs: int):
        """
        Build a BERTopic instance appropriate for the dataset size.

        For n_docs >= MIN_BERTOPIC_DOCS: standard BERTopic with HDBSCAN.
        For n_docs < MIN_BERTOPIC_DOCS: BERTopic with KMeans so every
          document gets a real topic assignment instead of -1.
        """
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(tokenizer=_mongolian_tokenizer, min_df=1)

        if n_docs >= MIN_BERTOPIC_DOCS:
            # Standard BERTopic — HDBSCAN with min_cluster_size tuned to
            # dataset size so it's not too greedy on medium datasets
            from hdbscan import HDBSCAN
            cluster_model = HDBSCAN(
                min_cluster_size=max(2, n_docs // 5),
                min_samples=1,
                prediction_data=True,
            )
            return BERTopic(
                language=self.language,
                embedding_model=self._load_embedding_model(),
                hdbscan_model=cluster_model,
                vectorizer_model=vectorizer,
                min_topic_size=2,
            )
        else:
            # Small dataset: KMeans always assigns every point to a cluster.
            # n_clusters chosen proportionally but capped at MAX_TINY_TOPICS.
            from sklearn.cluster import KMeans
            n_clusters = max(2, min(n_docs // 2, MAX_TINY_TOPICS))
            cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            return BERTopic(
                language=self.language,
                embedding_model=self._load_embedding_model(),
                hdbscan_model=cluster_model,
                vectorizer_model=vectorizer,
                min_topic_size=1,       # allow single-doc topics on tiny sets
                nr_topics="auto",
            )

    def fit_transform(self, texts: List[str]) -> Tuple[List[TopicResult], List[Dict]]:
        """
        Fit topic model on texts and return per-document topic assignments.

        Thresholds:
          < MIN_TINY_DOCS (3): returns empty — not enough data
          3 to 9 docs:         KMeans-backed BERTopic
          10+ docs:            standard HDBSCAN BERTopic

        Returns:
            (topic_results, topic_summary)
            topic_results — one TopicResult per input document
            topic_summary — list of {topic_id, name, count} dicts
        """
        # Filter empty strings — they confuse the embedding model
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]

        if len(non_empty) < MIN_TINY_DOCS:
            return [], [{
                "info": (
                    f"Topic modeling needs at least {MIN_TINY_DOCS} non-empty documents. "
                    f"Got {len(non_empty)}."
                )
            }]

        indices, valid_texts = zip(*non_empty)

        emb_model = self._load_embedding_model()
        embeddings = emb_model.encode(list(valid_texts), show_progress_bar=False)

        model = self._make_bertopic(len(valid_texts))
        topics, probs = model.fit_transform(list(valid_texts), embeddings)
        self._model = model

        topic_info = model.get_topic_info()

        # Build per-document results
        # Map from valid_texts index back to original texts index
        result_map: Dict[int, TopicResult] = {}
        for pos, (orig_idx, topic_id) in enumerate(zip(indices, topics)):
            prob = probs[pos] if hasattr(probs[pos], '__float__') else 0.0
            topic_words = model.get_topic(topic_id)
            keywords = [w for w, _ in topic_words[:5]] if topic_words else []

            topic_row = topic_info[topic_info["Topic"] == topic_id]
            if not topic_row.empty and "Name" in topic_row.columns:
                label = str(topic_row.iloc[0]["Name"])
            else:
                label = f"Topic {topic_id}" if topic_id != -1 else "Outlier"

            result_map[orig_idx] = TopicResult(
                topic_id=int(topic_id),
                topic_label=label,
                probability=float(prob),
                keywords=keywords,
            )

        # Fill results list aligned to original texts list
        # Documents that were empty strings get topic_id=-1
        results = []
        for i in range(len(texts)):
            if i in result_map:
                results.append(result_map[i])
            else:
                results.append(TopicResult(
                    topic_id=-1, topic_label="Empty", probability=0.0, keywords=[]
                ))

        # Build summary (exclude outlier topic -1 from summary)
        summary = []
        for _, row in topic_info.iterrows():
            tid = int(row["Topic"])
            summary.append({
                "topic_id": tid,
                "name": str(row.get("Name", f"Topic {tid}")),
                "count": int(row["Count"]),
            })

        return results, summary

    def get_topic_info(self) -> List[Dict]:
        """Return topic summary from the last fitted model."""
        if self._model is None:
            return []
        return [
            {
                "topic_id": int(row["Topic"]),
                "name": str(row.get("Name", f"Topic {row['Topic']}")),
                "count": int(row["Count"]),
            }
            for _, row in self._model.get_topic_info().iterrows()
        ]