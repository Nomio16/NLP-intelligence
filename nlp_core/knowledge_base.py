"""
knowledge_base.py — SQLite-backed storage for admin knowledge tree
                     AND persistent analysis result history.

Changes from original:
  1. seed_stopwords(words) — call this once at startup to populate the
     stopwords table with all the hardcoded MONGOLIAN_STOPWORDS so the
     admin can see and edit them in the UI.

  2. Two new tables:
       analysis_sessions  — one row per upload/analysis run (summary)
       analysis_documents — one row per document in that run

     This means analysis results survive server restarts and you can
     browse history from the admin panel.

  3. save_analysis() / get_analysis() / list_analyses() — public API
     for the new persistence layer.

  4. db_stats() — returns table row counts so the /api/admin/db-stats
     endpoint can show a quick health check.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from .models import KnowledgeEntry


class KnowledgeBase:
    """SQLite-backed knowledge base + analysis history store."""

    def __init__(self, db_path: str = "knowledge.db"):
        self.db_path = db_path
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode — better for concurrent reads (FastAPI async)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self):
        """Create all tables on first run. Safe to call repeatedly (IF NOT EXISTS)."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                -- Admin knowledge tree
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    word        TEXT NOT NULL,
                    category    TEXT DEFAULT '',
                    entity_type TEXT DEFAULT '',
                    synonyms    TEXT DEFAULT '[]',
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Admin custom label mappings (e.g. PER -> "Улс төрч")
                CREATE TABLE IF NOT EXISTS custom_labels (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_label TEXT NOT NULL,
                    custom_label   TEXT NOT NULL,
                    label_type     TEXT DEFAULT 'entity',
                    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(original_label, label_type)
                );

                -- Stopwords — ALL stopwords live here (seeded from hardcoded list
                -- on first startup, then admin can add/remove freely)
                CREATE TABLE IF NOT EXISTS stopwords (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    word       TEXT UNIQUE NOT NULL,
                    is_default INTEGER DEFAULT 0,  -- 1 = seeded from code, 0 = added by admin
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Analysis history — one row per upload/analysis run
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_filename  TEXT DEFAULT '',
                    total_documents  INTEGER DEFAULT 0,
                    sentiment_summary TEXT DEFAULT '{}',
                    entity_summary   TEXT DEFAULT '{}',
                    topic_summary    TEXT DEFAULT '[]'
                );

                -- Per-document results linked to a session
                CREATE TABLE IF NOT EXISTS analysis_documents (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  INTEGER NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
                    doc_index   INTEGER NOT NULL,
                    raw_text    TEXT DEFAULT '',
                    nlp_text    TEXT DEFAULT '',
                    source      TEXT DEFAULT '',
                    sentiment_label TEXT DEFAULT '',
                    sentiment_score REAL DEFAULT 0.0,
                    entities    TEXT DEFAULT '[]',
                    topic_id    INTEGER DEFAULT -1,
                    topic_label TEXT DEFAULT '',
                    topic_keywords TEXT DEFAULT '[]'
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_knowledge_word     ON knowledge_entries(word);
                CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_entries(category);
                CREATE INDEX IF NOT EXISTS idx_docs_session       ON analysis_documents(session_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_created   ON analysis_sessions(created_at);
            """)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Stopword seeding
    # ------------------------------------------------------------------

    def seed_stopwords(self, words: List[str]) -> int:
        """
        Populate the stopwords table from the hardcoded MONGOLIAN_STOPWORDS set.

        Call this once at server startup (services.py). Uses INSERT OR IGNORE
        so it's safe to call every restart — won't duplicate existing words.
        Returns the count of newly inserted words.

        is_default=1 marks these as system defaults. The admin UI can
        optionally show them differently (e.g. greyed out, not deletable).
        """
        conn = self._get_conn()
        try:
            before = conn.execute("SELECT COUNT(*) FROM stopwords").fetchone()[0]
            conn.executemany(
                "INSERT OR IGNORE INTO stopwords (word, is_default) VALUES (?, 1)",
                [(w.lower().strip(),) for w in words if w.strip()],
            )
            conn.commit()
            after = conn.execute("SELECT COUNT(*) FROM stopwords").fetchone()[0]
            return after - before
        finally:
            conn.close()

    def get_stopwords(self) -> List[str]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT word FROM stopwords ORDER BY word").fetchall()
            return [r["word"] for r in rows]
        finally:
            conn.close()

    def add_stopword(self, word: str) -> bool:
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO stopwords (word, is_default) VALUES (?, 0)",
                (word.lower().strip(),),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def delete_stopword(self, word: str) -> bool:
        """Delete a stopword. Default (seeded) stopwords can also be deleted."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM stopwords WHERE word = ?", (word.lower().strip(),))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_stopwords_with_meta(self) -> List[Dict]:
        """Return stopwords with is_default flag — useful for admin UI display."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT id, word, is_default, created_at FROM stopwords ORDER BY word"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Analysis persistence
    # ------------------------------------------------------------------

    def save_analysis(
        self,
        documents: List[Dict],
        sentiment_summary: Dict,
        entity_summary: Dict,
        topic_summary: List,
        source_filename: str = "",
    ) -> tuple:
        """
        Persist a full analysis run to the DB.

        Args:
            documents: list of dicts with keys: raw_text, nlp_text, source,
                       sentiment_label, sentiment_score, entities (list),
                       topic_id, topic_label, topic_keywords (list)
            sentiment_summary: {"positive": N, "neutral": N, "negative": N}
            entity_summary:    {"PER": [...], "LOC": [...], ...}
            topic_summary:     list of topic dicts from BERTopic
            source_filename:   original CSV filename if applicable

        Returns:
            (session_id, doc_ids) — session_id for the session, doc_ids list
            of DB ids for each inserted document (in order).
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO analysis_sessions
                   (source_filename, total_documents, sentiment_summary,
                    entity_summary, topic_summary)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    source_filename,
                    len(documents),
                    json.dumps(sentiment_summary, ensure_ascii=False),
                    json.dumps(entity_summary, ensure_ascii=False),
                    json.dumps(topic_summary, ensure_ascii=False),
                ),
            )
            session_id = cursor.lastrowid

            doc_ids = []
            for i, d in enumerate(documents):
                c = conn.execute(
                    """INSERT INTO analysis_documents
                       (session_id, doc_index, raw_text, nlp_text, source,
                        sentiment_label, sentiment_score, entities,
                        topic_id, topic_label, topic_keywords)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        i,
                        d.get("raw_text", ""),
                        d.get("nlp_text", ""),
                        d.get("source", ""),
                        d.get("sentiment_label", ""),
                        float(d.get("sentiment_score", 0.0)),
                        json.dumps(d.get("entities", []), ensure_ascii=False),
                        int(d.get("topic_id", -1)),
                        d.get("topic_label", ""),
                        json.dumps(d.get("topic_keywords", []), ensure_ascii=False),
                    ),
                )
                doc_ids.append(c.lastrowid)
            conn.commit()
            return session_id, doc_ids
        finally:
            conn.close()

    def list_analyses(self, limit: int = 20) -> List[Dict]:
        """Return the most recent analysis sessions (summary only, no documents)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT id, created_at, source_filename, total_documents,
                          sentiment_summary, topic_summary
                   FROM analysis_sessions
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            result = []
            for r in rows:
                result.append({
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "source_filename": r["source_filename"],
                    "total_documents": r["total_documents"],
                    "sentiment_summary": json.loads(r["sentiment_summary"]),
                    "topic_summary": json.loads(r["topic_summary"]),
                })
            return result
        finally:
            conn.close()

    def get_analysis(self, session_id: int) -> Optional[Dict]:
        """Return a full analysis session including all documents."""
        conn = self._get_conn()
        try:
            session = conn.execute(
                "SELECT * FROM analysis_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not session:
                return None

            docs = conn.execute(
                """SELECT * FROM analysis_documents
                   WHERE session_id = ? ORDER BY doc_index""",
                (session_id,),
            ).fetchall()

            return {
                "id": session["id"],
                "created_at": session["created_at"],
                "source_filename": session["source_filename"],
                "total_documents": session["total_documents"],
                "sentiment_summary": json.loads(session["sentiment_summary"]),
                "entity_summary": json.loads(session["entity_summary"]),
                "topic_summary": json.loads(session["topic_summary"]),
                "documents": [
                    {
                        "id": d["id"],
                        "doc_index": d["doc_index"],
                        "raw_text": d["raw_text"],
                        "nlp_text": d["nlp_text"],
                        "source": d["source"],
                        "sentiment": {
                            "label": d["sentiment_label"],
                            "score": d["sentiment_score"],
                        },
                        "entities": json.loads(d["entities"]),
                        "topic": {
                            "topic_id": d["topic_id"],
                            "topic_label": d["topic_label"],
                            "keywords": json.loads(d["topic_keywords"]),
                        },
                    }
                    for d in docs
                ],
            }
        finally:
            conn.close()

    def delete_analysis(self, session_id: int) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM analysis_sessions WHERE id = ?", (session_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    def update_document_annotations(
        self,
        doc_id: int,
        entities: list,
        sentiment_label: str,
        sentiment_score: float,
    ) -> bool:
        """Update a single document's entities and sentiment in the DB."""
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE analysis_documents
                   SET entities=?, sentiment_label=?, sentiment_score=?
                   WHERE id=?""",
                (
                    json.dumps(entities, ensure_ascii=False),
                    sentiment_label,
                    float(sentiment_score),
                    doc_id,
                ),
            )
            conn.commit()
            return conn.execute(
                "SELECT changes()"
            ).fetchone()[0] > 0
        finally:
            conn.close()

    def get_all_documents(self) -> List[Dict]:
        """Return all documents across all sessions for global re-analysis."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT id, session_id, doc_index, raw_text, nlp_text,
                          source, sentiment_label, sentiment_score,
                          entities, topic_id, topic_label, topic_keywords
                   FROM analysis_documents
                   ORDER BY session_id, doc_index"""
            ).fetchall()
            result = []
            for d in rows:
                result.append({
                    "id": d["id"],
                    "session_id": d["session_id"],
                    "doc_index": d["doc_index"],
                    "raw_text": d["raw_text"],
                    "nlp_text": d["nlp_text"],
                    "source": d["source"],
                    "sentiment_label": d["sentiment_label"],
                    "sentiment_score": d["sentiment_score"],
                    "entities": json.loads(d["entities"]),
                })
            return result
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Knowledge entries (unchanged from original)
    # ------------------------------------------------------------------

    def add_entry(self, entry: KnowledgeEntry) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO knowledge_entries (word, category, entity_type, synonyms) VALUES (?, ?, ?, ?)",
                (entry.word, entry.category, entry.entity_type, json.dumps(entry.synonyms)),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_entries(self, category: str = None) -> List[KnowledgeEntry]:
        conn = self._get_conn()
        try:
            if category:
                rows = conn.execute(
                    "SELECT * FROM knowledge_entries WHERE category = ? ORDER BY word",
                    (category,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM knowledge_entries ORDER BY category, word"
                ).fetchall()
            return [self._row_to_entry(r) for r in rows]
        finally:
            conn.close()

    def update_entry(self, entry_id: int, entry: KnowledgeEntry) -> bool:
        conn = self._get_conn()
        try:
            conn.execute(
                """UPDATE knowledge_entries
                   SET word=?, category=?, entity_type=?, synonyms=?,
                       updated_at=CURRENT_TIMESTAMP
                   WHERE id=?""",
                (entry.word, entry.category, entry.entity_type,
                 json.dumps(entry.synonyms), entry_id),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def delete_entry(self, entry_id: int) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM knowledge_entries WHERE id = ?", (entry_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_categories(self) -> List[str]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT category FROM knowledge_entries WHERE category != '' ORDER BY category"
            ).fetchall()
            return [r["category"] for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Custom labels (unchanged from original)
    # ------------------------------------------------------------------

    def set_label(self, original: str, custom: str, label_type: str = "entity"):
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO custom_labels (original_label, custom_label, label_type)
                   VALUES (?, ?, ?)
                   ON CONFLICT(original_label, label_type) DO UPDATE SET custom_label=?""",
                (original, custom, label_type, custom),
            )
            conn.commit()
        finally:
            conn.close()

    def get_labels(self, label_type: str = "entity") -> Dict[str, str]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT original_label, custom_label FROM custom_labels WHERE label_type = ?",
                (label_type,),
            ).fetchall()
            return {r["original_label"]: r["custom_label"] for r in rows}
        finally:
            conn.close()

    def delete_label(self, label_id: int) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM custom_labels WHERE id = ?", (label_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # DB stats — for admin health check endpoint
    # ------------------------------------------------------------------

    def db_stats(self) -> Dict[str, Any]:
        """Return row counts for all tables plus the DB file size."""
        conn = self._get_conn()
        try:
            stats = {}
            for table in (
                "knowledge_entries", "custom_labels", "stopwords",
                "analysis_sessions", "analysis_documents",
            ):
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count

            stats["db_path"] = self.db_path
            stats["db_size_kb"] = (
                round(os.path.getsize(self.db_path) / 1024, 1)
                if os.path.exists(self.db_path) else 0
            )
            return stats
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_entry(row) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=row["id"],
            word=row["word"],
            category=row["category"],
            entity_type=row["entity_type"],
            synonyms=json.loads(row["synonyms"]) if row["synonyms"] else [],
        )