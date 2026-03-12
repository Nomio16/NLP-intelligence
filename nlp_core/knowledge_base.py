"""
Knowledge Base — SQLite-backed admin-managed knowledge tree.
Stores entity labels, categories, synonyms for human-in-the-loop control.
"""

import sqlite3
import json
import os
from typing import List, Optional, Dict
from .models import KnowledgeEntry


class KnowledgeBase:
    """SQLite-backed knowledge base for admin management."""

    def __init__(self, db_path: str = "knowledge.db"):
        self.db_path = db_path
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    category TEXT DEFAULT '',
                    entity_type TEXT DEFAULT '',
                    synonyms TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS custom_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_label TEXT NOT NULL,
                    custom_label TEXT NOT NULL,
                    label_type TEXT DEFAULT 'entity',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS stopwords (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_knowledge_word ON knowledge_entries(word);
                CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_entries(category);
            """)
            conn.commit()
        finally:
            conn.close()

    # --- Knowledge Entries ---

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
                    "SELECT * FROM knowledge_entries WHERE category = ? ORDER BY word", (category,)
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
                   SET word=?, category=?, entity_type=?, synonyms=?, updated_at=CURRENT_TIMESTAMP 
                   WHERE id=?""",
                (entry.word, entry.category, entry.entity_type, json.dumps(entry.synonyms), entry_id),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def delete_entry(self, entry_id: int) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM knowledge_entries WHERE id=?", (entry_id,))
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

    # --- Custom Labels ---

    def set_label(self, original: str, custom: str, label_type: str = "entity"):
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO custom_labels (original_label, custom_label, label_type) 
                   VALUES (?, ?, ?)
                   ON CONFLICT DO UPDATE SET custom_label=?""",
                (original, custom, label_type, custom),
            )
            conn.commit()
        finally:
            conn.close()

    def get_labels(self, label_type: str = "entity") -> Dict[str, str]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT original_label, custom_label FROM custom_labels WHERE label_type=?",
                (label_type,),
            ).fetchall()
            return {r["original_label"]: r["custom_label"] for r in rows}
        finally:
            conn.close()

    def delete_label(self, label_id: int) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM custom_labels WHERE id=?", (label_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    # --- Stopwords ---

    def add_stopword(self, word: str) -> bool:
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO stopwords (word) VALUES (?)", (word,)
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def get_stopwords(self) -> List[str]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT word FROM stopwords ORDER BY word").fetchall()
            return [r["word"] for r in rows]
        finally:
            conn.close()

    def delete_stopword(self, word: str) -> bool:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM stopwords WHERE word=?", (word,))
            conn.commit()
            return True
        finally:
            conn.close()

    # --- Helpers ---

    @staticmethod
    def _row_to_entry(row) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=row["id"],
            word=row["word"],
            category=row["category"],
            entity_type=row["entity_type"],
            synonyms=json.loads(row["synonyms"]) if row["synonyms"] else [],
        )
