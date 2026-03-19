"""
services.py — shared singleton NLP services.

Changes from previous version:
  1. Seeds ALL hardcoded MONGOLIAN_STOPWORDS into the DB on startup so
     the admin can see and edit the full list. Uses INSERT OR IGNORE so
     server restarts never create duplicates.
  2. Preprocessor now reads stopwords FROM the DB (so admin additions/
     deletions are always reflected after reload_preprocessor()).
  3. MIN_TOPICS_DOCS lowered to match topic_modeler.py's new threshold of 3.
"""

import os
from typing import Optional

from nlp_core.knowledge_base import KnowledgeBase
from nlp_core.preprocessing import Preprocessor, MONGOLIAN_STOPWORDS
from nlp_core.ner_engine import NEREngine
from nlp_core.sentiment import SentimentAnalyzer
from nlp_core.topic_modeler import TopicModeler
from nlp_core.network_analyzer import NetworkAnalyzer

# ---------------------------------------------------------------------------
# DB path — resolves to webapp/knowledge.db regardless of cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# Check if an external DB_PATH is provided via environment variables (e.g., Colab Google Drive).
# Otherwise, default to the local knowledge.db inside the project folder.
DB_PATH = os.environ.get(
    "DB_PATH", 
    os.path.normpath(os.path.join(_HERE, "..", "..", "knowledge.db"))
)

# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------
kb = KnowledgeBase(db_path=DB_PATH)

# Seed all hardcoded stopwords into the DB on first run.
# INSERT OR IGNORE makes this safe to call on every restart.
_seeded = kb.seed_stopwords(list(MONGOLIAN_STOPWORDS))
if _seeded > 0:
    print(f"[services] Seeded {_seeded} default stopwords into DB.")

# Preprocessor reads stopwords from DB — now includes all defaults
# plus anything the admin has added via the UI.
preprocessor = Preprocessor(extra_stopwords=kb.get_stopwords())

# Heavy ML models — lazy-loaded inside the classes on first actual use.
# Keeping them as module-level objects means HuggingFace pipelines are
# only constructed ONCE per server lifetime, not once per request.
ner       = NEREngine()
sentiment = SentimentAnalyzer()
topic     = TopicModeler()
network   = NetworkAnalyzer()

# ---------------------------------------------------------------------------
# In-memory cache of the last analysis (for /network and /insights)
# ---------------------------------------------------------------------------
_last_analysis = None


def get_last_analysis():
    return _last_analysis


def set_last_analysis(result) -> None:
    global _last_analysis
    _last_analysis = result


# ---------------------------------------------------------------------------
# Reload — called by POST /admin/reload after admin changes
# ---------------------------------------------------------------------------

def reload_preprocessor() -> None:
    """
    Rebuild the Preprocessor with the latest stopword list from the DB.
    Call this after the admin adds or removes stopwords so the change
    takes effect on the next analysis request without restarting.
    """
    global preprocessor
    preprocessor = Preprocessor(extra_stopwords=kb.get_stopwords())