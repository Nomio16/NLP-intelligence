"""
Admin router — CRUD for knowledge base entries, custom labels, and stopwords.

Key change from original:
  Uses services.kb (the shared KnowledgeBase singleton) instead of creating
  its own private _kb instance. This is the critical fix: when the admin
  adds a stopword here, analysis.py's preprocessor can be reloaded from the
  same DB connection via POST /reload.

  Original had:
      _kb = None
      def get_kb():
          global _kb
          if _kb is None:
              _kb = KnowledgeBase(db_path=DB_PATH)
          return _kb
  This is a different object from analysis.py's preprocessor — they don't
  share state. Changes made here were invisible to the analysis pipeline.
"""

from typing import List
from fastapi import APIRouter, HTTPException
from adapters.api.schemas import (
    KnowledgeEntryRequest, KnowledgeEntryResponse,
    LabelRequest, StopwordRequest,
)
from adapters.api import services          # shared KB singleton
from nlp_core.models import KnowledgeEntry

router = APIRouter()

# Convenience alias — same instance as analysis.py uses
kb = services.kb


# ---------------------------------------------------------------------------
# Knowledge entries
# ---------------------------------------------------------------------------

@router.get("/knowledge", response_model=List[KnowledgeEntryResponse])
async def list_entries(category: str = None):
    """List all knowledge base entries, optionally filtered by category."""
    entries = kb.get_entries(category=category)
    return [_entry_to_response(e) for e in entries]


@router.post("/knowledge", response_model=KnowledgeEntryResponse)
async def create_entry(request: KnowledgeEntryRequest):
    """Add a new knowledge base entry."""
    entry = KnowledgeEntry(
        word=request.word,
        category=request.category,
        entity_type=request.entity_type,
        synonyms=request.synonyms or [],
    )
    entry_id = kb.add_entry(entry)
    return KnowledgeEntryResponse(
        id=entry_id, word=entry.word, category=entry.category,
        entity_type=entry.entity_type, synonyms=entry.synonyms,
    )


@router.put("/knowledge/{entry_id}")
async def update_entry(entry_id: int, request: KnowledgeEntryRequest):
    """Update an existing knowledge base entry."""
    entry = KnowledgeEntry(
        word=request.word,
        category=request.category,
        entity_type=request.entity_type,
        synonyms=request.synonyms or [],
    )
    ok = kb.update_entry(entry_id, entry)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")
    return {"status": "updated", "id": entry_id}


@router.delete("/knowledge/{entry_id}")
async def delete_entry(entry_id: int):
    """Delete a knowledge base entry."""
    kb.delete_entry(entry_id)
    return {"status": "deleted", "id": entry_id}


@router.get("/knowledge/categories", response_model=List[str])
async def list_categories():
    """List all distinct category values in the knowledge base."""
    return kb.get_categories()


# ---------------------------------------------------------------------------
# Custom labels
# ---------------------------------------------------------------------------

@router.get("/labels")
async def list_labels(label_type: str = "entity"):
    """
    Return all custom label mappings as a dict: {original_label: custom_label}.

    Example response:
        {"PER": "Улс төрч", "LOC": "Байршил"}

    These are applied to NER output in analysis.py so the frontend
    shows human-readable Mongolian labels instead of PER/LOC/ORG.
    """
    return kb.get_labels(label_type=label_type)


@router.post("/labels")
async def create_label(request: LabelRequest):
    """
    Create or update a custom label mapping.

    After saving, call POST /reload so analysis.py picks up the new mapping.
    """
    kb.set_label(
        request.original_label,
        request.custom_label,
        request.label_type,
    )
    return {
        "status": "created",
        "original": request.original_label,
        "custom": request.custom_label,
    }


@router.delete("/labels/{label_id}")
async def delete_label(label_id: int):
    """Delete a custom label by its DB id."""
    kb.delete_label(label_id)
    return {"status": "deleted", "id": label_id}


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

@router.get("/stopwords", response_model=List[str])
async def list_stopwords():
    """
    List all custom stopwords saved by the admin.

    These are in ADDITION to the hardcoded MONGOLIAN_STOPWORDS in
    preprocessing.py. They take effect after POST /reload is called.
    """
    return kb.get_stopwords()


@router.post("/stopwords")
async def add_stopword(request: StopwordRequest):
    """
    Add a custom stopword.

    After saving, call POST /reload so the preprocessor picks it up.
    Topic modeling will then exclude this word from topic vocabulary.
    """
    kb.add_stopword(request.word.lower().strip())
    return {"status": "added", "word": request.word}


@router.delete("/stopwords/{word}")
async def delete_stopword(word: str):
    """Remove a custom stopword."""
    kb.delete_stopword(word)
    return {"status": "deleted", "word": word}


# ---------------------------------------------------------------------------
# Reload trigger
# ---------------------------------------------------------------------------

@router.post("/reload")
async def reload():
    """
    Apply admin changes to the live preprocessor without restarting.

    Call this from the Admin frontend after:
      - Adding or removing custom stopwords
      - Adding or updating custom entity labels

    Returns the count of currently active custom stopwords so the UI
    can confirm the reload worked.
    """
    services.reload_preprocessor()
    return {
        "status": "reloaded",
        "custom_stopword_count": len(kb.get_stopwords()),
        "custom_label_count": len(kb.get_labels()),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry_to_response(e: KnowledgeEntry) -> KnowledgeEntryResponse:
    return KnowledgeEntryResponse(
        id=e.id,
        word=e.word,
        category=e.category,
        entity_type=e.entity_type,
        synonyms=e.synonyms,
    )