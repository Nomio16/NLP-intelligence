"""
Admin router — CRUD operations for knowledge base, labels, and stopwords.
"""

import os
from typing import List

from fastapi import APIRouter, HTTPException
from adapters.api.schemas import (
    KnowledgeEntryRequest, KnowledgeEntryResponse,
    LabelRequest, StopwordRequest,
)
from nlp_core.knowledge_base import KnowledgeBase
from nlp_core.models import KnowledgeEntry

router = APIRouter()

# Initialize knowledge base with db in the webapp directory
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "knowledge.db")
_kb = None


def get_kb():
    global _kb
    if _kb is None:
        _kb = KnowledgeBase(db_path=DB_PATH)
    return _kb


# --- Knowledge Entries ---

@router.get("/knowledge", response_model=List[KnowledgeEntryResponse])
async def list_entries(category: str = None):
    """List all knowledge base entries, optionally filtered by category."""
    kb = get_kb()
    entries = kb.get_entries(category=category)
    return [
        KnowledgeEntryResponse(
            id=e.id, word=e.word, category=e.category,
            entity_type=e.entity_type, synonyms=e.synonyms,
        )
        for e in entries
    ]


@router.post("/knowledge", response_model=KnowledgeEntryResponse)
async def create_entry(request: KnowledgeEntryRequest):
    """Add a new knowledge base entry."""
    kb = get_kb()
    entry = KnowledgeEntry(
        word=request.word, category=request.category,
        entity_type=request.entity_type, synonyms=request.synonyms,
    )
    entry_id = kb.add_entry(entry)
    return KnowledgeEntryResponse(
        id=entry_id, word=entry.word, category=entry.category,
        entity_type=entry.entity_type, synonyms=entry.synonyms,
    )


@router.put("/knowledge/{entry_id}")
async def update_entry(entry_id: int, request: KnowledgeEntryRequest):
    """Update a knowledge base entry."""
    kb = get_kb()
    entry = KnowledgeEntry(
        word=request.word, category=request.category,
        entity_type=request.entity_type, synonyms=request.synonyms,
    )
    kb.update_entry(entry_id, entry)
    return {"status": "updated", "id": entry_id}


@router.delete("/knowledge/{entry_id}")
async def delete_entry(entry_id: int):
    """Delete a knowledge base entry."""
    kb = get_kb()
    kb.delete_entry(entry_id)
    return {"status": "deleted", "id": entry_id}


@router.get("/knowledge/categories", response_model=List[str])
async def list_categories():
    """List all knowledge base categories."""
    kb = get_kb()
    return kb.get_categories()


# --- Custom Labels ---

@router.get("/labels")
async def list_labels(label_type: str = "entity"):
    """List all custom labels."""
    kb = get_kb()
    return kb.get_labels(label_type=label_type)


@router.post("/labels")
async def create_label(request: LabelRequest):
    """Create or update a custom label mapping."""
    kb = get_kb()
    kb.set_label(request.original_label, request.custom_label, request.label_type)
    return {"status": "created", "original": request.original_label, "custom": request.custom_label}


@router.delete("/labels/{label_id}")
async def delete_label(label_id: int):
    """Delete a custom label."""
    kb = get_kb()
    kb.delete_label(label_id)
    return {"status": "deleted", "id": label_id}


# --- Stopwords ---

@router.get("/stopwords", response_model=List[str])
async def list_stopwords():
    """List all custom stopwords."""
    kb = get_kb()
    return kb.get_stopwords()


@router.post("/stopwords")
async def add_stopword(request: StopwordRequest):
    """Add a custom stopword."""
    kb = get_kb()
    kb.add_stopword(request.word)
    return {"status": "added", "word": request.word}


@router.delete("/stopwords/{word}")
async def delete_stopword(word: str):
    """Delete a custom stopword."""
    kb = get_kb()
    kb.delete_stopword(word)
    return {"status": "deleted", "word": word}
