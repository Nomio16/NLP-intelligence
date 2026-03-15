"""
Pydantic schemas for API request/response validation.
These are adapter-level models (NOT domain models).
"""

from pydantic import BaseModel
from typing import List, Dict, Optional


# --- Request Models ---

class TextAnalysisRequest(BaseModel):
    text: str


class BatchAnalysisRequest(BaseModel):
    texts: List[str]
    run_ner: bool = True
    run_sentiment: bool = True
    run_topics: bool = False


# --- Response Models ---

class EntityResponse(BaseModel):
    word: str
    entity_group: str
    score: float = 0.0
    start: Optional[int] = None
    end: Optional[int] = None


class SentimentResponse(BaseModel):
    label: str
    score: float = 0.0


class TopicResponse(BaseModel):
    topic_id: int
    topic_label: str
    probability: float = 0.0
    keywords: List[str] = []


class DocumentResponse(BaseModel):
    id: str
    doc_id: Optional[int] = None  # DB row id for annotation editing
    text: str
    clean_text: str = ""
    source: str = ""
    entities: List[EntityResponse] = []
    topic: Optional[TopicResponse] = None
    sentiment: Optional[SentimentResponse] = None


class DocumentUpdateRequest(BaseModel):
    entities: List[EntityResponse] = []
    sentiment_label: str = ""
    sentiment_score: float = 0.0


class NetworkNodeResponse(BaseModel):
    id: str
    label: str
    entity_type: str
    frequency: int = 1


class NetworkEdgeResponse(BaseModel):
    source: str
    target: str
    weight: int = 1


class NetworkResponse(BaseModel):
    nodes: List[NetworkNodeResponse] = []
    edges: List[NetworkEdgeResponse] = []


class AnalysisResponse(BaseModel):
    documents: List[DocumentResponse] = []
    network: Optional[NetworkResponse] = None
    topic_summary: List[Dict] = []
    sentiment_summary: Dict[str, int] = {}
    entity_summary: Dict[str, List[Dict]] = {}
    total_documents: int = 0


class InsightResponse(BaseModel):
    category: str
    title: str
    description: str
    count: int = 0
    sample_texts: List[str] = []


# --- Admin Models ---

class KnowledgeEntryRequest(BaseModel):
    word: str
    category: str = ""
    entity_type: str = ""
    synonyms: List[str] = []


class KnowledgeEntryResponse(BaseModel):
    id: int
    word: str
    category: str = ""
    entity_type: str = ""
    synonyms: List[str] = []


class LabelRequest(BaseModel):
    original_label: str
    custom_label: str
    label_type: str = "entity"


class StopwordRequest(BaseModel):
    word: str
