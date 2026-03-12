"""
Domain models — plain Python dataclasses (no Pydantic, no framework deps).
These represent the core data structures used across all NLP services.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class EntityResult:
    """A single named entity found in text."""
    word: str
    entity_group: str  # PER, ORG, LOC, MISC
    score: float = 0.0
    start: int = 0
    end: int = 0


@dataclass
class TopicResult:
    """A topic assignment for a document."""
    topic_id: int
    topic_label: str
    probability: float = 0.0
    keywords: List[str] = field(default_factory=list)


@dataclass
class SentimentResult:
    """Sentiment classification for a text."""
    label: str  # positive, neutral, negative
    score: float = 0.0


@dataclass
class NetworkNode:
    """A node in the entity co-occurrence network."""
    id: str
    label: str
    entity_type: str  # PER, ORG, LOC
    frequency: int = 1


@dataclass
class NetworkEdge:
    """An edge (co-occurrence) between two entities."""
    source: str
    target: str
    weight: int = 1


@dataclass
class NetworkData:
    """Full network graph data for frontend rendering."""
    nodes: List[NetworkNode] = field(default_factory=list)
    edges: List[NetworkEdge] = field(default_factory=list)


@dataclass
class DocumentResult:
    """Analysis results for a single document/post."""
    id: str
    text: str
    clean_text: str = ""
    source: str = ""
    entities: List[EntityResult] = field(default_factory=list)
    topic: Optional[TopicResult] = None
    sentiment: Optional[SentimentResult] = None


@dataclass
class AnalysisResult:
    """Full analysis results for an entire dataset."""
    documents: List[DocumentResult] = field(default_factory=list)
    network: Optional[NetworkData] = None
    topic_summary: List[Dict] = field(default_factory=list)
    sentiment_summary: Dict[str, int] = field(default_factory=dict)
    entity_summary: Dict[str, List[Dict]] = field(default_factory=dict)
    total_documents: int = 0


@dataclass
class InsightItem:
    """A single insight extracted from the analysis."""
    category: str  # complaint, compliment, hot_issue, important
    title: str
    description: str
    count: int = 0
    sample_texts: List[str] = field(default_factory=list)


@dataclass
class KnowledgeEntry:
    """An entry in the admin knowledge base."""
    id: Optional[int] = None
    word: str = ""
    category: str = ""
    entity_type: str = ""
    synonyms: List[str] = field(default_factory=list)
