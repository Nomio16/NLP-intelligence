"""
Analysis router — handles CSV upload, text analysis, and network generation.
"""

import csv
import io
import uuid
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from adapters.api.schemas import (
    TextAnalysisRequest, BatchAnalysisRequest,
    AnalysisResponse, DocumentResponse, EntityResponse,
    SentimentResponse, TopicResponse,
    NetworkResponse, NetworkNodeResponse, NetworkEdgeResponse,
)

# Import domain services
from nlp_core.preprocessing import Preprocessor
from nlp_core.ner_engine import NEREngine
from nlp_core.sentiment import SentimentAnalyzer
from nlp_core.topic_modeler import TopicModeler
from nlp_core.network_analyzer import NetworkAnalyzer

router = APIRouter()

# Lazy-loaded services (initialized on first request)
_preprocessor = None
_ner_engine = None
_sentiment_analyzer = None
_topic_modeler = None
_network_analyzer = None

# In-memory store for last analysis results (for network/insights endpoints)
_last_analysis = None


def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = Preprocessor()
    return _preprocessor


def get_ner():
    global _ner_engine
    if _ner_engine is None:
        _ner_engine = NEREngine()
    return _ner_engine


def get_sentiment():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer


def get_topic_modeler():
    global _topic_modeler
    if _topic_modeler is None:
        _topic_modeler = TopicModeler()
    return _topic_modeler


def get_network_analyzer():
    global _network_analyzer
    if _network_analyzer is None:
        _network_analyzer = NetworkAnalyzer()
    return _network_analyzer


@router.post("/upload", response_model=AnalysisResponse)
async def upload_csv(
    file: UploadFile = File(...),
    run_ner: bool = True,
    run_sentiment: bool = True,
    run_topics: bool = False,
):
    """
    Upload a CSV file for analysis.
    CSV must have at minimum a 'text' or 'Text' column.
    Optional columns: 'ID', 'Source'.
    """
    global _last_analysis

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    text_content = content.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(text_content))
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    # Find the text column
    text_col = None
    for col_name in ["text", "Text", "clean_text", "cleaned_text", "content", "Content", "Message"]:
        if col_name in rows[0]:
            text_col = col_name
            break

    if text_col is None:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must have a 'text' or 'Text' column. Found columns: {list(rows[0].keys())}"
        )

    result = _run_analysis(rows, text_col, run_ner, run_sentiment, run_topics)
    _last_analysis = result
    return result


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze a single text."""
    global _last_analysis

    rows = [{"ID": str(uuid.uuid4())[:8], "Text": request.text, "Source": "direct"}]
    result = _run_analysis(rows, "Text", run_ner=True, run_sentiment=True, run_topics=False)
    _last_analysis = result
    return result


@router.post("/analyze/batch", response_model=AnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """Analyze a batch of texts."""
    global _last_analysis

    rows = [
        {"ID": str(uuid.uuid4())[:8], "Text": t, "Source": "batch"}
        for t in request.texts
    ]
    result = _run_analysis(rows, "Text", request.run_ner, request.run_sentiment, request.run_topics)
    _last_analysis = result
    return result


@router.post("/network", response_model=NetworkResponse)
async def get_network():
    """Get the network graph data from the last analysis."""
    if _last_analysis is None:
        raise HTTPException(status_code=404, detail="No analysis has been run yet. Upload data first.")
    if _last_analysis.network is None:
        raise HTTPException(status_code=404, detail="No network data available.")
    return _last_analysis.network


def _run_analysis(
    rows: List[dict],
    text_col: str,
    run_ner: bool = True,
    run_sentiment: bool = True,
    run_topics: bool = False,
) -> AnalysisResponse:
    """Core analysis pipeline."""
    preprocessor = get_preprocessor()

    documents = []
    all_entities = []
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

    # Extract texts
    texts = [row.get(text_col, "") for row in rows]
    ids = [row.get("ID", str(i)) for i, row in enumerate(rows)]
    sources = [row.get("Source", "") for row in rows]

    # Preprocess
    clean_texts = preprocessor.preprocess_batch(texts, deep_clean=False)

    # NER (if requested)
    ner_results = []
    if run_ner:
        ner_engine = get_ner()
        ner_results = ner_engine.recognize_batch(clean_texts)

    # Sentiment (if requested)
    sentiment_results = []
    if run_sentiment:
        sentiment_analyzer = get_sentiment()
        sentiment_results = sentiment_analyzer.analyze_batch(clean_texts)

    # Topic modeling (if requested)
    topic_results = []
    topic_summary = []
    if run_topics and len(clean_texts) >= 10:
        topic_modeler = get_topic_modeler()
        deep_texts = preprocessor.preprocess_batch(texts, deep_clean=True)
        topic_results, topic_summary = topic_modeler.fit_transform(deep_texts)

    # Build document results
    for i in range(len(texts)):
        entities = []
        if i < len(ner_results):
            entities = [
                EntityResponse(word=e.word, entity_group=e.entity_group, score=e.score)
                for e in ner_results[i]
            ]

        sentiment = None
        if i < len(sentiment_results):
            sr = sentiment_results[i]
            sentiment = SentimentResponse(label=sr.label, score=sr.score)
            sentiment_counts[sr.label] = sentiment_counts.get(sr.label, 0) + 1

        topic = None
        if i < len(topic_results):
            tr = topic_results[i]
            topic = TopicResponse(
                topic_id=tr.topic_id,
                topic_label=tr.topic_label,
                probability=tr.probability,
                keywords=tr.keywords,
            )

        documents.append(DocumentResponse(
            id=str(ids[i]),
            text=texts[i],
            clean_text=clean_texts[i] if i < len(clean_texts) else "",
            source=sources[i] if i < len(sources) else "",
            entities=entities,
            topic=topic,
            sentiment=sentiment,
        ))

    # Build network
    network = None
    if run_ner and ner_results:
        analyzer = get_network_analyzer()
        network_data = analyzer.build_network(ner_results)
        entity_stats = analyzer.get_entity_stats(ner_results)

        network = NetworkResponse(
            nodes=[
                NetworkNodeResponse(
                    id=n.id, label=n.label,
                    entity_type=n.entity_type, frequency=n.frequency,
                )
                for n in network_data.nodes
            ],
            edges=[
                NetworkEdgeResponse(source=e.source, target=e.target, weight=e.weight)
                for e in network_data.edges
            ],
        )
    else:
        entity_stats = {}

    return AnalysisResponse(
        documents=documents,
        network=network,
        topic_summary=topic_summary,
        sentiment_summary=sentiment_counts,
        entity_summary=entity_stats,
        total_documents=len(documents),
    )
