"""
Analysis router — handles CSV upload, text analysis, and network generation.

Changes from previous version:
  1. Every analysis result is now saved to the DB via kb.save_analysis()
     so results persist across server restarts.
  2. MIN_TOPICS_DOCS lowered to 3 to match topic_modeler.py's KMeans fallback.
  3. New endpoints:
       GET  /history          — list past analysis sessions
       GET  /history/{id}     — retrieve a full past session with documents
       DELETE /history/{id}   — delete a session
       GET  /db-stats         — show table row counts + DB file size
"""

import csv
import io
import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from adapters.api.schemas import (
    TextAnalysisRequest, BatchAnalysisRequest,
    AnalysisResponse, DocumentResponse, EntityResponse,
    SentimentResponse, TopicResponse,
    NetworkResponse, NetworkNodeResponse, NetworkEdgeResponse,
    DocumentUpdateRequest,
)
from adapters.api import services
from nlp_core.models import EntityResult

router = APIRouter()

# Minimum docs to attempt topic modeling.
# Matches topic_modeler.py MIN_TINY_DOCS — KMeans fallback handles 3-9 docs,
# standard HDBSCAN BERTopic handles 10+.
MIN_TOPICS_DOCS = 3


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=AnalysisResponse)
async def upload_csv(
    file: UploadFile = File(...),
    run_ner: bool = True,
    run_sentiment: bool = True,
    run_topics: bool = True,
):
    """Upload a CSV file for analysis. Must have a 'text' or 'Text' column."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig", errors="replace"))))

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    text_col = _find_text_column(rows[0])
    if text_col is None:
        raise HTTPException(
            status_code=400,
            detail=f"No text column found. Got columns: {list(rows[0].keys())}",
        )

    result = _run_analysis(rows, text_col, run_ner, run_sentiment, run_topics)
    services.set_last_analysis(result)

    # Persist to DB and attach doc_ids to each document in the response
    result = _save_and_attach_doc_ids(result, source_filename=file.filename)

    return result


# ---------------------------------------------------------------------------
# POST /analyze  (single text)
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze a single text string."""
    rows = [{"ID": str(uuid.uuid4())[:8], "Text": request.text, "Source": "direct"}]
    result = _run_analysis(rows, "Text", run_ner=True, run_sentiment=True, run_topics=False)
    services.set_last_analysis(result)
    result = _save_and_attach_doc_ids(result, source_filename="single-text")
    return result


# ---------------------------------------------------------------------------
# POST /analyze/batch
# ---------------------------------------------------------------------------

@router.post("/analyze/batch", response_model=AnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """Analyze a list of texts. Topics auto-enabled when >= 3 texts."""
    rows = [
        {"ID": str(uuid.uuid4())[:8], "Text": t, "Source": "batch"}
        for t in request.texts
    ]
    result = _run_analysis(
        rows, "Text",
        request.run_ner, request.run_sentiment, request.run_topics,
    )
    services.set_last_analysis(result)
    result = _save_and_attach_doc_ids(result, source_filename="batch")
    return result


# ---------------------------------------------------------------------------
# POST /network
# ---------------------------------------------------------------------------

@router.post("/network", response_model=NetworkResponse)
async def get_network():
    last = services.get_last_analysis()
    if last is None:
        raise HTTPException(status_code=404, detail="No analysis has been run yet.")
    if last.network is None:
        raise HTTPException(status_code=404, detail="No network data available.")
    return last.network


# ---------------------------------------------------------------------------
# POST /reload
# ---------------------------------------------------------------------------

@router.post("/reload")
async def reload():
    """Reload custom stopwords and labels from DB without restarting."""
    services.reload_preprocessor()
    return {
        "status": "reloaded",
        "custom_stopword_count": len(services.kb.get_stopwords()),
    }


# ---------------------------------------------------------------------------
# GET /history  — list past analysis sessions
# ---------------------------------------------------------------------------

@router.get("/history")
async def list_history(limit: int = 20):
    """
    List the most recent analysis sessions stored in the DB.
    Returns summary rows (no per-document detail).
    """
    return services.kb.list_analyses(limit=limit)


# ---------------------------------------------------------------------------
# GET /history/{session_id}  — retrieve a full past session
# ---------------------------------------------------------------------------

@router.get("/history/{session_id}")
async def get_history(session_id: int):
    """
    Retrieve a full analysis session by ID, including all documents.
    Use GET /history to find session IDs.
    """
    session = services.kb.get_analysis(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    return session


# ---------------------------------------------------------------------------
# DELETE /history/{session_id}
# ---------------------------------------------------------------------------

@router.delete("/history/{session_id}")
async def delete_history(session_id: int):
    """Delete a stored analysis session and all its documents."""
    services.kb.delete_analysis(session_id)
    return {"status": "deleted", "session_id": session_id}


# ---------------------------------------------------------------------------
# GET /db-stats  — health check showing DB table sizes
# ---------------------------------------------------------------------------

@router.get("/db-stats")
async def db_stats():
    """
    Show row counts for every table in knowledge.db plus file size.

    Example response:
      {
        "knowledge_entries": 12,
        "custom_labels": 3,
        "stopwords": 87,          ← should be 80+ after seeding
        "analysis_sessions": 5,
        "analysis_documents": 423,
        "db_path": "/home/.../webapp/knowledge.db",
        "db_size_kb": 128.4
      }

    This is the quickest way to confirm the DB is initialised and
    that stopword seeding worked.
    """
    return services.kb.db_stats()


# ---------------------------------------------------------------------------
# PATCH /documents/{doc_id}  — update annotations for a single document
# ---------------------------------------------------------------------------

@router.patch("/documents/{doc_id}")
async def update_document(doc_id: int, body: DocumentUpdateRequest):
    """
    Save edited entities and/or sentiment for a single stored document.
    Called by the inline annotation editor in the frontend.
    """
    entities = [
        {
            "word": e.word,
            "entity_group": e.entity_group,
            "score": e.score,
            "start": e.start,
            "end": e.end,
        }
        for e in body.entities
    ]
    ok = services.kb.update_document_annotations(
        doc_id=doc_id,
        entities=entities,
        sentiment_label=body.sentiment_label,
        sentiment_score=body.sentiment_score,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
    return {"ok": True, "doc_id": doc_id}


# ---------------------------------------------------------------------------
# GET /global-analysis  — run topic modeling + network on ALL stored documents
# ---------------------------------------------------------------------------

@router.get("/global-analysis")
async def global_analysis():
    """
    Run topic modeling and build a co-occurrence network using every document
    stored in the DB across all sessions.  NER and sentiment are NOT re-run —
    the stored results are reused so this is fast.
    """
    docs = services.kb.get_all_documents()
    if len(docs) < MIN_TOPICS_DOCS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Global analysis needs at least {MIN_TOPICS_DOCS} stored documents. "
                f"Currently have {len(docs)}."
            ),
        )

    nlp_texts = [d["nlp_text"] for d in docs]

    # Rebuild EntityResult objects from stored JSON for the network analyzer
    entity_results: List[List[EntityResult]] = []
    for d in docs:
        ents = d["entities"] if isinstance(d["entities"], list) else json.loads(d["entities"])
        entity_results.append([
            EntityResult(
                word=e.get("word", ""),
                entity_group=e.get("entity_group", "MISC"),
                score=float(e.get("score", 0.0)),
                start=int(e.get("start") or 0),
                end=int(e.get("end") or 0),
            )
            for e in ents
        ])

    # Topic modeling across all stored documents
    topic_summary: list = []
    try:
        _, topic_summary = services.topic.fit_transform(nlp_texts)
    except Exception as exc:
        topic_summary = [{"error": f"Topic modeling failed: {exc}"}]

    # Network co-occurrence graph across all stored documents
    nd = services.network.build_network(entity_results)
    network = NetworkResponse(
        nodes=[
            NetworkNodeResponse(
                id=n.id, label=n.label,
                entity_type=n.entity_type, frequency=n.frequency,
            )
            for n in nd.nodes
        ],
        edges=[
            NetworkEdgeResponse(source=e.source, target=e.target, weight=e.weight)
            for e in nd.edges
        ],
    )

    return {
        "total_documents": len(docs),
        "topic_summary": topic_summary,
        "network": network,
    }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _run_analysis(
    rows: List[dict],
    text_col: str,
    run_ner: bool,
    run_sentiment: bool,
    run_topics: bool,
) -> AnalysisResponse:
    preprocessor = services.preprocessor
    kb = services.kb

    raw_texts = [row.get(text_col, "") for row in rows]
    ids       = [row.get("ID", str(i)) for i, row in enumerate(rows)]
    sources   = [row.get("Source", "") for row in rows]

    # Dual preprocessing — one pass, two outputs
    nlp_texts: List[str] = []
    tm_texts:  List[str] = []
    for raw in raw_texts:
        nlp, tm = preprocessor.preprocess_dual(raw)
        nlp_texts.append(nlp)
        tm_texts.append(tm)

    # NER
    ner_results = []
    if run_ner:
        ner_results = services.ner.recognize_batch(nlp_texts)

    # Entity relabeling from admin custom labels
    custom_labels = kb.get_labels(label_type="entity") if run_ner else {}

    # Sentiment
    sentiment_results = []
    if run_sentiment:
        sentiment_results = services.sentiment.analyze_batch(nlp_texts)

    # Topic modeling — now works from 3 documents via KMeans fallback
    topic_results = []
    topic_summary = []
    if run_topics:
        if len(tm_texts) >= MIN_TOPICS_DOCS:
            try:
                topic_results, topic_summary = services.topic.fit_transform(tm_texts)
            except Exception as exc:
                topic_summary = [{"error": f"Topic modeling failed: {exc}"}]
        else:
            topic_summary = [{
                "info": (
                    f"Topic modeling needs at least {MIN_TOPICS_DOCS} documents. "
                    f"Got {len(tm_texts)}."
                )
            }]

    # Assemble per-document results
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    documents: List[DocumentResponse] = []

    for i in range(len(raw_texts)):
        entities: List[EntityResponse] = []
        if i < len(ner_results):
            for e in ner_results[i]:
                label = custom_labels.get(e.entity_group, e.entity_group)
                entities.append(EntityResponse(
                    word=e.word, entity_group=label, score=e.score,
                    start=e.start, end=e.end,
                ))

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
            text=raw_texts[i],
            clean_text=nlp_texts[i],
            source=sources[i],
            entities=entities,
            topic=topic,
            sentiment=sentiment,
        ))

    # Network / co-occurrence graph
    network = None
    entity_stats: dict = {}
    if run_ner and ner_results:
        nd = services.network.build_network(ner_results)
        entity_stats = services.network.get_entity_stats(ner_results)
        network = NetworkResponse(
            nodes=[
                NetworkNodeResponse(
                    id=n.id, label=n.label,
                    entity_type=n.entity_type, frequency=n.frequency,
                )
                for n in nd.nodes
            ],
            edges=[
                NetworkEdgeResponse(source=e.source, target=e.target, weight=e.weight)
                for e in nd.edges
            ],
        )

    return AnalysisResponse(
        documents=documents,
        network=network,
        topic_summary=topic_summary,
        sentiment_summary=sentiment_counts,
        entity_summary=entity_stats,
        total_documents=len(documents),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_text_column(first_row: dict):
    for name in ("text", "Text", "clean_text", "cleaned_text",
                 "content", "Content", "Message", "body", "Body"):
        if name in first_row:
            return name
    return None


def _save_and_attach_doc_ids(
    result: AnalysisResponse,
    source_filename: str = "",
) -> AnalysisResponse:
    """
    Persist the analysis to the DB and return a new AnalysisResponse where
    every DocumentResponse has its doc_id (DB row id) filled in.
    """
    docs = []
    for doc in result.documents:
        docs.append({
            "raw_text":        doc.text,
            "nlp_text":        doc.clean_text,
            "source":          doc.source,
            "sentiment_label": doc.sentiment.label if doc.sentiment else "",
            "sentiment_score": doc.sentiment.score if doc.sentiment else 0.0,
            "entities": [
                {
                    "word": e.word,
                    "entity_group": e.entity_group,
                    "score": e.score,
                    "start": e.start,
                    "end": e.end,
                }
                for e in (doc.entities or [])
            ],
            "topic_id":       doc.topic.topic_id if doc.topic else -1,
            "topic_label":    doc.topic.topic_label if doc.topic else "",
            "topic_keywords": doc.topic.keywords if doc.topic else [],
        })

    try:
        _session_id, doc_ids = services.kb.save_analysis(
            documents=docs,
            sentiment_summary=result.sentiment_summary,
            entity_summary=result.entity_summary,
            topic_summary=result.topic_summary,
            source_filename=source_filename,
        )
        # Attach the DB ids to the response documents
        new_docs = [
            doc.model_copy(update={"doc_id": did})
            for doc, did in zip(result.documents, doc_ids)
        ]
        return result.model_copy(update={"documents": new_docs})
    except Exception as exc:
        # Never let a DB write failure break the analysis response
        print(f"[analysis] Warning: could not save analysis to DB: {exc}")
        return result