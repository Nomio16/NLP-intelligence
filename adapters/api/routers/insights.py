"""
Insights router — extracts actionable insights from the last analysis.

Key change from original:
  Original imported _last_analysis directly from analysis.py:
      from adapters.api.routers.analysis import _last_analysis

  This is fragile because:
    a) It creates a tight coupling between two router modules
    b) Python resolves the name at import time — if analysis.py hasn't run
       yet, or if the variable is rebound, insights.py sees stale data
    c) It relies on a mutable module-level global, which is hard to test

  Now uses services.get_last_analysis() — the same function analysis.py
  calls services.set_last_analysis() on, so they're guaranteed in sync.

No changes to the insight logic itself — that works correctly.
"""

from typing import List
from fastapi import APIRouter, HTTPException
from adapters.api.schemas import InsightResponse
from adapters.api import services

router = APIRouter()


@router.post("/insights", response_model=List[InsightResponse])
async def get_insights():
    """
    Generate actionable insights from the most recent analysis.
    Must call /upload or /analyze first.
    """
    # Use services.get_last_analysis() instead of a direct module import
    last = services.get_last_analysis()
    if last is None:
        raise HTTPException(
            status_code=404,
            detail="No analysis has been run yet. Upload a CSV or analyze a text first.",
        )

    insights: List[InsightResponse] = []
    total = last.total_documents

    # ------------------------------------------------------------------
    # 1. Complaints: documents with negative sentiment
    # ------------------------------------------------------------------
    negative_docs = [
        d for d in last.documents
        if d.sentiment and d.sentiment.label == "negative"
    ]
    if negative_docs:
        insights.append(InsightResponse(
            category="complaint",
            title="Гомдол / Сөрөг сэтгэгдэл",
            description=f"{len(negative_docs)} нийтлэлд сөрөг хандлага илэрсэн",
            count=len(negative_docs),
            sample_texts=[d.text[:200] for d in negative_docs[:5]],
        ))

    # ------------------------------------------------------------------
    # 2. Compliments: documents with positive sentiment
    # ------------------------------------------------------------------
    positive_docs = [
        d for d in last.documents
        if d.sentiment and d.sentiment.label == "positive"
    ]
    if positive_docs:
        insights.append(InsightResponse(
            category="compliment",
            title="Магтаал / Эерэг сэтгэгдэл",
            description=f"{len(positive_docs)} нийтлэлд эерэг хандлага илэрсэн",
            count=len(positive_docs),
            sample_texts=[d.text[:200] for d in positive_docs[:5]],
        ))

    # ------------------------------------------------------------------
    # 3. Hot entities: top mentioned entity per type
    # ------------------------------------------------------------------
    if last.entity_summary:
        for etype, entities in last.entity_summary.items():
            if entities:
                top = entities[0]
                insights.append(InsightResponse(
                    category="hot_issue",
                    title=f"Хамгийн их дурдагдсан {etype}",
                    description=f'"{top["word"]}" — {top["count"]} удаа дурдагдсан',
                    count=top["count"],
                    sample_texts=[],
                ))

    # ------------------------------------------------------------------
    # 4. Sentiment ratio alerts
    # ------------------------------------------------------------------
    if total > 0:
        neg_pct = (len(negative_docs) / total) * 100
        pos_pct = (len(positive_docs) / total) * 100

        if neg_pct > 50:
            insights.append(InsightResponse(
                category="important",
                title="Анхааруулга: Сөрөг хандлага давамгай",
                description=f"Нийт нийтлэлийн {neg_pct:.0f}% нь сөрөг сэтгэгдэлтэй байна",
                count=len(negative_docs),
                sample_texts=[],
            ))
        elif pos_pct > 70:
            insights.append(InsightResponse(
                category="important",
                title="Сайн мэдээ: Эерэг хандлага давамгай",
                description=f"Нийт нийтлэлийн {pos_pct:.0f}% нь эерэг сэтгэгдэлтэй байна",
                count=len(positive_docs),
                sample_texts=[],
            ))

    # ------------------------------------------------------------------
    # 5. Topic insights (new — original had no topic insight generation)
    # ------------------------------------------------------------------
    if last.topic_summary:
        # Filter out error/info meta entries
        real_topics = [
            t for t in last.topic_summary
            if isinstance(t, dict) and "topic_id" in t and t.get("topic_id", -1) != -1
        ]
        if real_topics:
            # Sort by document count and surface the largest topic
            top_topic = max(real_topics, key=lambda t: t.get("count", 0))
            insights.append(InsightResponse(
                category="hot_issue",
                title=f"Гол сэдэв: {top_topic.get('name', 'Topic')}",
                description=f"{top_topic.get('count', 0)} нийтлэл энэ сэдвийг агуулна",
                count=top_topic.get("count", 0),
                sample_texts=[],
            ))

    return insights