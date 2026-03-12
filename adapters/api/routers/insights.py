"""
Insights router — extracts actionable insights (complaints, hot issues, etc.)
"""

from typing import List
from fastapi import APIRouter, HTTPException
from adapters.api.schemas import InsightResponse

router = APIRouter()


@router.post("/insights", response_model=List[InsightResponse])
async def get_insights():
    """
    Generate insights from the last analysis results.
    Must run /api/upload or /api/analyze first.
    """
    from adapters.api.routers.analysis import _last_analysis

    if _last_analysis is None:
        raise HTTPException(status_code=404, detail="No analysis has been run yet.")

    insights = []

    # 1. Complaints: negative sentiment documents
    negative_docs = [
        d for d in _last_analysis.documents
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

    # 2. Compliments: positive sentiment documents
    positive_docs = [
        d for d in _last_analysis.documents
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

    # 3. Hot entities: most mentioned entities
    if _last_analysis.entity_summary:
        for etype, entities in _last_analysis.entity_summary.items():
            if entities:
                top = entities[0]
                insights.append(InsightResponse(
                    category="hot_issue",
                    title=f"Хамгийн их дурдагдсан {etype}",
                    description=f"\"{top['word']}\" — {top['count']} удаа дурдагдсан",
                    count=top["count"],
                    sample_texts=[],
                ))

    # 4. Sentiment ratio insight
    total = _last_analysis.total_documents
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

    return insights
