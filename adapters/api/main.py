"""
FastAPI adapter — REST API entry point.
This is the outer adapter that wraps the NLP core domain layer.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adapters.api.routers import analysis, insights, admin

app = FastAPI(
    title="NLP Intelligence API",
    description="Social media content analysis: NER, Topic Modeling, Sentiment Analysis, Network Analysis",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(insights.router, prefix="/api", tags=["Insights"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])


@app.get("/")
async def root():
    return {
        "name": "NLP Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /api/upload",
            "analyze": "POST /api/analyze",
            "network": "POST /api/network",
            "insights": "POST /api/insights",
            "admin_entries": "GET/POST /api/admin/knowledge",
            "admin_labels": "GET/POST /api/admin/labels",
            "admin_stopwords": "GET/POST /api/admin/stopwords",
        },
    }
