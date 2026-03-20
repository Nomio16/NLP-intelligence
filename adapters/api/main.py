"""
FastAPI adapter — REST API entry point.
This is the outer adapter that wraps the NLP core domain layer.
"""

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from adapters.api.routers import analysis, insights, admin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NLP Intelligence API",
    description="Social media content analysis: NER, Topic Modeling, Sentiment Analysis, Network Analysis",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception on {request.method} {request.url}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # temporarily for testing
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
