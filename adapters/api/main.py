import logging
import traceback
import torch

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


@app.on_event("startup")
async def warmup_models():
    """
    Pre-load heavy ML models at server startup so the first user request
    doesn't trigger a 30-120s model download/load that causes ECONNRESET.
    """
    import asyncio
    from adapters.api import services

    async def _warmup():
        logger.info("[Warmup] Pre-loading NLP models in background...")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, services.ner.recognize, "warmup")
            logger.info("[Warmup] NER model loaded ✓")
        except Exception as e:
            logger.warning(f"[Warmup] NER failed: {e}")
        try:
            await loop.run_in_executor(None, services.sentiment.analyze, "warmup")
            logger.info("[Warmup] Sentiment model loaded ✓")
        except Exception as e:
            logger.warning(f"[Warmup] Sentiment failed: {e}")
        logger.info("[Warmup] All models ready.")

    # Run warmup in background so the server starts accepting requests immediately
    asyncio.create_task(_warmup())


@app.get("/")
async def root():
    return {
        "name": "NLP Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /api/health",
            "upload": "POST /api/upload",
            "analyze": "POST /api/analyze",
            "network": "POST /api/network",
            "insights": "POST /api/insights",
            "admin_entries": "GET/POST /api/admin/knowledge",
            "admin_labels": "GET/POST /api/admin/labels",
            "admin_stopwords": "GET/POST /api/admin/stopwords",
        },
    }


@app.get("/api/health")
async def health():
    """
    Quick health check used by the frontend on page load.
    Returns GPU availability and which NLP models are loaded.
    """
    from adapters.api import services
    gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu else None
    return {
        "status": "ok",
        "gpu": gpu,
        "gpu_name": gpu_name,
        "models": {
            "ner": services.ner._pipeline is not None,
            "sentiment": services.sentiment._pipeline is not None,
            "topic": services.topic._model is not None,
        },
    }
