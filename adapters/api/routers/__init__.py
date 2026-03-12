from .analysis import router as analysis_router
from .insights import router as insights_router
from .admin import router as admin_router

__all__ = ["analysis_router", "insights_router", "admin_router"]
