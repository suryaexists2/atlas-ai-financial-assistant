"""Health endpoints for uptime pings and deployment checks."""

import os

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.interfaces.api.deps import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
        "build": os.environ.get("RENDER_GIT_COMMIT"),
        "llm": {
            "provider": settings.llm_provider,
            "groq_key": bool(settings.groq_api_key) or bool(settings.groq_api_keys),
            "gemini_key": bool(settings.gemini_api_key),
            "openrouter_key": bool(settings.openrouter_api_key),
            "groq_model": settings.groq_llm_model,
            "gemini_model": settings.gemini_llm_model,
            "primary_model": settings.llm_model,
        },
    }


@router.get("/health/db")
async def health_db(db: AsyncSession = Depends(get_db)) -> dict:
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "reachable"}
    except Exception as exc:  # noqa: BLE001 - surface any connectivity failure
        return {"status": "error", "database": str(exc)}
