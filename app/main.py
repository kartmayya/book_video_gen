"""FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI

from app.routers.generate_context import router as generate_context_router

app = FastAPI(
    title="book_video_gen",
    description="Compiles self-contained video/audio generation payloads from book paragraph beats.",
    version="1.0.0",
)

app.include_router(generate_context_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
