"""FastAPI application entry point."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.books import router as books_router
from app.routers.generate_context import router as generate_context_router
from ingestion.llm_client import GpuWorkerPool

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # POST /api/compose-scene needs the LLM fleet to plan video shots; the
    # rest of the API (library/reader/generate-context) only needs Postgres,
    # per the README's "API/DB only" mode -- so a missing/unreachable vLLM
    # fleet must not block the whole app from starting.
    try:
        app.state.gpu_pool = GpuWorkerPool()
    except ValueError:
        logger.warning(
            "No vLLM endpoints configured (BVG_VLLM_ENDPOINTS) -- "
            "/api/compose-scene will return 503 until one is set"
        )
        app.state.gpu_pool = None
    yield
    if app.state.gpu_pool is not None:
        await app.state.gpu_pool.aclose()


app = FastAPI(
    title="book_video_gen",
    description="Compiles self-contained video/audio generation payloads from book paragraph beats.",
    version="1.0.0",
    lifespan=lifespan,
)

# The reader frontend (Vite dev server) runs on a different origin than the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_context_router)
app.include_router(books_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
