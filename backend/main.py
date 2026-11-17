"""FastAPI application entrypoint.

Run with:  uvicorn backend.main:app --reload
Interactive docs at:  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router
from .db.models import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables if they don't exist. Fine for a SQLite prototype.
    init_db()
    yield


app = FastAPI(title="Portfolio Optimizer", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
