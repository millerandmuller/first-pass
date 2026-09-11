"""Vercel's conventional FastAPI entrypoint path (docs.vercel.com/docs/frameworks/backend/fastapi).

Re-exports the real app from backend/api.py -- kept there so it also runs
under a plain local `uvicorn backend.api:app` for development, with no
Vercel-specific path assumptions in the application code itself.
"""

from backend.api import app  # noqa: F401
