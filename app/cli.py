from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("LLAMACLAW_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("LLAMACLAW_PORT", "8000")))
    uvicorn.run("app.main:app", host=host, port=port)
