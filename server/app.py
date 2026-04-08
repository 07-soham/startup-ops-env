"""Server entry point for HF Spaces multi-mode deployment.

This module provides the `main` entry point consumed by the
``[project.scripts] server = "server.app:main"`` declaration in
pyproject.toml.  OpenEnv expects a callable entry-point named ``server``
that starts the FastAPI application.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    """Start the StartupOps FastAPI server (OpenEnv multi-mode entry point)."""
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install uvicorn", file=sys.stderr)
        return 1

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    workers = int(os.environ.get("WORKERS", "1"))
    log_level = os.environ.get("LOG_LEVEL", "info")

    print(f"[server] StartupOps AI Simulator starting on {host}:{port}", flush=True)
    print(f"[server] OpenEnv endpoints: POST /reset  POST /step  GET /state", flush=True)
    print(f"[server] API docs: http://{host}:{port}/docs", flush=True)

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
