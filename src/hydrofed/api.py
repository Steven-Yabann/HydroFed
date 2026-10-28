"""HTTP API for reading saved HydroFed experiment results."""

import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from hydrofed.storage import ExperimentStore


DEFAULT_DASHBOARD_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)


def allowed_origins_from_environment() -> list[str]:
    raw_origins = os.environ.get(
        "HYDROFED_ALLOWED_ORIGINS",
        DEFAULT_DASHBOARD_ORIGINS,
    )
    origins = [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]

    if not origins:
        raise ValueError("HYDROFED_ALLOWED_ORIGINS must contain an origin")

    return origins


def create_app(
    db_path: str | Path | None = None,
) -> FastAPI:
    database_path = (
        Path(db_path)
        if db_path is not None
        else Path(
            os.environ.get(
                "HYDROFED_DB_PATH",
                "artifacts/experiments.db",
            )
        )
    )

    app = FastAPI(
        title="HydroFed API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins_from_environment(),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    store = ExperimentStore(database_path)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/results")
    def results(
        experiment: str | None = None,
        kind: str | None = None,
        client_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, list[dict]]:
        return {
            "results": store.list_results(
                experiment=experiment,
                kind=kind,
                client_id=client_id,
                limit=limit,
            )
        }

    return app