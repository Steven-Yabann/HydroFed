"""HTTP API for reading saved HydroFed experiment results."""

from pathlib import Path

from fastapi import FastAPI, Query

from hydrofed.storage import ExperimentStore
from fastapi.middleware.cors import CORSMiddleware


def create_app(
    db_path: str | Path = "artifacts/experiments.db",
) -> FastAPI:
    app = FastAPI(
        title="HydroFed API",
        version="0.1.0",
    )

    store = ExperimentStore(db_path)

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

    app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
    )

    return app

