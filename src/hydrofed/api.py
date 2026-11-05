"""HTTP API behind the HydroFed dashboard.

Every route except health, the sign-in routes and the provider list needs a
session token (``Authorization: Bearer ...``). Routes then check the role's
permissions from ``hydrofed.auth.ROLE_PERMISSIONS``:

- plant operators watch the fleet, review alerts and read performance results
- administrators also manage users, plant nodes and federated training
"""

import json
import os
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from hydrofed.auth import (
    ROLE_LABELS,
    ROLE_PERMISSIONS,
    AuthError,
    UserStore,
    authorization_redirect,
    configured_providers,
    exchange_code,
    issue_session,
    read_flow_cookie,
    read_session,
    secret_key,
    sso_user,
    verify_id_token,
)
from hydrofed.comparisons import comparison_digest
from hydrofed.jobs import JobError, TrainingLauncher
from hydrofed.operations import REVIEW_STATUSES, OperationsError, OperationsStore
from hydrofed.storage import ExperimentStore


DEFAULT_DASHBOARD_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)
OAUTH_COOKIE = "hydrofed_oauth"


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


class LoginRequest(BaseModel):
    email: str
    password: str


class NewUser(BaseModel):
    email: str
    name: str
    role: str
    password: str | None = None


class UserChanges(BaseModel):
    name: str | None = None
    role: str | None = None
    status: str | None = None
    password: str | None = None


class AlertReview(BaseModel):
    status: str
    note: str = ""


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


def create_app(
    db_path: str | Path | None = None,
    feed_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    partition_summary_path: str | Path | None = None,
    launcher: TrainingLauncher | None = None,
    clock=None,
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
    feed_directory = Path(feed_path or os.environ.get("HYDROFED_FEED_PATH", "artifacts/operator"))
    comparison_directory = Path(
        comparison_path or os.environ.get("HYDROFED_COMPARISON_PATH", "artifacts/comparison")
    )
    summary_path = Path(
        partition_summary_path
        or os.environ.get("HYDROFED_PARTITION_SUMMARY", "data/partitions_skab/partition_summary.json")
    )
    frontend_url = os.environ.get("HYDROFED_FRONTEND_URL", "http://localhost:5173").rstrip("/")
    public_api_url = os.environ.get("HYDROFED_PUBLIC_API_URL", "http://localhost:8000").rstrip("/")

    app = FastAPI(
        title="HydroFed API",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins_from_environment(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    store = ExperimentStore(database_path)
    users = UserStore(database_path)
    operations = OperationsStore(database_path, feed_directory, **({"clock": clock} if clock else {}))
    training = launcher or TrainingLauncher(database_path)
    key = secret_key(database_path)

    def current_user(authorization: str | None = Header(default=None)) -> dict:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Sign in to continue.")
        try:
            user = users.get(read_session(authorization[7:], key))
        except AuthError as error:
            raise HTTPException(401, str(error)) from error
        if user is None or user["status"] != "active":
            raise HTTPException(401, "This account is no longer active.")
        return user

    def requires(permission: str):
        def check(user: dict = Depends(current_user)) -> dict:
            if permission not in user["permissions"]:
                raise HTTPException(403, "Your role cannot do this.")
            return user
        return check

    def operations_call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except OperationsError as error:
            message = str(error)
            raise HTTPException(404 if message.startswith(("Unknown plant", "No operator feed")) else 400, message)

    # --- public ---

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/auth/providers")
    def providers() -> dict:
        return {
            "password": True,
            "providers": [
                {"id": provider.name, "label": provider.label}
                for provider in configured_providers().values()
            ],
        }

    @app.post("/api/auth/login")
    def login(request: LoginRequest) -> dict:
        try:
            user = users.authenticate(request.email, request.password)
        except AuthError as error:
            raise HTTPException(401, str(error)) from error
        return issue_session(user, key)

    @app.get("/api/auth/oauth/{provider}/start")
    def oauth_start(provider: str) -> RedirectResponse:
        settings = configured_providers().get(provider)
        if settings is None:
            raise HTTPException(404, f"{provider} sign-in is not configured.")
        url, cookie = authorization_redirect(
            settings, f"{public_api_url}/api/auth/oauth/{provider}/callback", key,
        )
        response = RedirectResponse(url, status_code=302)
        response.set_cookie(
            OAUTH_COOKIE, cookie, max_age=600, httponly=True, samesite="lax",
            secure=public_api_url.startswith("https://"), path="/api/auth/oauth",
        )
        return response

    @app.get("/api/auth/oauth/{provider}/callback")
    def oauth_callback(
        provider: str,
        request: Request,
        code: str | None = None,
        state: str | None = None,
        error: str | None = None,
    ) -> RedirectResponse:
        def finish(fragment: dict) -> RedirectResponse:
            response = RedirectResponse(f"{frontend_url}/#/auth/callback?{urlencode(fragment)}", status_code=302)
            response.delete_cookie(OAUTH_COOKIE, path="/api/auth/oauth")
            return response

        settings = configured_providers().get(provider)
        if settings is None:
            return finish({"error": f"{provider} sign-in is not configured."})
        if error or not code:
            return finish({"error": "Sign-in was cancelled."})
        try:
            flow = read_flow_cookie(request.cookies.get(OAUTH_COOKIE), provider, state, key)
            id_token = exchange_code(
                settings, code, f"{public_api_url}/api/auth/oauth/{provider}/callback", flow["verifier"],
            )
            identity = verify_id_token(settings, id_token, flow["nonce"])
            session = issue_session(sso_user(users, provider, identity), key)
        except AuthError as failure:
            return finish({"error": str(failure)})
        # The token travels in the URL fragment, which browsers never send to a server.
        return finish({"token": session["token"], "expires_at": session["expires_at"]})

    # --- signed-in users ---

    @app.get("/api/auth/me")
    def me(user: dict = Depends(current_user)) -> dict:
        return {"user": user}

    @app.post("/api/auth/password")
    def change_password(request: PasswordChange, user: dict = Depends(current_user)) -> dict:
        try:
            users.authenticate(user["email"], request.current_password)
            return {"user": users.update(user["user_id"], password=request.new_password)}
        except AuthError as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/results")
    def results(
        experiment: str | None = None,
        kind: str | None = None,
        client_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        run_id: str | None = None,
        _: dict = Depends(requires("view_performance")),
    ) -> dict[str, list[dict]]:
        return {
            "results": store.list_results(
                experiment=experiment,
                kind=kind,
                client_id=client_id,
                limit=limit,
                run_id=run_id,
            )
        }

    @app.get("/api/runs")
    def runs(
        experiment: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        _: dict = Depends(requires("view_performance")),
    ) -> dict[str, list[dict]]:
        return {
            "runs": store.list_runs(
                experiment=experiment,
                limit=limit,
            )
        }

    @app.get("/api/comparisons")
    def comparisons(
        role: str = Query(default="final", pattern="^(dev|final)$"),
        _: dict = Depends(requires("view_performance")),
    ) -> dict:
        return comparison_digest(comparison_directory, role)

    # --- plant operations ---

    @app.get("/api/fleet")
    def fleet(_: dict = Depends(requires("view_fleet"))) -> dict:
        return operations_call(operations.fleet)

    @app.get("/api/plants/{plant}")
    def plant(plant: str, _: dict = Depends(requires("view_fleet"))) -> dict:
        return operations_call(operations.plant_summary, plant)

    @app.get("/api/plants/{plant}/telemetry")
    def telemetry(
        plant: str,
        seconds: int = Query(default=600, ge=30, le=3600),
        _: dict = Depends(requires("view_fleet")),
    ) -> dict:
        return operations_call(operations.telemetry, plant, seconds)

    @app.get("/api/alerts")
    def alerts(
        plant: str | None = None,
        limit: int = Query(default=200, ge=1, le=1000),
        _: dict = Depends(requires("view_fleet")),
    ) -> dict:
        return {
            "alerts": operations_call(operations.alerts, plant, limit),
            "statuses": list(REVIEW_STATUSES),
        }

    @app.post("/api/alerts/{alert_id}/review")
    def review_alert(alert_id: str, review: AlertReview, user: dict = Depends(requires("review_alerts"))) -> dict:
        return {"alert": operations_call(operations.review_alert, alert_id, review.status, review.note, user)}

    # --- administration ---

    @app.get("/api/admin/users")
    def list_users(_: dict = Depends(requires("manage_users"))) -> dict:
        return {
            "users": users.list(),
            "roles": [
                {"id": role, "label": ROLE_LABELS.get(role, role), "permissions": list(permissions)}
                for role, permissions in ROLE_PERMISSIONS.items()
            ],
        }

    @app.post("/api/admin/users", status_code=201)
    def create_user(request: NewUser, admin: dict = Depends(requires("manage_users"))) -> dict:
        try:
            return {"user": users.create(
                request.email, request.name, request.role, request.password or None, admin["user_id"],
            )}
        except AuthError as error:
            raise HTTPException(400, str(error)) from error

    @app.patch("/api/admin/users/{user_id}")
    def update_user(user_id: int, changes: UserChanges, _: dict = Depends(requires("manage_users"))) -> dict:
        try:
            return {"user": users.update(
                user_id, name=changes.name, role=changes.role, status=changes.status,
                password=changes.password or None,
            )}
        except AuthError as error:
            raise HTTPException(400, str(error)) from error

    @app.delete("/api/admin/users/{user_id}")
    def delete_user(user_id: int, admin: dict = Depends(requires("manage_users"))) -> dict:
        if user_id == admin["user_id"]:
            raise HTTPException(400, "You cannot delete your own account.")
        try:
            users.delete(user_id)
        except AuthError as error:
            raise HTTPException(400, str(error)) from error
        return {"deleted": user_id}

    def partition_summary() -> dict:
        return json.loads(summary_path.read_text()) if summary_path.exists() else {}

    @app.get("/api/admin/nodes")
    def list_nodes(_: dict = Depends(requires("manage_nodes"))) -> dict:
        client_rounds = store.list_results(experiment="federated", kind="client_round", limit=500)
        return {"nodes": operations_call(operations.nodes, partition_summary(), client_rounds)}

    @app.patch("/api/admin/nodes/{plant}")
    def update_node(plant: str, changes: dict = Body(...), admin: dict = Depends(requires("manage_nodes"))) -> dict:
        return {"node": operations_call(operations.update_node, plant, changes, admin)}

    @app.get("/api/admin/federation")
    def federation_settings(_: dict = Depends(requires("manage_federation"))) -> dict:
        return {**operations.federation_config(), "training": training.status()}

    @app.put("/api/admin/federation")
    def update_federation(changes: dict = Body(...), admin: dict = Depends(requires("manage_federation"))) -> dict:
        try:
            return {**operations_call(operations.update_federation_config, changes, admin), "training": training.status()}
        except (TypeError, ValueError) as error:
            raise HTTPException(400, "Settings must be numbers within their limits.") from error

    @app.post("/api/admin/training")
    def start_training(admin: dict = Depends(requires("manage_federation"))) -> dict:
        clients = [plant for plant in operations_call(operations.plant_ids) if operations.node(plant)["participates"]]
        try:
            return training.start(operations.federation_config()["config"], clients, admin)
        except JobError as error:
            raise HTTPException(409, str(error)) from error

    @app.delete("/api/admin/training")
    def stop_training(_: dict = Depends(requires("manage_federation"))) -> dict:
        try:
            return training.stop()
        except JobError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/admin/training")
    def training_status(_: dict = Depends(requires("manage_federation"))) -> dict:
        return training.status()

    return app
