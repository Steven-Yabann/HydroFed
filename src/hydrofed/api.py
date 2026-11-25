"""HTTP API behind the HydroFed dashboard.

Every route except health, the sign-in routes and the provider list needs a
session token (``Authorization: Bearer ...``). Routes then check the role's
permissions from ``hydrofed.auth.ROLE_PERMISSIONS``:

- plant operators watch their own plant, review its alerts and message the
  administrators about it
- administrators see every plant, read performance results and manage users,
  plant nodes and federated training
"""

import json
import os
from pathlib import Path
from urllib.parse import urlencode

from fastapi import BackgroundTasks, Body, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from hydrofed.auth import (
    ROLE_LABELS,
    ROLE_PERMISSIONS,
    AuthError,
    UserStore,
    authorization_redirect,
    bootstrap_admin,
    visible_plants,
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
from hydrofed.jobs import GitHubTrainingLauncher, JobError, TrainingLauncher, make_launcher
from hydrofed.messages import MessageError, MessageStore
from hydrofed.notifications import Mailer, mailer_from_environment, notify_new_message
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
    plant_id: str | None = None


class UserChanges(BaseModel):
    name: str | None = None
    role: str | None = None
    status: str | None = None
    password: str | None = None
    plant_id: str | None = None


class AlertReview(BaseModel):
    status: str
    note: str = ""


class NewMessage(BaseModel):
    body: str
    alert_id: str | None = None


class Preferences(BaseModel):
    notify_messages: bool


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


def create_app(
    db_path: str | Path | None = None,
    feed_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    partition_summary_path: str | Path | None = None,
    launcher: TrainingLauncher | GitHubTrainingLauncher | None = None,
    clock=None,
    mailer: Mailer | None = None,
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
    bootstrap_admin(users)
    operations = OperationsStore(database_path, feed_directory, **({"clock": clock} if clock else {}))
    messages = MessageStore(database_path)
    email = mailer or mailer_from_environment()
    training = launcher or make_launcher(database_path)
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

    def own_plant(user: dict, plant: str) -> None:
        """Refuse plants outside an operator's assignment."""
        if not visible_plants(user, [plant]):
            raise HTTPException(403, "This plant is not assigned to your account.")

    def check_assignment(role: str | None, plant_id: str | None, current: dict | None = None) -> None:
        """Operators need a plant that exists; other roles see every plant."""
        role = role or (current or {}).get("role")
        plant = plant_id if plant_id is not None else (current or {}).get("plant_id")
        if "view_all_plants" in ROLE_PERMISSIONS.get(role, ()):
            return
        if not plant:
            raise HTTPException(400, "Choose the plant this operator monitors.")
        known = operations_call(operations.plant_ids)
        if plant not in known:
            raise HTTPException(400, f"Unknown plant: {plant}")

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

    @app.get("/api/auth/preferences")
    def preferences(user: dict = Depends(current_user)) -> dict:
        return {"notify_messages": user["notify_messages"], "email_available": email.available}

    @app.put("/api/auth/preferences")
    def update_preferences(request: Preferences, user: dict = Depends(current_user)) -> dict:
        updated = users.update(user["user_id"], notify_messages=request.notify_messages)
        return {"notify_messages": updated["notify_messages"], "email_available": email.available}

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
    def fleet(user: dict = Depends(requires("view_fleet"))) -> dict:
        plants = visible_plants(user, operations_call(operations.plant_ids))
        return operations_call(operations.fleet, plants)

    @app.get("/api/plants/{plant}")
    def plant(plant: str, user: dict = Depends(requires("view_fleet"))) -> dict:
        own_plant(user, plant)
        return operations_call(operations.plant_summary, plant)

    @app.get("/api/plants/{plant}/telemetry")
    def telemetry(
        plant: str,
        seconds: int = Query(default=600, ge=30, le=3600),
        user: dict = Depends(requires("view_fleet")),
    ) -> dict:
        own_plant(user, plant)
        return operations_call(operations.telemetry, plant, seconds)

    @app.get("/api/alerts")
    def alerts(
        plant: str | None = None,
        limit: int = Query(default=200, ge=1, le=1000),
        user: dict = Depends(requires("view_fleet")),
    ) -> dict:
        if plant:
            own_plant(user, plant)
            items = operations_call(operations.alerts, plant, limit)
        elif "view_all_plants" in user["permissions"]:
            items = operations_call(operations.alerts, None, limit)
        else:
            items = operations_call(operations.alerts, user["plant_id"], limit) if user["plant_id"] else []
        return {"alerts": items, "statuses": list(REVIEW_STATUSES)}

    @app.post("/api/alerts/{alert_id}/review")
    def review_alert(alert_id: str, review: AlertReview, user: dict = Depends(requires("review_alerts"))) -> dict:
        own_plant(user, alert_id.split(":", 1)[0])
        return {"alert": operations_call(operations.review_alert, alert_id, review.status, review.note, user)}

    # --- messages between operators and administrators ---

    def conversation_plant(user: dict, plant: str) -> str:
        own_plant(user, plant)
        if plant not in operations_call(operations.plant_ids):
            raise HTTPException(404, f"Unknown plant: {plant}")
        return plant

    @app.get("/api/messages")
    def message_threads(user: dict = Depends(requires("send_messages"))) -> dict:
        plants = visible_plants(user, operations_call(operations.plant_ids))
        threads = [
            {**summary, "plant_name": operations.node(summary["plant"])["display_name"]}
            for summary in messages.summaries(plants, user)
        ]
        return {"threads": threads, "unread": sum(thread["unread"] for thread in threads)}

    @app.get("/api/messages/{plant}")
    def message_thread(plant: str, user: dict = Depends(requires("send_messages"))) -> dict:
        conversation_plant(user, plant)
        return {
            "plant": plant,
            "plant_name": operations.node(plant)["display_name"],
            "messages": messages.thread(plant, user),
        }

    @app.post("/api/messages/{plant}", status_code=201)
    def send_message(
        plant: str,
        request: NewMessage,
        background: BackgroundTasks,
        user: dict = Depends(requires("send_messages")),
    ) -> dict:
        conversation_plant(user, plant)
        alert = None
        if request.alert_id:
            found = next(
                (item for item in operations_call(operations.alerts, plant, 10_000) if item["id"] == request.alert_id),
                None,
            )
            if found is None:
                raise HTTPException(400, "That alert does not belong to this plant or has not happened yet.")
            alert = {"id": found["id"], "severity": found["severity"], "started_at": found["started_at"]}
        try:
            message = messages.post(plant, user, request.body, alert)
        except MessageError as error:
            raise HTTPException(400, str(error)) from error
        # Emails go out after the response, so a slow mail server never delays sending.
        background.add_task(
            notify_new_message, email, users.list(), lambda account: messages.unread_for(plant, account),
            message, user, operations.node(plant)["display_name"], f"{frontend_url}/#/messages/{plant}",
        )
        return {"message": message}

    # --- administration ---

    @app.get("/api/admin/users")
    def list_users(_: dict = Depends(requires("manage_users"))) -> dict:
        return {
            "users": users.list(),
            "roles": [
                {"id": role, "label": ROLE_LABELS.get(role, role), "permissions": list(permissions)}
                for role, permissions in ROLE_PERMISSIONS.items()
            ],
            "plants": [
                {"id": plant, "name": operations.node(plant)["display_name"]}
                for plant in operations_call(operations.plant_ids)
            ],
        }

    @app.post("/api/admin/users", status_code=201)
    def create_user(request: NewUser, admin: dict = Depends(requires("manage_users"))) -> dict:
        check_assignment(request.role, request.plant_id)
        try:
            return {"user": users.create(
                request.email, request.name, request.role, request.password or None, admin["user_id"],
                request.plant_id,
            )}
        except AuthError as error:
            raise HTTPException(400, str(error)) from error

    @app.patch("/api/admin/users/{user_id}")
    def update_user(user_id: int, changes: UserChanges, _: dict = Depends(requires("manage_users"))) -> dict:
        current = users.get(user_id)
        if current is not None and (changes.role is not None or changes.plant_id is not None):
            check_assignment(changes.role, changes.plant_id, current)
        try:
            return {"user": users.update(
                user_id, name=changes.name, role=changes.role, status=changes.status,
                password=changes.password or None, plant_id=changes.plant_id,
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

    # --- training runner (GitHub Actions) ---

    @app.post("/api/training/report")
    def training_report(report: dict = Body(...), authorization: str | None = Header(default=None)) -> dict:
        """Receive a finished run from the training workflow, authenticated by the runner token."""
        token = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None
        try:
            return training.receive_report(token, report)
        except PermissionError as error:
            raise HTTPException(401, str(error)) from error
        except JobError as error:
            raise HTTPException(409, str(error)) from error
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(400, f"The report is incomplete: {error}") from error

    return app
