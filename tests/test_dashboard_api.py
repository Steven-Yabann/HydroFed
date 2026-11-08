import json

import pytest
from fastapi.testclient import TestClient

from hydrofed import auth
from hydrofed.api import create_app
from hydrofed.auth import AuthError, UserStore, identity_from_claims
from hydrofed.operations import REPLAY_EPOCH, PlantTimeline, health_state, severity

PASSWORD = "correct horse battery"
SENSORS = ("current", "pressure")


def write_feed(directory, plants=("plant_1", "plant_2"), length=100):
    """Write a small replay feed: each plant alerts from second 40 to 59."""
    directory.mkdir()
    index = {
        "source": {"threshold_mad": 24.0, "dataset": "test"},
        "sensors": [{"key": key, "label": key, "unit": ""} for key in SENSORS],
        "plants": [],
    }
    for plant in plants:
        alert = [1 if 40 <= second < 60 else 0 for second in range(length)]
        recording = {
            "id": "valve1/0.csv",
            "fault_type": "valve1",
            "role": "final",
            "seconds": list(range(400, 400 + length)),
            "score": [100.0 if flag else 1.0 for flag in alert],
            "label": alert,
            "alert": alert,
            "sensors": {
                "current": [5.0 if flag else 1.0 for flag in alert],
                "pressure": [1.0] * length,
            },
        }
        (directory / f"{plant}.json").write_text(json.dumps({"plant": plant, "recordings": [recording]}))
        index["plants"].append({"id": plant, "recordings": 1, "seconds": length, "alert_seconds": 20})
    (directory / "index.json").write_text(json.dumps(index))
    return directory


class FakeLauncher:
    def __init__(self):
        self.started = None

    def status(self):
        return {"available": True, "reason": "", "job": self.started}

    def start(self, config, clients, user):
        self.started = {"clients": clients, "config": config, "started_by": user["name"]}
        return self.status()

    def stop(self):
        return self.status()


@pytest.fixture
def app_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDROFED_SECRET_KEY", "test-secret")
    database = tmp_path / "experiments.db"
    users = UserStore(database)
    users.create("admin@example.com", "Ada Admin", "admin", PASSWORD)
    users.create("operator@example.com", "Otis Operator", "operator", PASSWORD)
    launcher = FakeLauncher()
    # Second 70 of plant_1's loop: its alert (seconds 40-59) has ended.
    app = create_app(
        database,
        feed_path=write_feed(tmp_path / "feed"),
        comparison_path=tmp_path / "missing",
        partition_summary_path=tmp_path / "summary.json",
        launcher=launcher,
        clock=lambda: REPLAY_EPOCH + 70,
    )
    return TestClient(app), users, launcher


def sign_in(client, email):
    response = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_password_sign_in_returns_the_user_and_permissions(app_setup):
    client, _, _ = app_setup

    headers = sign_in(client, "Operator@Example.com")
    user = client.get("/api/auth/me", headers=headers).json()["user"]

    assert user["role"] == "operator"
    assert "view_fleet" in user["permissions"]
    assert "manage_users" not in user["permissions"]
    assert user["last_provider"] == "password"


def test_wrong_password_and_unknown_email_get_the_same_message(app_setup):
    client, _, _ = app_setup

    wrong = client.post("/api/auth/login", json={"email": "operator@example.com", "password": "nope-nope-nope"})
    unknown = client.post("/api/auth/login", json={"email": "who@example.com", "password": PASSWORD})

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_protected_routes_need_a_session(app_setup):
    client, _, _ = app_setup

    assert client.get("/api/fleet").status_code == 401
    assert client.get("/api/fleet", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_operators_cannot_reach_administration(app_setup):
    client, _, _ = app_setup
    headers = sign_in(client, "operator@example.com")

    assert client.get("/api/admin/users", headers=headers).status_code == 403
    assert client.get("/api/admin/federation", headers=headers).status_code == 403
    assert client.post("/api/admin/training", headers=headers).status_code == 403


def test_disabled_accounts_lose_their_session(app_setup):
    client, users, _ = app_setup
    headers = sign_in(client, "operator@example.com")
    operator = users.find_by_email("operator@example.com")

    users.update(operator["user_id"], status="disabled")

    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_admin_adds_changes_and_removes_users(app_setup):
    client, _, _ = app_setup
    headers = sign_in(client, "admin@example.com")

    created = client.post(
        "/api/admin/users",
        json={"email": "new@example.com", "name": "New Person", "role": "operator"},
        headers=headers,
    )
    assert created.status_code == 201
    user = created.json()["user"]
    assert user["has_password"] is False

    duplicate = client.post(
        "/api/admin/users",
        json={"email": "NEW@example.com", "name": "Again", "role": "operator"},
        headers=headers,
    )
    assert duplicate.status_code == 400

    promoted = client.patch(f"/api/admin/users/{user['user_id']}", json={"role": "admin"}, headers=headers)
    assert promoted.json()["user"]["role"] == "admin"

    assert client.delete(f"/api/admin/users/{user['user_id']}", headers=headers).status_code == 200


def test_the_last_administrator_cannot_be_demoted(app_setup):
    client, users, _ = app_setup
    headers = sign_in(client, "admin@example.com")
    admin = users.find_by_email("admin@example.com")

    response = client.patch(f"/api/admin/users/{admin['user_id']}", json={"role": "operator"}, headers=headers)

    assert response.status_code == 400
    assert "administrator" in response.json()["detail"]


def test_short_passwords_are_refused(tmp_path):
    with pytest.raises(AuthError):
        UserStore(tmp_path / "db.sqlite").create("a@example.com", "A", "operator", "short")


def test_single_sign_on_only_admits_existing_accounts(tmp_path):
    users = UserStore(tmp_path / "db.sqlite")
    users.create("known@example.com", "Known", "operator")

    user = auth.sso_user(users, "google", {"email": "known@example.com", "name": "Known"})
    assert user["last_provider"] == "google"

    with pytest.raises(AuthError, match="does not have a HydroFed account"):
        auth.sso_user(users, "google", {"email": "stranger@example.com", "name": "Stranger"})


def test_google_identities_need_a_verified_email():
    with pytest.raises(AuthError):
        identity_from_claims("google", {"email": "a@example.com", "email_verified": False})

    identity = identity_from_claims("microsoft", {"preferred_username": "A@Example.com", "name": "A"})
    assert identity == {"email": "a@example.com", "name": "A"}


def test_providers_are_listed_only_when_configured(app_setup, monkeypatch):
    client, _, _ = app_setup
    assert client.get("/api/auth/providers").json()["providers"] == []

    monkeypatch.setenv("HYDROFED_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("HYDROFED_GOOGLE_CLIENT_SECRET", "client-secret")

    assert client.get("/api/auth/providers").json()["providers"] == [{"id": "google", "label": "Google"}]


def test_oauth_start_redirects_with_pkce_and_a_state_cookie(app_setup, monkeypatch):
    client, _, _ = app_setup
    monkeypatch.setenv("HYDROFED_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("HYDROFED_GOOGLE_CLIENT_SECRET", "client-secret")

    response = client.get("/api/auth/oauth/google/start", follow_redirects=False)

    assert response.status_code == 302
    location = response.headers["location"]
    assert location.startswith("https://accounts.google.com/")
    assert "code_challenge_method=S256" in location
    assert "hydrofed_oauth" in response.cookies


def test_oauth_callback_signs_in_a_known_user(app_setup, monkeypatch):
    client, _, _ = app_setup
    monkeypatch.setenv("HYDROFED_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("HYDROFED_GOOGLE_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr("hydrofed.api.exchange_code", lambda *args: "id-token")
    monkeypatch.setattr(
        "hydrofed.api.verify_id_token",
        lambda provider, token, nonce: {"email": "operator@example.com", "name": "Otis"},
    )
    start = client.get("/api/auth/oauth/google/start", follow_redirects=False)
    state = start.headers["location"].split("state=")[1].split("&")[0]

    callback = client.get(f"/api/auth/oauth/google/callback?code=abc&state={state}", follow_redirects=False)

    assert callback.status_code == 302
    assert "#/auth/callback?token=" in callback.headers["location"]


def test_oauth_callback_rejects_a_mismatched_state(app_setup, monkeypatch):
    client, _, _ = app_setup
    monkeypatch.setenv("HYDROFED_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("HYDROFED_GOOGLE_CLIENT_SECRET", "client-secret")
    client.get("/api/auth/oauth/google/start", follow_redirects=False)

    callback = client.get("/api/auth/oauth/google/callback?code=abc&state=forged", follow_redirects=False)

    assert "error=" in callback.headers["location"]
    assert "token=" not in callback.headers["location"]


def test_fleet_reports_each_plant_at_its_replay_second(app_setup):
    client, _, _ = app_setup
    headers = sign_in(client, "operator@example.com")

    fleet = client.get("/api/fleet", headers=headers).json()

    assert [plant["id"] for plant in fleet["plants"]] == ["plant_1", "plant_2"]
    plant_1 = fleet["plants"][0]
    assert plant_1["position"] == 70
    assert plant_1["status"] == "normal"
    assert plant_1["name"] == "Plant 1"


def test_alerts_appear_once_they_have_happened(app_setup):
    client, _, _ = app_setup
    headers = sign_in(client, "operator@example.com")

    alerts = client.get("/api/alerts?plant=plant_1", headers=headers).json()["alerts"]

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["severity"] == "critical"
    assert alert["duration_seconds"] == 20
    assert alert["ongoing"] is False
    assert alert["sensor"] == "current"
    assert alert["status"] == "open"


def test_operators_review_alerts(app_setup):
    client, _, _ = app_setup
    headers = sign_in(client, "operator@example.com")
    alert_id = client.get("/api/alerts?plant=plant_1", headers=headers).json()["alerts"][0]["id"]

    reviewed = client.post(
        f"/api/alerts/{alert_id}/review",
        json={"status": "acknowledged", "note": "Crew dispatched"},
        headers=headers,
    ).json()["alert"]

    assert reviewed["status"] == "acknowledged"
    assert reviewed["reviewed_by"] == "Otis Operator"
    bad = client.post(f"/api/alerts/{alert_id}/review", json={"status": "ignored"}, headers=headers)
    assert bad.status_code == 400


def test_telemetry_returns_the_recent_window(app_setup):
    client, _, _ = app_setup
    headers = sign_in(client, "operator@example.com")

    telemetry = client.get("/api/plants/plant_1/telemetry?seconds=30", headers=headers).json()

    assert len(telemetry["score"]) == 30
    # The window covers loop seconds 41-70; the alert ran until second 59.
    assert telemetry["alert"] == [1] * 19 + [0] * 11
    assert set(telemetry["sensors"]) == set(SENSORS)
    assert client.get("/api/plants/plant_9", headers=headers).status_code == 404


def test_admin_saves_federation_settings_within_limits(app_setup):
    client, _, _ = app_setup
    headers = sign_in(client, "admin@example.com")

    saved = client.put("/api/admin/federation", json={"rounds": 10, "local_epochs": 2}, headers=headers)
    assert saved.status_code == 200
    assert saved.json()["config"]["rounds"] == 10

    too_many = client.put("/api/admin/federation", json={"rounds": 9999}, headers=headers)
    assert too_many.status_code == 400
    unknown = client.put("/api/admin/federation", json={"secret": 1}, headers=headers)
    assert unknown.status_code == 400


def test_training_uses_only_participating_plants(app_setup):
    client, _, launcher = app_setup
    headers = sign_in(client, "admin@example.com")

    # Two plants is the minimum, so pausing one of two is refused.
    paused = client.patch("/api/admin/nodes/plant_2", json={"participates": False}, headers=headers)
    assert paused.status_code == 400

    renamed = client.patch("/api/admin/nodes/plant_2", json={"display_name": "Masinga"}, headers=headers)
    assert renamed.json()["node"]["display_name"] == "Masinga"

    client.post("/api/admin/training", headers=headers)
    assert launcher.started["clients"] == ["plant_1", "plant_2"]


def test_comparisons_report_what_is_missing(app_setup):
    client, _, _ = app_setup
    headers = sign_in(client, "operator@example.com")

    digest = client.get("/api/comparisons", headers=headers).json()

    assert digest["available"] == {"regimes": False, "models": False, "scenarios": False, "personalised": False}


def test_severity_and_health_bands():
    assert severity(30, 24) == "medium"
    assert severity(50, 24) == "high"
    assert severity(100, 24) == "critical"
    assert health_state(5, 24) == "normal"
    assert health_state(15, 24) == "watch"
    assert health_state(30, 24) == "alarm"


def test_alert_episodes_merge_short_gaps():
    alert = [0] * 10 + [1] * 5 + [0] * 5 + [1] * 5 + [0] * 30 + [1] * 3
    recording = {
        "id": "r", "fault_type": "other", "alert": alert, "score": [50.0 if a else 0.0 for a in alert],
        "label": alert, "sensors": {"current": [float(a) for a in alert]}, "seconds": list(range(len(alert))),
    }

    episodes = PlantTimeline("plant_1", [recording]).episodes

    assert [(episode["start"], episode["end"]) for episode in episodes] == [(10, 24), (55, 57)]


def test_first_admin_comes_from_the_environment_on_a_fresh_database(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDROFED_SECRET_KEY", "test-secret")
    monkeypatch.setenv("HYDROFED_ADMIN_EMAIL", "Owner@Example.com")
    monkeypatch.setenv("HYDROFED_ADMIN_NAME", "Olu Owner")
    monkeypatch.setenv("HYDROFED_ADMIN_PASSWORD", PASSWORD)
    database = tmp_path / "experiments.db"
    client = TestClient(create_app(database, feed_path=tmp_path / "feed", launcher=FakeLauncher()))

    token = client.post("/api/auth/login", json={"email": "owner@example.com", "password": PASSWORD}).json()["token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert me["user"]["role"] == "admin"

    # Restarting with different variables must not add a second administrator.
    monkeypatch.setenv("HYDROFED_ADMIN_EMAIL", "someone-else@example.com")
    create_app(database, feed_path=tmp_path / "feed", launcher=FakeLauncher())
    assert [user["email"] for user in UserStore(database).list()] == ["owner@example.com"]


def test_admin_variables_are_ignored_once_an_administrator_exists(app_setup, monkeypatch, tmp_path):
    _, users, _ = app_setup
    monkeypatch.setenv("HYDROFED_ADMIN_EMAIL", "late@example.com")
    assert auth.bootstrap_admin(users) is None
    assert users.find_by_email("late@example.com") is None
