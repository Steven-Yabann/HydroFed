import json

import httpx
import pytest
from fastapi.testclient import TestClient

from hydrofed import notifications
from hydrofed.api import create_app
from hydrofed.auth import UserStore
from hydrofed.notifications import BrevoMailer, Mailer, SmtpMailer, compose, mailer_from_environment
from hydrofed.operations import REPLAY_EPOCH
from test_dashboard_api import PASSWORD, FakeLauncher, sign_in, write_feed


class FakeMailer(Mailer):
    backend = "fake"
    available = True

    def __init__(self, failing=()):
        self.sent = []
        self.failing = set(failing)

    def send(self, to, name, subject, text, html_body):
        if to in self.failing:
            raise OSError("mail server down")
        self.sent.append({"to": to, "name": name, "subject": subject, "text": text, "html": html_body})


@pytest.fixture
def mail_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDROFED_SECRET_KEY", "test-secret")
    monkeypatch.setenv("HYDROFED_FRONTEND_URL", "https://dashboard.example.com")
    database = tmp_path / "experiments.db"
    users = UserStore(database)
    users.create("admin@example.com", "Ada Admin", "admin", PASSWORD)
    users.create("second@example.com", "Sam Second", "admin", PASSWORD)
    users.create("operator@example.com", "Otis Operator", "operator", PASSWORD, plant_id="plant_1")
    users.create("other@example.com", "Olu Other", "operator", PASSWORD, plant_id="plant_2")
    mailer = FakeMailer()
    app = create_app(
        database,
        feed_path=write_feed(tmp_path / "feed"),
        comparison_path=tmp_path / "missing",
        partition_summary_path=tmp_path / "summary.json",
        launcher=FakeLauncher(),
        clock=lambda: REPLAY_EPOCH + 70,
        mailer=mailer,
    )
    return TestClient(app), users, mailer


def recipients_of(mailer):
    return sorted(email["to"] for email in mailer.sent)


def test_an_operator_message_emails_every_administrator(mail_setup):
    client, _, mailer = mail_setup
    operator = sign_in(client, "operator@example.com")

    client.post("/api/messages/plant_1", json={"body": "Pump 2 is noisy.\nCan someone check?"}, headers=operator)

    assert recipients_of(mailer) == ["admin@example.com", "second@example.com"]
    email = mailer.sent[0]
    assert email["subject"] == "Otis Operator wrote about Plant 1"
    assert "Pump 2 is noisy." in email["text"]
    assert "https://dashboard.example.com/#/messages/plant_1" in email["text"]
    assert "<p>Can someone check?</p>" in email["html"]


def test_an_administrator_reply_emails_only_that_plants_operator(mail_setup):
    client, _, mailer = mail_setup
    admin = sign_in(client, "admin@example.com")

    client.post("/api/messages/plant_1", json={"body": "Please send today's readings."}, headers=admin)

    assert recipients_of(mailer) == ["operator@example.com"]


def test_one_email_per_conversation_until_it_is_read(mail_setup):
    client, _, mailer = mail_setup
    operator = sign_in(client, "operator@example.com")
    admin = sign_in(client, "admin@example.com")

    client.post("/api/messages/plant_1", json={"body": "First"}, headers=operator)
    client.post("/api/messages/plant_1", json={"body": "Second"}, headers=operator)
    assert recipients_of(mailer) == ["admin@example.com", "second@example.com"]

    client.get("/api/messages/plant_1", headers=admin)
    client.post("/api/messages/plant_1", json={"body": "Third"}, headers=operator)
    # Ada read the conversation, so she hears about the third; Sam has not read the first yet.
    assert recipients_of(mailer) == ["admin@example.com", "admin@example.com", "second@example.com"]


def test_people_can_turn_the_emails_off(mail_setup):
    client, _, mailer = mail_setup
    admin = sign_in(client, "admin@example.com")
    operator = sign_in(client, "operator@example.com")

    assert client.get("/api/auth/preferences", headers=admin).json() == {
        "notify_messages": True, "email_available": True,
    }
    changed = client.put("/api/auth/preferences", json={"notify_messages": False}, headers=admin).json()
    assert changed["notify_messages"] is False

    client.post("/api/messages/plant_1", json={"body": "Hello"}, headers=operator)
    assert recipients_of(mailer) == ["second@example.com"]


def test_disabled_accounts_get_no_email(mail_setup):
    client, users, mailer = mail_setup
    users.update(users.find_by_email("second@example.com")["user_id"], status="disabled")
    operator = sign_in(client, "operator@example.com")

    client.post("/api/messages/plant_1", json={"body": "Hello"}, headers=operator)

    assert recipients_of(mailer) == ["admin@example.com"]


def test_a_failed_email_does_not_lose_the_message(mail_setup):
    client, _, mailer = mail_setup
    mailer.failing.add("admin@example.com")
    operator = sign_in(client, "operator@example.com")

    response = client.post("/api/messages/plant_1", json={"body": "Hello"}, headers=operator)

    assert response.status_code == 201
    assert recipients_of(mailer) == ["second@example.com"]
    assert len(client.get("/api/messages/plant_1", headers=operator).json()["messages"]) == 1


def test_alert_messages_say_which_alert():
    message = {
        "id": 1, "plant": "plant_1", "body": "Real?",
        "author": {"name": "Otis Operator", "role_label": "Plant operator"},
        "alert": {"id": "plant_1:0:0", "severity": "critical", "started_at": 1_793_491_200},
    }

    subject, text, html_body = compose(message, "Plant 1", "https://dashboard.example.com/#/messages/plant_1")

    assert "About the critical alert from 01 Nov 2026, 00:00 UTC." in text
    assert "critical alert" in html_body


def test_email_is_off_without_settings(monkeypatch):
    for name in ("HYDROFED_EMAIL_FROM", "HYDROFED_BREVO_API_KEY", "HYDROFED_SMTP_HOST"):
        monkeypatch.delenv(name, raising=False)
    assert mailer_from_environment().available is False

    monkeypatch.setenv("HYDROFED_SMTP_HOST", "smtp.gmail.com")
    assert mailer_from_environment().available is False  # still needs a sender

    monkeypatch.setenv("HYDROFED_EMAIL_FROM", "HydroFed <alerts@example.com>")
    assert isinstance(mailer_from_environment(), SmtpMailer)

    monkeypatch.setenv("HYDROFED_BREVO_API_KEY", "key")
    assert isinstance(mailer_from_environment(), BrevoMailer)


def test_brevo_receives_the_email_over_https():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers["api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"messageId": "1"})

    mailer = BrevoMailer("HydroFed <alerts@example.com>", "secret", httpx.Client(transport=httpx.MockTransport(handler)))
    mailer.send("ada@example.com", "Ada Admin", "Subject", "Text", "<p>Text</p>")

    assert seen["url"] == notifications.BREVO_URL
    assert seen["key"] == "secret"
    assert seen["body"]["sender"] == {"name": "HydroFed", "email": "alerts@example.com"}
    assert seen["body"]["to"] == [{"email": "ada@example.com", "name": "Ada Admin"}]


def test_smtp_uses_starttls_and_signs_in(monkeypatch):
    calls = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            calls.append(("starttls",))

        def login(self, username, password):
            calls.append(("login", username))

        def send_message(self, message):
            calls.append(("send", message["To"], message["Subject"]))

    monkeypatch.setattr(notifications.smtplib, "SMTP", FakeSMTP)
    SmtpMailer("HydroFed <alerts@example.com>", "smtp.gmail.com", 587, "alerts@example.com", "app-password").send(
        "ada@example.com", "Ada Admin", "Subject", "Text", "<p>Text</p>",
    )

    assert calls == [
        ("connect", "smtp.gmail.com", 587),
        ("starttls",),
        ("login", "alerts@example.com"),
        ("send", "Ada Admin <ada@example.com>", "Subject"),
    ]
