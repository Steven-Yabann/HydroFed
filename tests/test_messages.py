from test_dashboard_api import app_setup, sign_in  # noqa: F401  (app_setup is a fixture)


def test_operator_and_admin_talk_in_the_plant_conversation(app_setup):
    client, _, _ = app_setup
    operator = sign_in(client, "operator@example.com")
    admin = sign_in(client, "admin@example.com")

    sent = client.post("/api/messages/plant_1", json={"body": "  Pump 2 is noisy today.  "}, headers=operator)
    assert sent.status_code == 201
    message = sent.json()["message"]
    assert message["body"] == "Pump 2 is noisy today."
    assert message["author"]["name"] == "Otis Operator"
    assert message["author"]["role_label"] == "Plant operator"
    assert message["mine"] is True

    inbox = client.get("/api/messages", headers=admin).json()
    assert inbox["unread"] == 1
    plant_1 = next(thread for thread in inbox["threads"] if thread["plant"] == "plant_1")
    assert plant_1["unread"] == 1
    assert plant_1["plant_name"] == "Plant 1"
    assert plant_1["last_message"]["body"] == "Pump 2 is noisy today."

    thread = client.get("/api/messages/plant_1", headers=admin).json()
    assert [item["body"] for item in thread["messages"]] == ["Pump 2 is noisy today."]
    assert thread["messages"][0]["mine"] is False
    assert client.get("/api/messages", headers=admin).json()["unread"] == 0

    client.post("/api/messages/plant_1", json={"body": "Thanks, sending someone."}, headers=admin)
    assert client.get("/api/messages", headers=operator).json()["unread"] == 1
    bodies = [item["body"] for item in client.get("/api/messages/plant_1", headers=operator).json()["messages"]]
    assert bodies == ["Pump 2 is noisy today.", "Thanks, sending someone."]
    assert client.get("/api/messages", headers=operator).json()["unread"] == 0


def test_operators_only_reach_their_own_plant_conversation(app_setup):
    client, _, _ = app_setup
    operator = sign_in(client, "operator@example.com")
    admin = sign_in(client, "admin@example.com")

    threads = client.get("/api/messages", headers=operator).json()["threads"]
    assert [thread["plant"] for thread in threads] == ["plant_1"]
    assert client.get("/api/messages/plant_2", headers=operator).status_code == 403
    assert client.post("/api/messages/plant_2", json={"body": "Hello"}, headers=operator).status_code == 403

    assert len(client.get("/api/messages", headers=admin).json()["threads"]) == 2
    assert client.get("/api/messages/plant_9", headers=admin).status_code == 404


def test_empty_and_overlong_messages_are_refused(app_setup):
    client, _, _ = app_setup
    operator = sign_in(client, "operator@example.com")

    assert client.post("/api/messages/plant_1", json={"body": "   "}, headers=operator).status_code == 400
    long = client.post("/api/messages/plant_1", json={"body": "x" * 2001}, headers=operator)
    assert long.status_code == 400
    assert "2000" in long.json()["detail"]


def test_a_message_can_refer_to_one_of_the_plants_alerts(app_setup):
    client, _, _ = app_setup
    operator = sign_in(client, "operator@example.com")
    admin = sign_in(client, "admin@example.com")
    alert = client.get("/api/alerts?plant=plant_1", headers=operator).json()["alerts"][0]

    message = client.post(
        "/api/messages/plant_1", json={"body": "Is this real?", "alert_id": alert["id"]}, headers=operator,
    ).json()["message"]
    assert message["alert"] == {"id": alert["id"], "severity": "critical", "started_at": alert["started_at"]}

    plant_2_alert = client.get("/api/alerts?plant=plant_2", headers=admin).json()["alerts"][0]["id"]
    wrong = client.post(
        "/api/messages/plant_1", json={"body": "Wrong plant", "alert_id": plant_2_alert}, headers=admin,
    )
    assert wrong.status_code == 400


def test_an_operator_without_a_plant_has_no_conversation(app_setup):
    client, users, _ = app_setup
    users.create("new@example.com", "Nia New", "operator", "correct horse battery")
    headers = sign_in(client, "new@example.com")

    assert client.get("/api/messages", headers=headers).json() == {"threads": [], "unread": 0}
    assert client.post("/api/messages/plant_1", json={"body": "Hi"}, headers=headers).status_code == 403
