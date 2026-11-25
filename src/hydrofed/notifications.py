"""Email people when a message arrives in a plant conversation.

A message from an operator goes to the administrators; a message from an
administrator goes to the plant's operator. Each person gets one email per
conversation and no more until they have read it, so a busy conversation
does not flood their inbox. Anyone can turn the emails off under Your
account.

Email goes out over SMTP (for example Gmail with an app password) or over
Brevo's HTTPS API, which still works where a host blocks SMTP ports:

- ``HYDROFED_EMAIL_FROM``: the sender, such as ``HydroFed <alerts@example.com>``
- ``HYDROFED_BREVO_API_KEY``: send through Brevo, or
- ``HYDROFED_SMTP_HOST``, ``HYDROFED_SMTP_PORT`` (587, or 465 for SSL),
  ``HYDROFED_SMTP_USERNAME`` and ``HYDROFED_SMTP_PASSWORD``

With neither configured, no email is sent.
"""

from __future__ import annotations

import html
import logging
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formataddr, parseaddr

import httpx

log = logging.getLogger("hydrofed.notifications")

BREVO_URL = "https://api.brevo.com/v3/smtp/email"


class Mailer:
    """Sends nothing; the stand-in when email is not configured."""

    backend = "off"
    available = False

    def send(self, to: str, name: str, subject: str, text: str, html_body: str) -> None:
        return None


class SmtpMailer(Mailer):
    backend = "smtp"
    available = True

    def __init__(self, sender: str, host: str, port: int = 587, username: str = "", password: str = ""):
        self.sender, self.host, self.port = sender, host, port
        self.username, self.password = username, password

    def send(self, to: str, name: str, subject: str, text: str, html_body: str) -> None:
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = formataddr((name, to))
        message["Subject"] = subject
        message.set_content(text)
        message.add_alternative(html_body, subtype="html")
        connection = smtplib.SMTP_SSL if self.port == 465 else smtplib.SMTP
        with connection(self.host, self.port, timeout=20) as server:
            if self.port != 465:
                server.starttls()
            if self.username:
                server.login(self.username, self.password)
            server.send_message(message)


class BrevoMailer(Mailer):
    backend = "brevo"
    available = True

    def __init__(self, sender: str, api_key: str, http: httpx.Client | None = None):
        self.sender_name, self.sender_email = parseaddr(sender)
        self.api_key = api_key
        self.http = http or httpx.Client(timeout=20)

    def send(self, to: str, name: str, subject: str, text: str, html_body: str) -> None:
        response = self.http.post(
            BREVO_URL,
            headers={"api-key": self.api_key, "accept": "application/json"},
            json={
                "sender": {"name": self.sender_name or "HydroFed", "email": self.sender_email},
                "to": [{"email": to, "name": name}],
                "subject": subject,
                "textContent": text,
                "htmlContent": html_body,
            },
        )
        response.raise_for_status()


def mailer_from_environment() -> Mailer:
    sender = os.environ.get("HYDROFED_EMAIL_FROM", "").strip()
    if not sender or "@" not in sender:
        return Mailer()
    if os.environ.get("HYDROFED_BREVO_API_KEY"):
        return BrevoMailer(sender, os.environ["HYDROFED_BREVO_API_KEY"])
    if os.environ.get("HYDROFED_SMTP_HOST"):
        return SmtpMailer(
            sender,
            os.environ["HYDROFED_SMTP_HOST"],
            int(os.environ.get("HYDROFED_SMTP_PORT", "587")),
            os.environ.get("HYDROFED_SMTP_USERNAME", ""),
            os.environ.get("HYDROFED_SMTP_PASSWORD", ""),
        )
    return Mailer()


def recipients(accounts: list[dict], plant: str, author: dict) -> list[dict]:
    """Who hears about a message: the other side of the conversation."""
    author_is_admin = "view_all_plants" in author["permissions"]
    chosen = []
    for account in accounts:
        if account["user_id"] == author["user_id"] or account["status"] != "active":
            continue
        if not account.get("notify_messages", True):
            continue
        is_admin = "view_all_plants" in account["permissions"]
        if author_is_admin and not is_admin and account.get("plant_id") == plant:
            chosen.append(account)
        elif not author_is_admin and is_admin:
            chosen.append(account)
    return chosen


def compose(message: dict, plant_name: str, link: str) -> tuple[str, str, str]:
    """Subject, plain text and HTML for one new message."""
    author = message["author"]
    subject = f"{author['name']} wrote about {plant_name}"
    alert_line = ""
    if message.get("alert"):
        started = datetime.fromtimestamp(message["alert"]["started_at"], tz=timezone.utc)
        alert_line = f"About the {message['alert']['severity']} alert from {started:%d %b %Y, %H:%M} UTC."
    footer = (
        "You will not get another email about this conversation until you have read it. "
        "To stop these emails, turn them off under Your account on the dashboard."
    )
    text = "\n\n".join(part for part in (
        f"{author['name']} ({author['role_label']}) wrote in the {plant_name} conversation on HydroFed:",
        message["body"],
        alert_line,
        f"Read and reply: {link}",
        footer,
    ) if part)
    paragraphs = "".join(f"<p>{html.escape(line)}</p>" for line in message["body"].split("\n") if line.strip())
    html_body = (
        '<div style="font-family:Arial,sans-serif;font-size:15px;line-height:1.5;color:#0f1f17;max-width:560px">'
        f"<p>{html.escape(author['name'])} ({html.escape(author['role_label'])}) wrote in the "
        f"<strong>{html.escape(plant_name)}</strong> conversation on HydroFed:</p>"
        f'<div style="border-left:3px solid #16855a;padding:4px 14px;margin:12px 0">{paragraphs}</div>'
        + (f'<p style="color:#4a5a52">{html.escape(alert_line)}</p>' if alert_line else "")
        + f'<p><a href="{html.escape(link)}" style="color:#0f5f3e;font-weight:bold">Read and reply</a></p>'
        f'<p style="color:#7b8a82;font-size:13px">{html.escape(footer)}</p></div>'
    )
    return subject, text, html_body


def notify_new_message(
    mailer: Mailer,
    accounts: list[dict],
    unread_for,
    message: dict,
    author: dict,
    plant_name: str,
    link: str,
) -> list[str]:
    """Email everyone who should hear about ``message``; return who was emailed.

    ``unread_for(account)`` gives the account's unread count in the
    conversation. Only a count of one (this message) sends an email.
    """
    if not mailer.available:
        return []
    subject, text, html_body = compose(message, plant_name, link)
    sent = []
    for account in recipients(accounts, message["plant"], author):
        if unread_for(account) != 1:
            continue
        try:
            mailer.send(account["email"], account["name"], subject, text, html_body)
            sent.append(account["email"])
        except Exception:  # an email failure must never lose the message itself
            log.exception("Could not email user %s about message %s", account["user_id"], message["id"])
    return sent
