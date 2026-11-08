"""User accounts, sessions and single sign-on for the HydroFed dashboard.

Accounts live in the experiment SQLite database. An administrator adds each
person with a role before they can sign in, whether by password or through
Google or Microsoft. Single sign-on only proves who someone is; it never
creates an account, because plant telemetry should not be open to any
Google or Microsoft user.

Sessions are signed JSON Web Tokens. Set ``HYDROFED_SECRET_KEY`` in
deployment; without it a random key is kept next to the database.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import bcrypt
import jwt

# Each role maps to the permissions the API checks. Adding a role, such as a
# maintenance engineer or researcher, only needs an entry here.
ROLE_PERMISSIONS: dict[str, tuple[str, ...]] = {
    "operator": ("view_fleet", "review_alerts", "view_performance"),
    "admin": (
        "view_fleet", "review_alerts", "view_performance",
        "manage_federation", "manage_nodes", "manage_users",
    ),
}
ROLE_LABELS = {"operator": "Plant operator", "admin": "System administrator"}
STATUSES = ("active", "disabled")
SESSION_HOURS = 8
MIN_PASSWORD_LENGTH = 10

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    password_hash TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    last_provider TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at TEXT,
    created_by INTEGER
);
"""

USER_COLUMNS = "user_id, email, name, role, status, last_provider, created_at, last_login_at, password_hash"


class AuthError(Exception):
    """A sign-in or account change that must be refused."""


def hash_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Passwords need at least {MIN_PASSWORD_LENGTH} characters.")
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def _public(row: tuple) -> dict:
    user = dict(zip(USER_COLUMNS.split(", "), row))
    user["has_password"] = bool(user.pop("password_hash"))
    user["permissions"] = list(ROLE_PERMISSIONS.get(user["role"], ()))
    user["role_label"] = ROLE_LABELS.get(user["role"], user["role"])
    return user


class UserStore:
    """Read and change dashboard accounts in the experiment database."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as database:
            database.execute(USERS_SCHEMA)

    def _query(self, sql: str, parameters: tuple = ()) -> list[tuple]:
        with sqlite3.connect(self.path) as database:
            return database.execute(sql, parameters).fetchall()

    def get(self, user_id: int) -> dict | None:
        rows = self._query(f"SELECT {USER_COLUMNS} FROM users WHERE user_id = ?", (user_id,))
        return _public(rows[0]) if rows else None

    def find_by_email(self, email: str) -> dict | None:
        rows = self._query(f"SELECT {USER_COLUMNS} FROM users WHERE email = ?", (email.strip(),))
        return _public(rows[0]) if rows else None

    def list(self) -> list[dict]:
        return [_public(row) for row in self._query(f"SELECT {USER_COLUMNS} FROM users ORDER BY name")]

    def count_active_admins(self) -> int:
        return self._query("SELECT COUNT(*) FROM users WHERE role = 'admin' AND status = 'active'")[0][0]

    def create(
        self,
        email: str,
        name: str,
        role: str,
        password: str | None = None,
        created_by: int | None = None,
    ) -> dict:
        email = email.strip().lower()
        if "@" not in email or email.startswith("@") or email.endswith("@"):
            raise AuthError("Enter a valid email address.")
        if role not in ROLE_PERMISSIONS:
            raise AuthError(f"Unknown role: {role}")
        if not name.strip():
            raise AuthError("Enter the person's name.")
        password_hash = hash_password(password) if password else None
        try:
            with sqlite3.connect(self.path) as database:
                cursor = database.execute(
                    "INSERT INTO users(email, name, role, password_hash, created_by) VALUES(?, ?, ?, ?, ?)",
                    (email, name.strip(), role, password_hash, created_by),
                )
        except sqlite3.IntegrityError as error:
            raise AuthError(f"{email} already has an account.") from error
        return self.get(cursor.lastrowid)

    def update(
        self,
        user_id: int,
        name: str | None = None,
        role: str | None = None,
        status: str | None = None,
        password: str | None = None,
    ) -> dict:
        user = self.get(user_id)
        if user is None:
            raise AuthError("That account does not exist.")
        if role is not None and role not in ROLE_PERMISSIONS:
            raise AuthError(f"Unknown role: {role}")
        if status is not None and status not in STATUSES:
            raise AuthError(f"Unknown status: {status}")
        loses_admin = user["role"] == "admin" and user["status"] == "active" and (
            (role is not None and role != "admin") or status == "disabled"
        )
        if loses_admin and self.count_active_admins() <= 1:
            raise AuthError("HydroFed needs at least one active administrator.")

        changes = {
            "name": name.strip() if name is not None else None,
            "role": role,
            "status": status,
            "password_hash": hash_password(password) if password else None,
        }
        changes = {column: value for column, value in changes.items() if value is not None}
        if changes:
            assignments = ", ".join(f"{column} = ?" for column in changes)
            with sqlite3.connect(self.path) as database:
                database.execute(
                    f"UPDATE users SET {assignments} WHERE user_id = ?",
                    (*changes.values(), user_id),
                )
        return self.get(user_id)

    def delete(self, user_id: int) -> None:
        user = self.get(user_id)
        if user is None:
            raise AuthError("That account does not exist.")
        if user["role"] == "admin" and user["status"] == "active" and self.count_active_admins() <= 1:
            raise AuthError("HydroFed needs at least one active administrator.")
        with sqlite3.connect(self.path) as database:
            database.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    def record_login(self, user_id: int, provider: str) -> None:
        with sqlite3.connect(self.path) as database:
            database.execute(
                "UPDATE users SET last_login_at = CURRENT_TIMESTAMP, last_provider = ? WHERE user_id = ?",
                (provider, user_id),
            )

    def authenticate(self, email: str, password: str) -> dict:
        rows = self._query(f"SELECT {USER_COLUMNS} FROM users WHERE email = ?", (email.strip(),))
        # Compare against a dummy hash for unknown emails so timing does not
        # reveal which addresses have accounts.
        stored_hash = rows[0][-1] if rows else None
        valid = verify_password(password, stored_hash or _DUMMY_HASH) and stored_hash is not None
        if not valid:
            raise AuthError("Email or password is incorrect.")
        user = _public(rows[0])
        if user["status"] != "active":
            raise AuthError("This account has been disabled. Contact your administrator.")
        self.record_login(user["user_id"], "password")
        return self.get(user["user_id"])


def bootstrap_admin(users: UserStore) -> dict | None:
    """Create the first administrator from the environment on a fresh database.

    Hosted deployments have no shell to run ``scripts/create_user.py``, so
    ``HYDROFED_ADMIN_EMAIL`` (with ``HYDROFED_ADMIN_NAME`` and optionally
    ``HYDROFED_ADMIN_PASSWORD``) creates one when no active administrator
    exists. Without a password the account signs in with Google or Microsoft.
    Once an administrator exists the variables are ignored.
    """
    email = os.environ.get("HYDROFED_ADMIN_EMAIL", "").strip()
    if not email or users.count_active_admins() > 0 or users.find_by_email(email):
        return None
    name = os.environ.get("HYDROFED_ADMIN_NAME", "").strip() or email.split("@")[0]
    password = os.environ.get("HYDROFED_ADMIN_PASSWORD") or None
    return users.create(email, name, "admin", password)


_DUMMY_HASH = bcrypt.hashpw(b"hydrofed-dummy-password", bcrypt.gensalt()).decode()


def secret_key(database_path: Path) -> str:
    """Return the signing key, keeping a generated one beside the database."""
    configured = os.environ.get("HYDROFED_SECRET_KEY")
    if configured:
        return configured
    key_file = database_path.parent / ".hydrofed_secret"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_urlsafe(48)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(key)
    key_file.chmod(0o600)
    return key


def issue_session(user: dict, key: str, hours: int = SESSION_HOURS) -> dict:
    expires = datetime.now(timezone.utc) + timedelta(hours=hours)
    token = jwt.encode(
        {"sub": str(user["user_id"]), "role": user["role"], "exp": expires, "typ": "session"},
        key,
        algorithm="HS256",
    )
    return {"token": token, "expires_at": expires.isoformat(timespec="seconds"), "user": user}


def read_session(token: str, key: str) -> int:
    try:
        claims = jwt.decode(token, key, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as error:
        raise AuthError("Your session has expired. Sign in again.") from error
    except jwt.InvalidTokenError as error:
        raise AuthError("Sign in to continue.") from error
    if claims.get("typ") != "session":
        raise AuthError("Sign in to continue.")
    return int(claims["sub"])


# --- Single sign-on (OpenID Connect authorisation code flow with PKCE) ---


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    jwks_url: str
    issuers: tuple[str, ...] = ()
    issuer_prefix: str = ""

    def accepts_issuer(self, issuer: str) -> bool:
        if self.issuers:
            return issuer in self.issuers
        return issuer.startswith(self.issuer_prefix) and issuer.endswith("/v2.0")


def configured_providers() -> dict[str, Provider]:
    """Return the providers whose client ID and secret are set."""
    providers = {}
    google_id = os.environ.get("HYDROFED_GOOGLE_CLIENT_ID")
    google_secret = os.environ.get("HYDROFED_GOOGLE_CLIENT_SECRET")
    if google_id and google_secret:
        providers["google"] = Provider(
            "google", "Google", google_id, google_secret,
            "https://accounts.google.com/o/oauth2/v2/auth",
            "https://oauth2.googleapis.com/token",
            "https://www.googleapis.com/oauth2/v3/certs",
            issuers=("https://accounts.google.com", "accounts.google.com"),
        )
    microsoft_id = os.environ.get("HYDROFED_MICROSOFT_CLIENT_ID")
    microsoft_secret = os.environ.get("HYDROFED_MICROSOFT_CLIENT_SECRET")
    if microsoft_id and microsoft_secret:
        tenant = os.environ.get("HYDROFED_MICROSOFT_TENANT", "organizations")
        base = f"https://login.microsoftonline.com/{tenant}"
        providers["microsoft"] = Provider(
            "microsoft", "Microsoft", microsoft_id, microsoft_secret,
            f"{base}/oauth2/v2.0/authorize",
            f"{base}/oauth2/v2.0/token",
            f"{base}/discovery/v2.0/keys",
            issuer_prefix="https://login.microsoftonline.com/",
        )
    return providers


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def authorization_redirect(provider: Provider, redirect_uri: str, key: str) -> tuple[str, str]:
    """Return the provider URL to visit and a signed cookie holding the flow state."""
    state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    verifier, challenge = pkce_pair()
    query = urlencode({
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "prompt": "select_account",
    })
    cookie = jwt.encode(
        {
            "provider": provider.name, "state": state, "nonce": nonce, "verifier": verifier,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=10), "typ": "oauth",
        },
        key,
        algorithm="HS256",
    )
    return f"{provider.authorize_url}?{query}", cookie


def read_flow_cookie(cookie: str | None, provider: str, state: str | None, key: str) -> dict:
    if not cookie or not state:
        raise AuthError("The sign-in attempt expired. Try again.")
    try:
        flow = jwt.decode(cookie, key, algorithms=["HS256"])
    except jwt.InvalidTokenError as error:
        raise AuthError("The sign-in attempt expired. Try again.") from error
    if flow.get("typ") != "oauth" or flow.get("provider") != provider:
        raise AuthError("The sign-in attempt did not match. Try again.")
    if not secrets.compare_digest(flow["state"], state):
        raise AuthError("The sign-in attempt did not match. Try again.")
    return flow


def exchange_code(provider: Provider, code: str, redirect_uri: str, verifier: str) -> str:
    """Swap an authorisation code for the provider's ID token."""
    import httpx

    response = httpx.post(
        provider.token_url,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": provider.client_id,
            "client_secret": provider.client_secret,
            "code_verifier": verifier,
        },
        headers={"Accept": "application/json"},
        timeout=10,
    )
    if response.status_code != 200 or "id_token" not in response.json():
        raise AuthError(f"{provider.label} did not confirm the sign-in.")
    return response.json()["id_token"]


def verify_id_token(provider: Provider, id_token: str, nonce: str) -> dict:
    """Check the ID token's signature, audience, issuer and nonce; return email and name."""
    try:
        signing_key = jwt.PyJWKClient(provider.jwks_url).get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token, signing_key.key, algorithms=["RS256"], audience=provider.client_id,
            options={"verify_iss": False},
        )
    except jwt.PyJWTError as error:
        raise AuthError(f"{provider.label} returned a token HydroFed could not verify.") from error
    if not provider.accepts_issuer(claims.get("iss", "")):
        raise AuthError(f"{provider.label} returned a token from an unexpected issuer.")
    if not secrets.compare_digest(claims.get("nonce", ""), nonce):
        raise AuthError("The sign-in attempt did not match. Try again.")
    return identity_from_claims(provider.name, claims)


def identity_from_claims(provider: str, claims: dict) -> dict:
    if provider == "google":
        if not claims.get("email_verified"):
            raise AuthError("Google has not verified this email address.")
        email = claims.get("email")
    else:
        email = claims.get("email") or claims.get("preferred_username")
    if not email:
        raise AuthError("The provider did not share an email address.")
    return {"email": email.lower(), "name": claims.get("name") or email}


def sso_user(users: UserStore, provider: str, identity: dict) -> dict:
    """Return the HydroFed account for a verified single sign-on identity."""
    user = users.find_by_email(identity["email"])
    if user is None:
        raise AuthError(
            f"{identity['email']} does not have a HydroFed account. Ask an administrator to add you."
        )
    if user["status"] != "active":
        raise AuthError("This account has been disabled. Contact your administrator.")
    users.record_login(user["user_id"], provider)
    return users.get(user["user_id"])
