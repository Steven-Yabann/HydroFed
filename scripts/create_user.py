"""Add a HydroFed dashboard account from the command line.

Use it to create the first administrator, who can then add everyone else
from the dashboard:

    .venv/bin/python scripts/create_user.py --email you@example.com --name "Your Name" --role admin

Operators also need ``--plant`` (for example ``--plant plant_1``): they see
only that plant and its alerts. You are asked for a password unless
``--sso-only`` is given, in which case the person signs in with Google or
Microsoft using the same email.
"""

from __future__ import annotations

import argparse
import getpass
import os

from hydrofed.auth import ROLE_PERMISSIONS, AuthError, UserStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", choices=sorted(ROLE_PERMISSIONS), default="operator")
    parser.add_argument("--db", default=os.environ.get("HYDROFED_DB_PATH", "artifacts/experiments.db"))
    parser.add_argument("--plant", help="the plant an operator monitors, such as plant_1")
    parser.add_argument("--sso-only", action="store_true", help="create the account without a password")
    args = parser.parse_args()
    if "view_all_plants" not in ROLE_PERMISSIONS[args.role] and not args.plant:
        parser.error("operators need --plant")

    password = None
    if not args.sso_only:
        password = getpass.getpass("Password (at least 10 characters): ")
        if password != getpass.getpass("Repeat password: "):
            raise SystemExit("Passwords did not match.")
    try:
        user = UserStore(args.db).create(args.email, args.name, args.role, password, plant_id=args.plant)
    except AuthError as error:
        raise SystemExit(str(error)) from error
    print(f"Created {user['role_label'].lower()} {user['email']} (id {user['user_id']}).")


if __name__ == "__main__":
    main()
