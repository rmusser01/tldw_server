"""Invoke the normal admin bootstrap without putting planned credentials in argv."""

import asyncio
import importlib
import json
import os
import stat
from pathlib import Path


def main() -> None:
    """Certify only a normal true return from the frozen supported function."""
    if os.environ.get("AUTH_MODE") != "multi_user":
        raise RuntimeError("Admin bootstrap requires multi_user mode")
    request = json.loads(os.environ["MATRIX_ADMIN_REQUEST"])
    receipt_path = Path(request["receiptPath"])
    if receipt_path.exists():
        raise FileExistsError("Admin bootstrap attempt receipt already exists")
    credentials_path = Path(request["credentialsPath"])
    if stat.S_IMODE(credentials_path.stat().st_mode) != 0o600:
        raise RuntimeError("Planned credentials must be mode0600")
    credentials = json.loads(credentials_path.read_text())
    admin = credentials.get("accounts", {}).get("admin", {})
    if (
        credentials.get("accountStatus") != "planned-not-created"
        or not isinstance(admin, dict)
        or any(
            not isinstance(admin.get(field), str) or not admin[field].strip()
            for field in ("username", "password", "email")
        )
    ):
        raise RuntimeError("Valid planned admin credentials are required")
    bootstrap = importlib.import_module("tldw_Server_API.app.core.AuthNZ.create_admin")
    if not Path(bootstrap.__file__).resolve().is_relative_to(Path(request["sourceRoot"]).resolve()):
        raise RuntimeError("Admin bootstrap is outside the frozen source root")
    success = asyncio.run(
        bootstrap.create_admin_user_non_interactive(
            username=admin["username"], password=admin["password"], email=admin["email"]
        )
    )
    if success is not True:
        raise RuntimeError("Admin bootstrap did not complete successfully")
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(
            {
                "status": "completed",
                "action": "bootstrap-admin",
                "token": request["token"],
                "preparationHash": request["preparationHash"],
            },
            output,
        )


if __name__ == "__main__":
    main()
