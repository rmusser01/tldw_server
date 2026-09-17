"""Prove normal return from the frozen initializer without trusting CLI exit zero."""

import asyncio
import importlib
import json
import os
from pathlib import Path


def main() -> None:
    """Write this attempt's proof only after the existing initializer returns."""
    request = json.loads(os.environ["MATRIX_INIT_REQUEST"])
    receipt_path = Path(request["receiptPath"])
    if receipt_path.exists():
        raise FileExistsError("Initialization attempt receipt already exists")
    initializer = importlib.import_module("tldw_Server_API.app.core.AuthNZ.initialize")
    if not Path(initializer.__file__).resolve().is_relative_to(Path(request["sourceRoot"]).resolve()):
        raise RuntimeError("Initializer is outside the frozen source root")
    asyncio.run(initializer.main(non_interactive=True))
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(
            {"status": "completed", "token": request["token"], "preparationHash": request["preparationHash"]}, output
        )


if __name__ == "__main__":
    main()
