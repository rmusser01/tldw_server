"""Offline regression controls for the candidate's real APT acquisition policy.

Run with the pinned Debian interpreter in a networkless container. Only its
loopback server is used; no package is installed or repository trust changed.
"""

import hashlib
import shlex
import subprocess  # nosec B404
import tempfile
import threading
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CONFIG = Path("/etc/apt/apt.conf.d/80candidate-acquisition")
PAYLOAD = b"candidate acquisition regression fixture\n"


class AcquisitionTests(unittest.TestCase):
    """Exercise recovery, exhaustion and integrity with APT, not a command fake."""

    def test_transient_503_recovers_beyond_default_retry_budget(self):
        self.acquire("transient", expected_exit=0, expected_requests=5)

    def test_persistent_503_stops_after_five_retries(self):
        self.acquire("persistent", expected_exit=100, expected_requests=6)

    def test_corrupt_success_response_fails_hash_verification(self):
        self.acquire("corrupt", expected_exit=100, expected_requests=1)

    def test_repository_requests_use_alternate_endpoint_for_same_snapshot(self):
        """Catch wrong Dockerfile source wiring, suite drift, or extra repositories."""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(  # nosec B603
                [
                    "/usr/bin/apt-get",
                    "-o",
                    f"Dir::State::lists={directory}/lists",
                    "--print-uris",
                    "update",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        requests = [shlex.split(line)[0] for line in result.stdout.splitlines() if line.strip()]
        expected = [
            f"https://snapshot-cloudflare.debian.org/archive/{archive}/20260906T000000Z/dists/{suite}/{index}"
            for archive, suite in (
                ("debian", "trixie"),
                ("debian", "trixie-updates"),
                ("debian-security", "trixie-security"),
            )
            for index in (
                "InRelease",
                "main/source/Sources.xz",
                "main/binary-amd64/Packages.xz",
                "main/binary-all/Packages.xz",
            )
        ]
        self.assertCountEqual(requests, expected)

    def acquire(self, scenario: str, expected_exit: int, expected_requests: int) -> None:
        self.assertTrue(CONFIG.is_file(), "candidate acquisition configuration is missing")
        requests = Counter()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests["count"] += 1
                failing = scenario == "persistent" or (scenario == "transient" and requests["count"] <= 4)
                content = b"No healthy backends" if failing else PAYLOAD
                if scenario == "corrupt":
                    content = b"corrupted package bytes"
                self.send_response(503 if failing else 200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, *_args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, tempfile.TemporaryDirectory() as directory:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            destination = Path(directory) / "download"
            try:
                result = subprocess.run(  # nosec B603
                    [
                        "/usr/lib/apt/apt-helper",
                        "-c",
                        str(CONFIG),
                        # Test-only: remove waiting while preserving real retry decisions.
                        "-o",
                        "Acquire::Retries::Delay=false",
                        "download-file",
                        f"http://127.0.0.1:{server.server_port}/fixture",
                        str(destination),
                        "SHA256:" + hashlib.sha256(PAYLOAD).hexdigest(),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, expected_exit, result.stdout + result.stderr)
                self.assertEqual(requests["count"], expected_requests)
                if expected_exit == 0:
                    self.assertEqual(destination.read_bytes(), PAYLOAD)
                else:
                    self.assertFalse(destination.exists(), "failed download must not become a usable artifact")
                if scenario == "corrupt":
                    self.assertIn("Hash Sum mismatch", result.stderr)
            finally:
                server.shutdown()
                worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
