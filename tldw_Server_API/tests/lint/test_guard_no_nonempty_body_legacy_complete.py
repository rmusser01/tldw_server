"""Exercise the actual pre-commit guard, including its endpoint boundary."""

from pathlib import Path

import pytest
from Helper_Scripts.checks import guard_no_nonempty_legacy_complete as guard

pytestmark = pytest.mark.unit


def test_no_nonempty_body_post_to_legacy_complete() -> None:
    """Scan the real repository tests using the same guard as pre-commit."""
    assert guard.main() == 0


@pytest.mark.parametrize(
    "suffix,body,expected",
    [
        ("complete", '{"message": "hello"}', 1),
        ("complete?stream=true", '{"message": "hello"}', 1),
        ("complete/", '{"message": "hello"}', 1),
        ("complete", "{}", 0),
        ("complete-v2", '{"message": "hello"}', 0),
        ("completions", '{"message": "hello"}', 0),
    ],
)
def test_guard_matches_only_legacy_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    suffix: str, body: str, expected: int,
) -> None:
    """Reject legacy payloads while permitting their documented replacements."""
    tests_root = tmp_path / "tldw_Server_API" / "tests"
    tests_root.mkdir(parents=True)
    (tests_root / "test_example.py").write_text(
        f'client.post("/api/v1/chats/123/{suffix}", json={body})\n', encoding="utf-8",
    )
    monkeypatch.setattr(guard, "__file__", str(tmp_path / "Helper_Scripts" / "checks" / "guard.py"))
    assert guard.main() == expected
