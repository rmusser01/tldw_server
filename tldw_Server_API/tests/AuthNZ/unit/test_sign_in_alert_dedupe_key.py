"""new_sign_in_dedupe_key groups sign-ins the way the storm guard needs (TASK-15649)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tldw_Server_API.app.core.AuthNZ.sign_in_alert import (
    NEW_SIGN_IN_DEDUPE_BUCKET_SECONDS,
    new_sign_in_dedupe_key,
)
from tldw_Server_API.app.core.AuthNZ.single_user_session import SESSION_DEVICE_ID

NOW = datetime(2026, 11, 25, 12, 0, 30, tzinfo=timezone.utc)


def _key(*, device_id=None, ip="203.0.113.7", ua="UA-1", now=NOW, user_id=1) -> str:
    return new_sign_in_dedupe_key(user_id=user_id, device_id=device_id, ip_address=ip, user_agent=ua, now=now)


def test_same_install_groups_even_when_ip_and_user_agent_differ():
    assert _key(device_id="dev-aaaaaaaa", ip="1.1.1.1", ua="x") == _key(device_id="dev-aaaaaaaa", ip="2.2.2.2", ua="y")


def test_different_installs_do_not_group():
    assert _key(device_id="dev-aaaaaaaa") != _key(device_id="dev-bbbbbbbb")


def test_without_an_install_id_the_ip_and_user_agent_group():
    assert _key() == _key(device_id=None)
    assert _key() != _key(ip="198.51.100.9")
    assert _key() != _key(ua="UA-2")


def test_the_bare_single_user_cookie_tag_is_not_a_group():
    assert _key(device_id=SESSION_DEVICE_ID) == _key(device_id=None)
    assert _key(device_id=SESSION_DEVICE_ID, ip="198.51.100.9") != _key(device_id=SESSION_DEVICE_ID)


def test_a_new_bucket_gives_a_new_key():
    later = NOW + timedelta(seconds=NEW_SIGN_IN_DEDUPE_BUCKET_SECONDS)
    assert _key(device_id="dev-aaaaaaaa") != _key(device_id="dev-aaaaaaaa", now=later)


def test_other_users_never_share_a_key():
    assert _key(user_id=1) != _key(user_id=2)


def test_the_key_is_a_hash_that_never_contains_the_install_id_ip_or_user_agent():
    key = _key(device_id="dev-aaaaaaaa", ip="203.0.113.7", ua="UA-1")
    assert key.startswith("new_sign_in:")
    assert all(part not in key for part in ("dev-aaaaaaaa", "203.0.113.7", "UA-1"))
