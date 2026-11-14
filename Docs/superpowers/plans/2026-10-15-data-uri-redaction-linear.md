# Data-URI Redaction Is Linear Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The three chat-path data-URI redactors stop being quadratic on crafted input, by sharing the bounded pattern LLM usage accounting PR 1a already uses (TASK-13560).

**Architecture:** One compiled pattern, `DATA_URI_REDACT_RE`, lives in `core/Utils/image_validation.py` beside the existing `DATA_URI_PATTERN`. The four sites that redact data:image payloads from free text import it: `core/Chat/chat_service.py` (`_sanitize_data_uris`, used by `estimate_tokens_from_json`), `api/v1/endpoints/chat.py` (`_sanitize_json_for_rate_limit`), `api/v1/schemas/chat_validators.py` (`validate_request_size`, string path), and `core/LLM_Calls/usage_accounting.py` (`_prompt_estimate`, already bounded; its private copy is deleted). Each keeps its own replacement text (`<omitted>` or `<redacted>`).

**Tech Stack:** Python `re`, pytest.

**Spec:** `backlog/tasks/task-13560 - Data-URI-redaction-regex-is-quadratic-on-crafted-input-in-the-chat-path.md` (no design doc: a bounded bug fix). Context: finding `chat-7` in `Docs/superpowers/reviews/chat/2026-09-21-stage2-orchestration-streaming.md`.

## Global Constraints

- **Branch:** `fix/data-uri-redaction-linear` from `origin/dev`; the PR targets `dev`.
- **Pattern, verbatim:** `re.compile(r'(data:image[^,"\s]{0,128},)[^"\s]+', re.IGNORECASE)`, copied from `core/LLM_Calls/usage_accounting.py:70` at base.
- **Budget (AC #2):** a 500 KB message of repeated `data:image` markers is redacted in under two seconds at each call site.
- **Test first:** the RED run below happens before any source change.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit id `TASK-13560`, `git add -A backlog/tasks`.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **No new dependency.** Pillow, which `image_validation` imports, is already in `[project] dependencies`.
- **Out of scope (`chat-7`):** `core/Chat/prompt_cost_envelope.py`'s hand-rolled `_sanitize_data_uris` (linear already; its output feeds prompt fingerprints, so changing it changes fingerprints), the two token rounding rules, and the now-redundant outer `_sanitize_json_for_rate_limit` pass at the three `estimate_tokens_from_json(_sanitize_json_for_rate_limit(...))` call sites in `api/v1/endpoints/chat.py`.

## Review Focus

1. **A real data URI after many crafted markers** must still be redacted: the bound must not stop a later legitimate match. Test: `test_real_data_uri_after_crafted_markers_is_still_redacted`.
2. **A payload outside a `"url"` key at the endpoint** (`"content": "data:image/png;base64,..."`). The endpoint's old pattern matched only under `"url":`, and case-sensitively, so `_estimate_tokens_for_queue` counted the base64 as prompt text; it is now redacted in any key and any case, as `chat_service` already did. The three `estimate_tokens_from_json(_sanitize_json_for_rate_limit(...))` call sites (rate limit, billing reservation) do not move: the inner `chat_service` pass already redacted these. Test: `test_endpoint_queue_estimate_ignores_a_payload_outside_a_url_key`.
3. **An uppercase `DATA:IMAGE/...` URI in the size validator.** The validator's old pattern was case-sensitive and counted it; it is now redacted (URI schemes are case-insensitive). Test: `test_request_size_validator_redacts_an_uppercase_data_uri`.
4. **A header longer than 128 characters, or containing whitespace or a `"`** (`data:image/png; base64,...`, or a quoted parameter that `json.dumps` writes as `name=\"a.png\"`), is left whole and counted in full at all three sites; the old patterns redacted it. That is the bound that keeps redaction linear. It over-counts, the safe direction for an estimate, but at the size validator a large image with such a header is now rejected as too large where it was accepted. Such headers are malformed or very rare (browsers and SDKs write `data:image/<type>;base64,`), so this is accepted rather than widened. Test (the length case): `test_media_type_over_128_chars_is_counted_not_redacted`.
5. **An ordinary `data:image/png;base64,` payload** is still redacted by the service and the endpoint (the validator's is covered by `tests/Chat_NEW/unit/test_request_size_and_image_limits.py`). Test: `test_ordinary_data_uri_payload_is_still_redacted`.

---

### Task 1: One bounded redaction pattern for the chat path

**Files:**
- Create: `tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py`
- Modify: `tldw_Server_API/app/core/Utils/image_validation.py` (after `DATA_URI_PATTERN`)
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py` (imports; `_DATA_URI_RE`, `_sanitize_data_uris`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (the `image_validation` import; `_sanitize_json_for_rate_limit`)
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_validators.py` (imports; `DATA_URI_REDACT_PATTERN`; `validate_request_size`)
- Modify: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` (`import re`; `_DATA_URI_RE`; `_prompt_estimate`)
- Modify: `backlog/tasks/task-13560*.md` (backlog-py only)
- Add: this plan, `Docs/superpowers/plans/2026-10-15-data-uri-redaction-linear.md`

**Interfaces:**
- Produces: `tldw_Server_API.app.core.Utils.image_validation.DATA_URI_REDACT_RE`, a compiled `re.Pattern[str]`; group 1 is the `data:image/...,` prefix to keep.
- Removes: `chat_validators.DATA_URI_REDACT_PATTERN`, `chat_service._DATA_URI_RE`, `usage_accounting._DATA_URI_RE`. Nothing else in the repository references them (`git grep` at base: only their own modules).

Set once for every step: `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` and `export TLDW_TEST_NO_DOCKER=1`.

- [ ] **Step 1: Branch**

```bash
git fetch -q origin dev
git switch -c fix/data-uri-redaction-linear origin/dev
```

- [ ] **Step 2: Write the failing tests**

Create `tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py` with exactly:

```python
"""Data-URI redaction stays linear on crafted input (TASK-13560).

An unbounded ``(data:image[^,]*,)`` rescans to the end of the text for every
``data:image`` with no comma after it, so a 500 KB body of such markers took
seconds to redact. Each chat call site must handle it in well under two seconds.
The call sites now share one pattern, so the endpoint's estimate also redacts a
payload outside a "url" key and the size validator an uppercase data URI.
"""

import time

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas import chat_validators
from tldw_Server_API.app.core.Chat import chat_service

# ~510 KB of markers with no comma anywhere: every one of them would rescan to the end.
_CRAFTED = '"url":"data:image' * 30_000


def _assert_fast(fn) -> None:
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"redaction took {elapsed:.1f}s on {len(_CRAFTED)} chars"


@pytest.mark.unit
def test_chat_service_token_estimate_is_linear():
    _assert_fast(lambda: chat_service.estimate_tokens_from_json(_CRAFTED))


@pytest.mark.unit
def test_chat_endpoint_rate_limit_sanitizer_is_linear():
    _assert_fast(lambda: chat_endpoint._sanitize_json_for_rate_limit(_CRAFTED))


@pytest.mark.unit
def test_request_size_validator_is_linear(monkeypatch):
    monkeypatch.setattr(chat_validators, "MAX_REQUEST_SIZE", 10_000_000)
    _assert_fast(lambda: chat_validators.validate_request_size(_CRAFTED))


@pytest.mark.unit
@pytest.mark.parametrize(
    "redact",
    [
        chat_service._sanitize_data_uris,
        chat_endpoint._sanitize_json_for_rate_limit,
    ],
)
def test_ordinary_data_uri_payload_is_still_redacted(redact):
    body = '{"url": "data:image/png;base64,' + "A" * 400 + '"}'
    redacted = redact(body)
    assert "A" * 400 not in redacted
    assert "data:image/png;base64," in redacted


@pytest.mark.unit
def test_real_data_uri_after_crafted_markers_is_still_redacted():
    body = _CRAFTED + '"url":"data:image/png;base64,' + "A" * 400 + '"'
    assert "A" * 400 not in chat_service._sanitize_data_uris(body)


@pytest.mark.unit
def test_endpoint_queue_estimate_ignores_a_payload_outside_a_url_key():
    body = '{"messages":[{"role":"user","content":"data:image/png;base64,' + "A" * 4000 + '"}]}'
    assert "A" * 4000 not in chat_endpoint._sanitize_json_for_rate_limit(body)
    assert chat_endpoint._estimate_tokens_for_queue(body) < 100


@pytest.mark.unit
def test_request_size_validator_redacts_an_uppercase_data_uri(monkeypatch):
    monkeypatch.setattr(chat_validators, "MAX_REQUEST_SIZE", 1000)
    body = '{"url": "DATA:IMAGE/PNG;BASE64,' + "A" * 5000 + '"}'
    assert chat_validators.validate_request_size(body) is True


@pytest.mark.unit
def test_media_type_over_128_chars_is_counted_not_redacted():
    # The bound that keeps redaction linear: such a URI is left whole, so it is over-counted.
    body = "data:image/" + "x" * 200 + ",payload"
    assert chat_service._sanitize_data_uris(body) == body
```

- [ ] **Step 3: Run them to see them fail**

Run: `$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py`

Expected: `6 failed, 3 passed` (about 40 s, the quadratic runs take most of it). Measured on `origin/dev` 2026-10-15:
- the three `_is_linear` tests: `AssertionError: redaction took 5.4s on 510000 chars` (5.4 to 5.8 s each);
- `test_endpoint_queue_estimate_ignores_a_payload_outside_a_url_key`: `assert 'AAAA...' not in '{"messages"...'`;
- `test_request_size_validator_redacts_an_uppercase_data_uri`: `ValueError: Failed to validate request size: Request too large (max 1000 chars, got 5033)`;
- `test_media_type_over_128_chars_is_counted_not_redacted`: `assert 'data:image/x...xxx,<omitted>' == 'data:image/x...xxxxx,payload'`.

The three that pass are guards (ordinary redaction x2, a real URI after crafted markers): the old patterns already handle them.

- [ ] **Step 4: Apply the source change**

Apply exactly this diff (save it to a file and `git apply` it, or make the edits by hand):

```diff
diff --git a/tldw_Server_API/app/api/v1/endpoints/chat.py b/tldw_Server_API/app/api/v1/endpoints/chat.py
index 59962e9790..670f66a3cc 100644
--- a/tldw_Server_API/app/api/v1/endpoints/chat.py
+++ b/tldw_Server_API/app/api/v1/endpoints/chat.py
@@ -68,6 +68,7 @@ from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
 from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError
 from tldw_Server_API.app.core.Utils.base64url import SignatureMismatchError, verify_signed_token
 from tldw_Server_API.app.core.Utils.image_validation import (
+    DATA_URI_REDACT_RE,
     get_max_base64_bytes,
     validate_image_url,
 )
@@ -6299,8 +6300,7 @@ def _sanitize_json_for_rate_limit(request_json: str) -> str:
     token estimation reflects text size, not binary data.
     """
     try:
-        pattern = re.compile(r"(\"url\"\s*:\s*\"data:image[^,]*,)[^\"\s]+")
-        return pattern.sub(r"\1<omitted>", request_json)
+        return DATA_URI_REDACT_RE.sub(r"\1<omitted>", request_json)
     except _CHAT_ENDPOINT_NONCRITICAL_EXCEPTIONS:
         return request_json

diff --git a/tldw_Server_API/app/api/v1/schemas/chat_validators.py b/tldw_Server_API/app/api/v1/schemas/chat_validators.py
index abddcd6692..4fb749b116 100644
--- a/tldw_Server_API/app/api/v1/schemas/chat_validators.py
+++ b/tldw_Server_API/app/api/v1/schemas/chat_validators.py
@@ -12,6 +12,7 @@ from typing import Any, Optional
 from loguru import logger

 from tldw_Server_API.app.core.custom_openai_providers import iter_custom_openai_provider_names
+from tldw_Server_API.app.core.Utils.image_validation import DATA_URI_REDACT_RE

 #######################################################################################################################
 #
@@ -23,10 +24,6 @@ CONVERSATION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')
 # Valid character ID pattern (numeric or name)
 CHARACTER_ID_PATTERN = re.compile(r'^(\d+|[a-zA-Z0-9_\- ]{1,100})$')

-# Pre-compiled pattern for redacting base64 image data in request size validation
-# Matches data:image URIs and captures the prefix for redaction
-DATA_URI_REDACT_PATTERN = re.compile(r'(data:image[^,]*,)[^"\s]+')
-
 # Maximum tool definition size (reduced from 10KB to 5KB for security)
 MAX_TOOL_DEFINITION_SIZE = 5000  # characters

@@ -338,7 +335,7 @@ def validate_request_size(request_data: Any) -> bool:
         # If it's already a JSON string, avoid re-serializing: redact via regex only.
         if isinstance(request_data, str):
             # Use pre-compiled pattern to replace base64 payload with <redacted>
-            request_json = DATA_URI_REDACT_PATTERN.sub(r'\1<redacted>', request_data)
+            request_json = DATA_URI_REDACT_RE.sub(r'\1<redacted>', request_data)
         else:
             # Convert to a serializable dict and sanitize recursively
             raw = request_data.model_dump() if hasattr(request_data, 'model_dump') else request_data
diff --git a/tldw_Server_API/app/core/Chat/chat_service.py b/tldw_Server_API/app/core/Chat/chat_service.py
index 25ab2882ea..f84439aa33 100644
--- a/tldw_Server_API/app/core/Chat/chat_service.py
+++ b/tldw_Server_API/app/core/Chat/chat_service.py
@@ -222,6 +222,7 @@ from tldw_Server_API.app.core.Usage.pricing_catalog import (
 )
 from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage
 from tldw_Server_API.app.core.Utils.cpu_bound_handler import process_large_json_async
+from tldw_Server_API.app.core.Utils.image_validation import DATA_URI_REDACT_RE

 _CHAT_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
     ChatAPIError,
@@ -3023,13 +3024,10 @@ def estimate_tokens_from_json(request_json: str) -> int:
         return 1


-_DATA_URI_RE = re.compile(r'(data:image[^,]*,)[^"\s]+', re.IGNORECASE)
-
-
 def _sanitize_data_uris(text: str) -> str:
     """Redact data URI payloads to avoid inflating token estimates."""
     try:
-        return _DATA_URI_RE.sub(r"\1<omitted>", text)
+        return DATA_URI_REDACT_RE.sub(r"\1<omitted>", text)
     except _CHAT_NONCRITICAL_EXCEPTIONS:
         return text

diff --git a/tldw_Server_API/app/core/LLM_Calls/usage_accounting.py b/tldw_Server_API/app/core/LLM_Calls/usage_accounting.py
index 0b55a6cc91..12340bfc8a 100644
--- a/tldw_Server_API/app/core/LLM_Calls/usage_accounting.py
+++ b/tldw_Server_API/app/core/LLM_Calls/usage_accounting.py
@@ -11,7 +11,6 @@ import functools
 import inspect
 import json
 import math
-import re
 import time
 from collections.abc import AsyncIterator, Callable, Iterator
 from contextlib import suppress
@@ -28,6 +27,7 @@ from tldw_Server_API.app.core.Usage.llm_call_context import (
     in_accounted_call,
 )
 from tldw_Server_API.app.core.Usage.llm_quota_gate import check_user_async, check_user_sync, user_to_check
+from tldw_Server_API.app.core.Utils.image_validation import DATA_URI_REDACT_RE

 _ACCOUNTED_METHODS = ("chat", "achat", "stream", "astream")
 _MARK = "__llm_accounted__"
@@ -64,12 +64,6 @@ def _approx_tokens(char_count: int) -> int:
     return max(1, char_count // 4) if char_count > 0 else 0


-# Redacts base64 image payloads before the estimate, as the chat endpoint's estimator does.
-# The media-type part is bounded: an unbounded ``[^,]*`` rescans to the end of the text for
-# every ``data:image`` with no comma after it, which is quadratic on a crafted prompt.
-_DATA_URI_RE = re.compile(r'(data:image[^,"\s]{0,128},)[^"\s]+', re.IGNORECASE)
-
-
 def _prompt_estimate(request: Any) -> int:
     """Estimated prompt tokens: the messages plus the system prompt, with image payloads redacted."""
     try:
@@ -79,7 +73,7 @@ def _prompt_estimate(request: Any) -> int:
         system_message = request.get("system_message")
         if isinstance(system_message, str):
             text += system_message
-        return max(1, len(_DATA_URI_RE.sub(r"\1<omitted>", text)) // 4)
+        return max(1, len(DATA_URI_REDACT_RE.sub(r"\1<omitted>", text)) // 4)
     except Exception:  # noqa: BLE001 - an estimate must never fail a call
         return 1

diff --git a/tldw_Server_API/app/core/Utils/image_validation.py b/tldw_Server_API/app/core/Utils/image_validation.py
index 3440ed7092..5760ba9888 100644
--- a/tldw_Server_API/app/core/Utils/image_validation.py
+++ b/tldw_Server_API/app/core/Utils/image_validation.py
@@ -202,6 +202,12 @@ MAX_BASE64_STRING_LENGTH = int(MAX_BASE64_BYTES * 4 / 3) + 100
 # Regex pattern for data URI validation
 DATA_URI_PATTERN = re.compile(r'^data:([^;]+);base64,(.+)$')

+# Finds data:image payloads in free text (a request body, a prompt) for redaction; group 1 is
+# the "data:image/...," prefix to keep. The media-type part is bounded: an unbounded ``[^,]*``
+# rescans to the end of the text for every ``data:image`` with no comma after it, which is
+# quadratic on crafted input.
+DATA_URI_REDACT_RE = re.compile(r'(data:image[^,"\s]{0,128},)[^"\s]+', re.IGNORECASE)
+
 #######################################################################################################################
 #
 # Functions:
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py`

Expected: `9 passed`. The shared pattern redacts the 510 KB crafted body in about 5 ms.

- [ ] **Step 6: Run the tests that cover the changed redactors**

```bash
$PY -m pytest -p no:cacheprovider -q \
  tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py \
  tldw_Server_API/tests/Chat/unit/test_chat_endpoint_helpers.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_queue_estimate.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_token_estimates.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_request_size_and_image_limits.py \
  tldw_Server_API/tests/schemas/test_chat_validators.py \
  tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py \
  tldw_Server_API/tests/Resource_Governance/test_e2e_tokens_daily_cap.py \
  tldw_Server_API/tests/Chat/integration/test_chat_fixes_integration.py
```

Expected: `131 passed`.

- [ ] **Step 7: Run every test file that imports a changed module**

```bash
git grep -l -P 'core\.Utils\.image_validation|core\.Utils import [^\n]*image_validation|core\.LLM_Calls\.usage_accounting|core\.LLM_Calls import [^\n]*usage_accounting|core\.Chat\.chat_service\b|core\.Chat import [^\n]*\bchat_service\b|schemas\.chat_validators\b|schemas import [^\n]*\bchat_validators\b|v1\.endpoints\.chat\b|v1\.endpoints import [^\n]*\bchat\b' -- 'tldw_Server_API/tests/*.py' > .superpowers/sdd/t13560_files.txt
wc -l < .superpowers/sdd/t13560_files.txt
RUN_JOBS=1 RUN_EVALUATIONS=1 $PY -m pytest -p no:cacheprovider -q -n 4 $(cat .superpowers/sdd/t13560_files.txt) tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py
```

Use `-P`: with `-E`, macOS `git grep` reads `\b` as a literal `b`, so `endpoints\.chat\b` matches `chatbooks` and misses every module that imports the chat endpoint. The list is 189 files; the new test file is named separately because `git grep` does not see it until Step 9 commits it.

Expected: `2 failed, 5594 passed, 35 skipped` (about 18 minutes; measured on the prototype). Both failures also fail on unchanged `origin/dev` `d7e43120e6`, where the same 189 files gave `3 failed, 5584 passed, 35 skipped`:
- `Services/test_router_groups_contract.py::test_iter_admin_router_specs_populates_expected_specs` (`assert '/api/v1/admin' == '/api/v1'`): commit `9ba7897af7` (admin perf A) moved the billing router under `/api/v1/admin` and did not update the test.
- `Character_Chat_NEW/integration/test_chat_import_local_chats.py::test_another_owner_reusing_ids_never_touches_the_first_owners_chat[postgres]` (`assert 500 == 201`; it runs only when a Postgres is reachable).

Three others depend on which tests share an xdist worker, and may or may not fail in a given run:
- `Admin/test_admin_llm_provider_test_runtime.py::test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers`: failed in the base run, passed with the change.
- The five `Services/test_router_groups_contract.py::test_iter_minimal_optional_router_specs_*audio*` tests (and their five `*media*` siblings): they fail whenever a Claims test ran earlier on the same worker. `tests/Claims/conftest.py`'s session-scoped autouse fixture `_claims_suite_minimal_app_env` sets `ROUTES_DISABLE` (media, audio, ...) and `MINIMAL_TEST_APP=1` in `os.environ` until the session ends, so `route_enabled("audio")` is False and the routes are dropped before any import. Reproduced on unchanged `origin/dev`: `pytest -p no:randomly tests/Claims/test_claims_service_fva_analyze.py tests/Services/test_router_groups_contract.py` gives 11 failed. With the change they failed in two of three sweeps, and passed in the base sweep, as xdist placed the Claims test.
- `Character_Chat_NEW/.../test_character_chat_images.py::test_character_image_reads_leave_event_loop_and_release_operation[postgres-*]` (outside this list): flaky on base too, per the plan review.

None of these is caused by this change or fixed here. If a failure outside these lists appears, stop and compare with `origin/dev` before going on.

- [ ] **Step 8: Lint and security**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/core/Utils/image_validation.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/api/v1/schemas/chat_validators.py tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py
uvx bandit -ll tldw_Server_API/app/core/Utils/image_validation.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/api/v1/schemas/chat_validators.py
```

Expected: ruff reports only `chat_service.py:10:1: I001`, which `origin/dev` already has (one finding before and after; it is the existing `Character_Chat.constants` import order, not the new line). Bandit: `No issues identified.`

- [ ] **Step 9: Commit the fix**

```bash
git add tldw_Server_API/tests/Chat/unit/test_data_uri_redaction_is_linear.py tldw_Server_API/app/core/Utils/image_validation.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/api/v1/schemas/chat_validators.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py
git commit -m "fix(chat): data-URI redaction is linear on crafted input; one bounded pattern for every chat-path redactor (TASK-13560)

The chat service, the chat endpoint and the request size validator each used
(data:image[^,]*,), which rescans to the end of the text for every data:image
with no comma after it: a 510 KB body of such markers took about 5.5 s per call.
They now share DATA_URI_REDACT_RE (core/Utils/image_validation.py), the bounded
pattern LLM usage accounting already used: about 5 ms on the same body.

Sharing it also ends two divergences: the endpoint's estimate redacted only a
lowercase payload under a \"url\" key, and the size validator only lowercase
data:image. The bound has one cost: a header longer than 128 characters or
containing whitespace or a quote is now counted in full, so at the size
validator a large image with such a malformed header is rejected as too large.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 10: Close the task and commit the plan**

The notes below quote the measured results. If Steps 5-8 printed different lines, put the actual ones in the notes.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13560 --append-notes "Fixed: DATA_URI_REDACT_RE in core/Utils/image_validation.py, the bounded pattern from usage_accounting (whose private copy is removed), is used by chat_service._sanitize_data_uris, the endpoint's _sanitize_json_for_rate_limit and validate_request_size. A 510 KB body of '\"url\":\"data:image' markers: about 5.5 s per site before, about 5 ms after. Behaviour changes: the endpoint's queue estimate now also redacts a payload outside a \"url\" key and in any case, and the size validator an uppercase DATA:IMAGE URI; a header longer than 128 characters or containing whitespace or a quote is left whole (counted in full, so at the size validator a large image with such a header is now too large). Tests: the new file 9 passed (before the fix 6 failed, the 3 guards passed); the redaction tests 131 passed; every test file importing a changed module (189 files plus the new one) 2 failed, 5594 passed, 35 skipped, both failures also failing on unchanged origin/dev (test_iter_admin_router_specs_populates_expected_specs; the Postgres owner-reuse chat import). Bandit -ll on the five source files: no issues. Ruff: only the I001 at chat_service.py:10 that origin/dev already has. Left to chat-7 (Docs/superpowers/reviews/chat/2026-09-21-stage2-orchestration-streaming.md): prompt_cost_envelope's own scanner (linear; feeds prompt fingerprints), the two rounding rules, and the redundant outer _sanitize_json_for_rate_limit pass before estimate_tokens_from_json." --check-ac 1 --check-ac 2 --check-ac 3 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "Data-URI redaction on the chat path is linear: the chat service, the chat endpoint and the request size validator share one bounded pattern (DATA_URI_REDACT_RE in core/Utils/image_validation.py) with LLM usage accounting. A crafted 510 KB message takes about 5 ms instead of about 5.5 s per call site; a test holds each site under two seconds." --status Done
git add -A backlog/tasks
git add Docs/superpowers/plans/2026-10-15-data-uri-redaction-linear.md
git commit -m "docs(chat): plan for linear data-URI redaction; close TASK-13560

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: `git show --stat HEAD` lists the task file and the plan only.
