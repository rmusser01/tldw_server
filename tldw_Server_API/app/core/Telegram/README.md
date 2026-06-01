# Telegram

Telegram is a small core helper package for deterministic Telegram chat-to-session mapping. Most Telegram webhook, linking, approval, policy, delivery, and admin behavior lives in endpoint, service, and AuthNZ repository modules; this package provides the reusable conversation key derivation used by those flows.

## Start Here

- `session_mapper.py` contains the session-key and conversation-id helpers.
- Related API surface: `app/api/v1/endpoints/telegram.py`, `app/api/v1/endpoints/telegram_support.py`, and `app/api/v1/endpoints/integrations_control_plane.py`.
- Related schemas: `app/api/v1/schemas/telegram_schemas.py`.
- Related tests: `tests/Telegram/`.

## Responsibilities

- Coerce Telegram chat/user identifiers into non-empty string values.
- Build deterministic Telegram session keys from bot/user/chat context.
- Derive assistant, persona, and character conversation identifiers from Telegram sessions.
- Keep session identity logic shared between webhook, command, approval, and delivery flows.

## Module Map

- `session_mapper.py` - identifier coercion plus session and conversation key builders.

## How It Connects

- `app/api/v1/endpoints/telegram.py` handles webhook and command entry points that use these session keys.
- `app/api/v1/endpoints/telegram_support.py` and `app/api/v1/endpoints/integrations_control_plane.py` expose support/admin operations around Telegram integrations.
- `app/core/AuthNZ/repos/telegram_runtime_repo.py` and `telegram_approvals_repo.py` store runtime and approval state.
- `app/services/telegram_delivery_service.py` handles outbound delivery.

## Extension Points

- For new conversation modes, add deterministic key helpers in `session_mapper.py` and extend `tests/Telegram/test_telegram_session_mapper.py`.
- For webhook or command behavior, start in `app/api/v1/endpoints/telegram.py`; this core package only owns identity mapping.
- For approval or delivery changes, inspect the AuthNZ Telegram repos and `app/services/telegram_delivery_service.py`.

## Testing

- `tests/Telegram/test_telegram_session_mapper.py`
- `tests/Telegram/test_telegram_webhook.py`
- `tests/Telegram/test_telegram_commands.py`
- `tests/Telegram/test_telegram_linking_and_policy.py`
- `tests/Telegram/test_telegram_approvals.py`
- `tests/Telegram/test_telegram_jobs_and_delivery.py`
- `tests/Telegram/test_telegram_admin_api.py`
- `tests/Telegram/test_telegram_admin_link_inventory.py`
- `tests/Telegram/test_telegram_schemas.py`
- AuthNZ-related Telegram tests also live under `tests/AuthNZ/unit/`.

## Gotchas

- Keep session keys stable. Changing the key format can orphan existing chat sessions and approval records.
- The core package does not own Telegram runtime policy, webhook authentication, or outbound delivery; those are implemented in adjacent endpoint/service/repository modules.
