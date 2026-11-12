"""Discord OAuth install/admin endpoints.

The flow itself lives in ``_chatops.oauth_admin``, shared with Slack: these two
modules were 90.9% identical after normalising ``discord``/``slack`` and
``guild``/``team``, and the divergence had already started costing real work --
the IDOR fix on ``GET /{discord|slack}/jobs/{job_id}`` had to be written four
times. What is Discord-specific is the descriptor below; the functions are thin
wrappers that keep the existing keyword names so the router is untouched.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User

from ._chatops import oauth_admin as _shared
from ._chatops.oauth_admin import ChatOpsOAuthProvider

DISCORD = ChatOpsOAuthProvider(
    name="discord",
    label="Discord",
    entity_id_key="guild_id",
    entity_name_key="guild_name",
    response_entity_field="guild",
    policy_scope_label="guild",
    client_id_setting="DISCORD_CLIENT_ID",
    # Discord answers 400 here because the caller can supply guild_id on the
    # callback query; Slack derives it solely from the token response and answers
    # 502. Both are public contracts, so they are kept rather than unified.
    missing_entity_status=400,
    missing_entity_detail="Discord OAuth callback is missing guild_id",
    installation_extra_keys=("refresh_token", "scope"),
)


async def discord_oauth_start_impl(
    *,
    user: User,
    workspace_org_id: int | None,
    oauth_client_id: Callable[[], str | None],
    oauth_redirect_uri: Callable[[], str],
    oauth_state_ttl_seconds: Callable[[], int],
    get_oauth_state_repo: Callable[[], Awaitable[Any]],
    encrypt_discord_payload: Callable[[dict[str, Any]], str],
    oauth_auth_url: Callable[[], str],
    oauth_scope: Callable[[], str],
    oauth_permissions: Callable[[], str | None],
    urlencode_fn: Callable[[dict[str, str]], str],
) -> dict[str, Any]:
    return await _shared.oauth_start(
        DISCORD,
        user=user,
        workspace_org_id=workspace_org_id,
        oauth_client_id=oauth_client_id,
        oauth_redirect_uri=oauth_redirect_uri,
        oauth_state_ttl_seconds=oauth_state_ttl_seconds,
        get_oauth_state_repo=get_oauth_state_repo,
        encrypt_payload=encrypt_discord_payload,
        oauth_auth_url=oauth_auth_url,
        oauth_scope=oauth_scope,
        oauth_permissions=oauth_permissions,
        urlencode_fn=urlencode_fn,
    )


async def discord_oauth_callback_impl(
    *,
    code: str,
    state: str,
    guild_id: str | None,
    guild_name: str | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_oauth_state_repo: Callable[[], Awaitable[Any]],
    oauth_client_id: Callable[[], str | None],
    oauth_client_secret: Callable[[], str | None],
    oauth_token_url: Callable[[], str],
    discord_oauth_token_exchange: Callable[..., Awaitable[dict[str, Any]]],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_discord_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_discord_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    return await _shared.oauth_callback(
        DISCORD,
        code=code,
        state=state,
        # Discord alone accepts these from the callback query as a fallback when
        # the token response carries no guild object.
        fallback_entity_id=guild_id,
        fallback_entity_name=guild_name,
        coerce_nonempty_string=coerce_nonempty_string,
        get_oauth_state_repo=get_oauth_state_repo,
        oauth_client_id=oauth_client_id,
        oauth_client_secret=oauth_client_secret,
        oauth_token_url=oauth_token_url,
        oauth_token_exchange=discord_oauth_token_exchange,
        get_user_secret_repo=get_user_secret_repo,
        get_workspace_provider_installations_repo=get_workspace_provider_installations_repo,
        resolve_workspace_org_id=resolve_workspace_org_id,
        decrypt_payload=decrypt_discord_payload,
        normalize_installations_payload=normalize_installations_payload,
        encrypt_payload=encrypt_discord_payload,
    )


def discord_admin_get_policy_impl(
    *,
    guild_id: str | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    discord_policy_for_guild: Callable[[str | None], dict[str, Any]],
) -> dict[str, Any]:
    return _shared.admin_get_policy(
        DISCORD,
        entity_id=guild_id,
        coerce_nonempty_string=coerce_nonempty_string,
        policy_for_entity=discord_policy_for_guild,
    )


def discord_admin_set_policy_impl(
    *,
    payload: dict[str, Any] | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    set_discord_policy: Callable[[str | None, dict[str, Any]], tuple[str | None, dict[str, Any]]],
    emit_discord_counter: Callable[..., None],
) -> dict[str, Any]:
    return _shared.admin_set_policy(
        DISCORD,
        payload=payload,
        coerce_nonempty_string=coerce_nonempty_string,
        set_policy=set_discord_policy,
        emit_counter=emit_discord_counter,
        counter_name="discord_policy_updates_total",
    )


async def discord_admin_list_installations_impl(
    *,
    user: User,
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    decrypt_discord_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    public_installation_record: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    return await _shared.admin_list_installations(
        DISCORD,
        user=user,
        get_user_secret_repo=get_user_secret_repo,
        decrypt_payload=decrypt_discord_payload,
        normalize_installations_payload=normalize_installations_payload,
        public_installation_record=public_installation_record,
    )


async def discord_admin_delete_installation_impl(
    *,
    guild_id: str,
    user: User,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_discord_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_discord_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    return await _shared.admin_delete_installation(
        DISCORD,
        entity_id=guild_id,
        user=user,
        coerce_nonempty_string=coerce_nonempty_string,
        get_user_secret_repo=get_user_secret_repo,
        get_workspace_provider_installations_repo=get_workspace_provider_installations_repo,
        resolve_workspace_org_id=resolve_workspace_org_id,
        decrypt_payload=decrypt_discord_payload,
        normalize_installations_payload=normalize_installations_payload,
        encrypt_payload=encrypt_discord_payload,
    )


async def discord_admin_set_installation_state_impl(
    *,
    guild_id: str,
    payload: dict[str, Any] | None,
    user: User,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_discord_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_discord_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    return await _shared.admin_set_installation_state(
        DISCORD,
        entity_id=guild_id,
        payload=payload,
        user=user,
        coerce_nonempty_string=coerce_nonempty_string,
        get_user_secret_repo=get_user_secret_repo,
        get_workspace_provider_installations_repo=get_workspace_provider_installations_repo,
        resolve_workspace_org_id=resolve_workspace_org_id,
        decrypt_payload=decrypt_discord_payload,
        normalize_installations_payload=normalize_installations_payload,
        encrypt_payload=encrypt_discord_payload,
    )
