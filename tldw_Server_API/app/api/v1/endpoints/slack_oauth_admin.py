"""Slack OAuth install/admin endpoints.

The flow itself lives in ``_chatops.oauth_admin``, shared with Discord: these two
modules were 90.9% identical after normalising ``discord``/``slack`` and
``guild``/``team``. What is Slack-specific is the descriptor below; the functions
are thin wrappers that keep the existing keyword names so the router is untouched.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User

from ._chatops import oauth_admin as _shared
from ._chatops.oauth_admin import ChatOpsOAuthProvider

SLACK = ChatOpsOAuthProvider(
    name="slack",
    label="Slack",
    entity_id_key="team_id",
    entity_name_key="team_name",
    response_entity_field="team",
    # Slack's policy metric says "workspace" where Discord's says "guild". Both are
    # existing metric label values, so they are kept rather than unified.
    policy_scope_label="workspace",
    client_id_setting="SLACK_CLIENT_ID",
    # Slack derives the workspace solely from the token response, so a missing id is
    # an upstream fault (502), not a bad request.
    missing_entity_status=502,
    missing_entity_detail="Slack OAuth response missing team.id",
    installation_extra_keys=("enterprise_id", "bot_user_id", "scope"),
    installation_nested_keys={"authed_user_id": ("authed_user", "id")},
    # Slack's token response carries an explicit ok flag.
    requires_ok_flag=True,
    # Slack's token endpoint does not take grant_type, and its authorize URL takes
    # no response_type.
    sends_grant_type=False,
    sends_response_type=False,
    blank_key_hint_as_empty=True,
)


async def slack_oauth_start_impl(
    *,
    user: User,
    workspace_org_id: int | None,
    oauth_client_id: Callable[[], str | None],
    oauth_redirect_uri: Callable[[], str],
    oauth_state_ttl_seconds: Callable[[], int],
    get_oauth_state_repo: Callable[[], Awaitable[Any]],
    encrypt_slack_payload: Callable[[dict[str, Any]], str],
    oauth_auth_url: Callable[[], str],
    oauth_scopes: Callable[[], str],
    urlencode_fn: Callable[[dict[str, str]], str],
) -> dict[str, Any]:
    return await _shared.oauth_start(
        SLACK,
        user=user,
        workspace_org_id=workspace_org_id,
        oauth_client_id=oauth_client_id,
        oauth_redirect_uri=oauth_redirect_uri,
        oauth_state_ttl_seconds=oauth_state_ttl_seconds,
        get_oauth_state_repo=get_oauth_state_repo,
        encrypt_payload=encrypt_slack_payload,
        oauth_auth_url=oauth_auth_url,
        oauth_scope=oauth_scopes,
        urlencode_fn=urlencode_fn,
    )


async def slack_oauth_callback_impl(
    *,
    code: str,
    state: str,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_oauth_state_repo: Callable[[], Awaitable[Any]],
    oauth_client_id: Callable[[], str | None],
    oauth_client_secret: Callable[[], str | None],
    oauth_token_url: Callable[[], str],
    slack_oauth_token_exchange: Callable[..., Awaitable[dict[str, Any]]],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_slack_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_slack_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    return await _shared.oauth_callback(
        SLACK,
        code=code,
        state=state,
        coerce_nonempty_string=coerce_nonempty_string,
        get_oauth_state_repo=get_oauth_state_repo,
        oauth_client_id=oauth_client_id,
        oauth_client_secret=oauth_client_secret,
        oauth_token_url=oauth_token_url,
        oauth_token_exchange=slack_oauth_token_exchange,
        get_user_secret_repo=get_user_secret_repo,
        get_workspace_provider_installations_repo=get_workspace_provider_installations_repo,
        resolve_workspace_org_id=resolve_workspace_org_id,
        decrypt_payload=decrypt_slack_payload,
        normalize_installations_payload=normalize_installations_payload,
        encrypt_payload=encrypt_slack_payload,
    )


def slack_admin_get_policy_impl(
    *,
    team_id: str | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    slack_policy_for_workspace: Callable[[str | None], dict[str, Any]],
) -> dict[str, Any]:
    return _shared.admin_get_policy(
        SLACK,
        entity_id=team_id,
        coerce_nonempty_string=coerce_nonempty_string,
        policy_for_entity=slack_policy_for_workspace,
    )


def slack_admin_set_policy_impl(
    *,
    payload: dict[str, Any] | None,
    coerce_nonempty_string: Callable[[Any], str | None],
    set_slack_policy: Callable[[str | None, dict[str, Any]], tuple[str | None, dict[str, Any]]],
    emit_slack_counter: Callable[..., None],
) -> dict[str, Any]:
    return _shared.admin_set_policy(
        SLACK,
        payload=payload,
        coerce_nonempty_string=coerce_nonempty_string,
        set_policy=set_slack_policy,
        emit_counter=emit_slack_counter,
        counter_name="slack_policy_updates_total",
    )


async def slack_admin_list_installations_impl(
    *,
    user: User,
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    decrypt_slack_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    public_installation_record: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    return await _shared.admin_list_installations(
        SLACK,
        user=user,
        get_user_secret_repo=get_user_secret_repo,
        decrypt_payload=decrypt_slack_payload,
        normalize_installations_payload=normalize_installations_payload,
        public_installation_record=public_installation_record,
    )


async def slack_admin_delete_installation_impl(
    *,
    team_id: str,
    user: User,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_slack_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_slack_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    return await _shared.admin_delete_installation(
        SLACK,
        entity_id=team_id,
        user=user,
        coerce_nonempty_string=coerce_nonempty_string,
        get_user_secret_repo=get_user_secret_repo,
        get_workspace_provider_installations_repo=get_workspace_provider_installations_repo,
        resolve_workspace_org_id=resolve_workspace_org_id,
        decrypt_payload=decrypt_slack_payload,
        normalize_installations_payload=normalize_installations_payload,
        encrypt_payload=encrypt_slack_payload,
    )


async def slack_admin_set_installation_state_impl(
    *,
    team_id: str,
    payload: dict[str, Any] | None,
    user: User,
    coerce_nonempty_string: Callable[[Any], str | None],
    get_user_secret_repo: Callable[[], Awaitable[Any]],
    get_workspace_provider_installations_repo: Callable[[], Awaitable[Any]],
    resolve_workspace_org_id: Callable[[int], Awaitable[int]],
    decrypt_slack_payload: Callable[[Any], Any],
    normalize_installations_payload: Callable[[Any], dict[str, Any]],
    encrypt_slack_payload: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    return await _shared.admin_set_installation_state(
        SLACK,
        entity_id=team_id,
        payload=payload,
        user=user,
        coerce_nonempty_string=coerce_nonempty_string,
        get_user_secret_repo=get_user_secret_repo,
        get_workspace_provider_installations_repo=get_workspace_provider_installations_repo,
        resolve_workspace_org_id=resolve_workspace_org_id,
        decrypt_payload=decrypt_slack_payload,
        normalize_installations_payload=normalize_installations_payload,
        encrypt_payload=encrypt_slack_payload,
    )
