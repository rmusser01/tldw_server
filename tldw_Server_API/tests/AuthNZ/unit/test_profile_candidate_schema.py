from __future__ import annotations

from copy import deepcopy

from tldw_Server_API.app.core.AuthNZ.profile_candidate_schema import (
    _COLUMN_KINDS,
    _EXPECTED_DEFAULTS,
    _REQUIRED_NOT_NULL,
    PROFILE_CANDIDATE_COLUMNS,
    PROFILE_CANDIDATE_FOREIGN_KEYS,
    PROFILE_CANDIDATE_PRIMARY_KEYS,
    PROFILE_CANDIDATE_TABLES,
    PROFILE_CANDIDATE_UNIQUE_KEYS,
    _default_matches,
    profile_candidate_schema_is_valid,
)

_POSTGRES_TYPES = {
    "integer": "bigint",
    "text": "text",
    "identifier": "text",
    "boolean": "boolean",
    "json": "jsonb",
    "timestamp": "timestamp with time zone",
}


def _postgres_contract():
    columns = {}
    for table_name in PROFILE_CANDIDATE_TABLES:
        columns[table_name] = {}
        for column_name in PROFILE_CANDIDATE_COLUMNS[table_name]:
            kind = _COLUMN_KINDS[table_name][column_name]
            expected_default = _EXPECTED_DEFAULTS[table_name].get(column_name)
            default = None
            if expected_default == "current_timestamp":
                default = "CURRENT_TIMESTAMP"
            elif expected_default == "true":
                default = "true"
            elif expected_default is not None:
                default = f"'{expected_default}'::text"
            columns[table_name][column_name] = {
                "data_type": _POSTGRES_TYPES[kind],
                "not_null": column_name in _REQUIRED_NOT_NULL[table_name],
                "default": default,
                "is_identity": False,
                "identity_generation": None,
            }
    columns["organizations"]["id"]["default"] = (
        "nextval('public.organizations_id_seq'::regclass)"
    )
    columns["teams"]["id"]["default"] = (
        "nextval('public.teams_id_seq'::regclass)"
    )
    foreign_keys = {
        table_name: {
            (source, "public", target, target_column, action)
            for source, target, target_column, action in PROFILE_CANDIDATE_FOREIGN_KEYS[table_name]
        }
        for table_name in PROFILE_CANDIDATE_TABLES
    }
    return {
        "backend": "postgres",
        "columns": columns,
        "primary_keys": deepcopy(PROFILE_CANDIDATE_PRIMARY_KEYS),
        "unique_keys": deepcopy(PROFILE_CANDIDATE_UNIQUE_KEYS),
        "foreign_keys": foreign_keys,
    }


def test_postgres_candidate_contract_requires_generated_hierarchy_ids() -> None:
    metadata = _postgres_contract()
    assert profile_candidate_schema_is_valid(**metadata)

    metadata["columns"]["organizations"]["id"]["default"] = None

    assert not profile_candidate_schema_is_valid(**metadata)


def test_postgres_candidate_contract_requires_public_foreign_key_targets() -> None:
    metadata = _postgres_contract()
    metadata["foreign_keys"]["org_members"] = {
        (
            source,
            "shadow" if source == "user_id" else schema,
            target,
            target_column,
            action,
        )
        for source, schema, target, target_column, action in metadata["foreign_keys"][
            "org_members"
        ]
    }

    assert not profile_candidate_schema_is_valid(**metadata)


def test_postgres_candidate_contract_accepts_native_uuid_organization_identifier() -> None:
    metadata = _postgres_contract()
    metadata["columns"]["organizations"]["uuid"]["data_type"] = "uuid"

    assert profile_candidate_schema_is_valid(**metadata)


def test_candidate_contract_requires_non_null_override_update_timestamps() -> None:
    metadata = _postgres_contract()
    metadata["columns"]["user_config_overrides"]["updated_at"]["not_null"] = False

    assert not profile_candidate_schema_is_valid(**metadata)


def test_default_matching_rejects_expressions_after_casts() -> None:
    assert not _default_matches("'member'::text || '_admin'::text", "member")
    assert not _default_matches("true::boolean AND false", "true")
