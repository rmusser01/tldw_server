"""PostgreSQL 13-compatible Claims monitoring JSON helper DDL."""

from __future__ import annotations

POSTGRES_CLAIMS_JSON_HELPER_DDL = (
    """
    CREATE OR REPLACE FUNCTION tldw_claims_safe_json(value TEXT)
    RETURNS JSON
    LANGUAGE plpgsql
    IMMUTABLE
    PARALLEL SAFE
    AS $function$
    BEGIN
        IF value IS NULL OR btrim(value) = '' THEN
            RETURN '{}'::JSON;
        END IF;
        RETURN value::JSON;
    EXCEPTION WHEN invalid_text_representation THEN
        RETURN '{}'::JSON;
    END;
    $function$
    """,
    """
    CREATE OR REPLACE FUNCTION tldw_claims_compact_json(value JSON)
    RETURNS TEXT
    LANGUAGE plpgsql
    IMMUTABLE
    PARALLEL SAFE
    AS $function$
    DECLARE
        kind TEXT;
    BEGIN
        IF value IS NULL THEN
            RETURN '{}';
        END IF;
        kind := json_typeof(value);
        IF kind = 'object' THEN
            RETURN COALESCE(
                (
                    SELECT '{' || string_agg(
                        to_json(entry.key)::TEXT || ':' || tldw_claims_compact_json(entry.item),
                        ',' ORDER BY entry.ordinality
                    ) || '}'
                    FROM json_each(value) WITH ORDINALITY AS entry(key, item, ordinality)
                ),
                '{}'
            );
        ELSIF kind = 'array' THEN
            RETURN COALESCE(
                (
                    SELECT '[' || string_agg(
                        tldw_claims_compact_json(entry.item),
                        ',' ORDER BY entry.ordinality
                    ) || ']'
                    FROM json_array_elements(value) WITH ORDINALITY AS entry(item, ordinality)
                ),
                '[]'
            );
        END IF;
        RETURN btrim(value::TEXT);
    END;
    $function$
    """,
)

__all__ = ["POSTGRES_CLAIMS_JSON_HELPER_DDL"]
