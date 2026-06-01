# MCP External Runtime Installer Status Design

## Goal

Harden the standalone MCP gateway's external server install/update surface so frontends and operators can distinguish unsupported, unavailable, failed, disabled, and successful installer states without exposing secrets or adding real package-manager execution in this slice.

## Scope

This slice keeps the existing `ExternalServerInstaller` adapter seam and the default `NullExternalServerInstaller`. It adds a public, sanitized installer status contract to runtime status rows and normalizes install/update operation payloads. Real npm, pip, uvx, or other third-party installation execution remains out of scope.

## Architecture

`GatewayExternalRuntimeManager` remains the orchestration boundary. It loads the external server definition, enforces enabled/not-found checks, delegates optional work to the configured installer, and converts installer adapter output into a stable public payload. The installer adapter may return richer metadata, but the runtime manager owns public response normalization and redaction.

Runtime status rows gain an `installer` object from `ExternalServerInstaller.get_status(server)`. The manager sanitizes that object before returning it. If the installer status call fails, the manager logs diagnostics with traceback and returns a deterministic unavailable installer status for that server instead of breaking the whole status list.

Install/update operations use the same sanitization rules. Expected unavailable or unsupported results pass through with normalized defaults. Unexpected adapter exceptions are logged and surfaced as `GatewayExternalRuntimeError` with `external_server_install_failed` or `external_server_update_failed`, without raw exception text in HTTP responses.

## Public Payload Rules

Allowed public fields are lightweight scalars and scalar lists/maps such as `ok`, `available`, `reason_code`, `server_id`, `installer`, `version`, `installed_version`, `latest_version`, `message`, `details`, `required_fields`, and `warnings`. Sensitive values are removed recursively from public payloads when keys indicate secrets, tokens, passwords, credentials, headers, env, authorization, or command arguments.

Operation responses always include `ok`, `available`, `reason_code`, and `server_id`. Status responses always include `available`, `reason_code`, and `server_id`.

## Error Handling

Disabled and not-found servers continue to use the existing `GatewayExternalRuntimeError` paths. Installer adapter failures are treated as runtime operation failures, logged with traceback for diagnostics, and exposed with stable reason codes. Status collection is best-effort per server so one failing installer status check does not prevent reporting other runtime rows.

## Tests

Focused pytest coverage will verify default unsupported status, successful fake installer status and install/update responses, disabled/not-found operation behavior, adapter exception handling, and redaction of nested secret-looking fields from status and operation responses.
