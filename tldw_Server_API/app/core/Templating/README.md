# Templating

Templating provides the sandboxed Jinja rendering helper used by chat dictionaries, chatbooks, and prompt-adjacent flows. It builds a constrained environment from configuration, validates expression-only rendering where requested, exposes safe date/random helpers, and enforces timeout and output-size limits.

## Start Here

- `template_renderer.py` is the only implementation file and contains the renderer, environment options, context objects, and error types.
- There is no dedicated REST endpoint for this package; it is consumed by chat and chatbook code.
- Related tests: `tests/Chat_NEW/unit/test_template_renderer.py`.

## Responsibilities

- Build a sandboxed Jinja environment with controlled globals and filters.
- Render templates with `TemplateContext`, `TemplateOptions`, and environment-derived defaults.
- Enforce render timeout, output-size, cache, and expression-only constraints.
- Provide deterministic and bounded date/random helper behavior.
- Raise `TemplateRenderError` for rendering and validation failures.

## Module Map

- `template_renderer.py` - sandbox environment creation, option loading, context handling, validation, and rendering.

## How It Connects

- `app/core/Chat/chat_dictionary.py` uses templating for dictionary-based chat substitutions.
- `app/core/Chatbooks/chatbook_service.py` uses templating for chatbook tool/content rendering.
- `app/api/v1/endpoints/chat.py` references prompt templating behavior while preparing chat input.
- Configuration comes from the `[Chat-Templating]` section and environment keys handled by `options_from_env`.

## Extension Points

- For a new safe helper or filter, add it in `template_renderer.py` and cover sandbox behavior in `tests/Chat_NEW/unit/test_template_renderer.py`.
- For new configuration options, extend `TemplateOptions`, `options_from_env`, and tests for default/override behavior.
- For a new consumer, pass an explicit `TemplateContext` instead of exposing raw application objects to templates.

## Testing

- `tests/Chat_NEW/unit/test_template_renderer.py`

## Gotchas

- The renderer is intentionally sandboxed; avoid adding helpers that can reach the filesystem, network, process environment, or arbitrary Python objects.
- Expression-only mode is a validation path, not just a style preference, and callers may depend on it to reject block templates.
- Output limits and timeout settings are part of the safety boundary for user-provided templates.
