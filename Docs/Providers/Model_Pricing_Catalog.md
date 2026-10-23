# Model Pricing Catalog (Primary Model Source)

The pricing catalog at `tldw_Server_API/Config_Files/model_pricing.json` is the primary reference for
listing available commercial LLM models across the API and WebUI. Entries here both:

- Define per‑1K token pricing for usage tracking (prompt/completion in USD), and
- Seed the available models list returned by `GET /api/v1/llm/providers` (merged with any models in `config.txt`).

When you add a model to this file (or to the `PRICING_OVERRIDES` env var), it becomes selectable in the WebUI
and available to the Chat API as `provider/model`.

## How It Works

- Source order:
  1. `PRICING_OVERRIDES` (JSON in env)
  2. `Config_Files/model_pricing.json`
  3. Built‑in conservative defaults
- Admin reload (no restart): `POST /api/v1/admin/llm-usage/pricing/reload`
- Providers API: `GET /api/v1/llm/providers` includes models from the pricing catalog for commercial providers.
- Embedding model IDs are intentionally filtered out from the Chat model lists.

## Editing the Catalog

- Format: JSON object by provider, then model id → `{prompt: number, completion: number}` (USD per 1K tokens), with optional `cache_read`, `cache_write` and `cache_write_1h`: the prompt-cache read, 5-minute write and 1-hour write rates, also per 1K tokens (a missing `cache_write_1h` falls back to `cache_write`). Without them, cached tokens cost the `prompt` rate. An optional `long_prompt` object (`above_tokens` plus a full rate set) replaces the entry's rates for a request whose whole prompt is longer than `above_tokens`; it applies only where the caller knows the prompt length (usage costing), not in the no-length estimators `get_rates` and `compute_token_cost`, which use the base rates. An optional `"hidden": true` prices a model without listing it in selectors. Anthropic dotted names such as `claude-sonnet-4.5` need no entry: lookups, the model-availability check and the outgoing request map them to the dashed API ID (`claude-sonnet-4-5`). An override replaces the whole entry, so an override of an entry that has cache rates, a `long_prompt` or `hidden` must repeat them: an override of `claude-haiku-5-5` without `long_prompt` silently loses the tier.
- Example (OpenAI text models only):

```
{
  "openai": {
    "gpt-4o":        { "prompt": 0.005,  "completion": 0.015 },
    "gpt-4o-mini":   { "prompt": 0.001,  "completion": 0.002 },
    "gpt-4.1":       { "prompt": 0.010,  "completion": 0.030 },
    "o3-mini":       { "prompt": 0.001,  "completion": 0.002 }
  }
}
```

Tip: Keep values conservative if you’re unsure, then update with exact rates from provider pricing pages.

## Provider Quick Links

- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing (model IDs: https://platform.claude.com/docs/en/about-claude/models/overview)
- OpenAI (text models): https://developers.openai.com/api/docs/pricing
- Z.ai: https://docs.z.ai/guides/overview/pricing
- Moonshot (Kimi): https://platform.moonshot.ai/docs/pricing/chat#generation-model-kimi-k2
- Cohere: https://docs.cohere.com/docs/models
- Minimax: https://platform.minimax.io/docs/guides/pricing

## Example Snippets by Provider

These examples illustrate the expected shape. Replace with current values from the linked pages above.

Anthropic (keys are API model IDs; an override repeats the cache rates, the 1-hour rate and any `long_prompt`):
```
{
  "anthropic": {
    "claude-opus-4-7":   { "prompt": 0.005, "completion": 0.025, "cache_read": 0.0005, "cache_write": 0.00625, "cache_write_1h": 0.01 },
    "claude-sonnet-4-6": { "prompt": 0.003, "completion": 0.015, "cache_read": 0.0003, "cache_write": 0.00375, "cache_write_1h": 0.006 },
    "claude-haiku-5-5":  { "prompt": 0.0001, "completion": 0.0005, "cache_read": 0.00001, "cache_write": 0.000125, "cache_write_1h": 0.0002,
                           "long_prompt": { "above_tokens": 100000, "prompt": 0.0005, "completion": 0.0025, "cache_read": 0.00005, "cache_write": 0.000625, "cache_write_1h": 0.001 } }
  }
}
```

OpenAI (text models only – do not include embeddings here):
```
{
  "openai": {
    "gpt-4o":      { "prompt": 0.005, "completion": 0.015 },
    "gpt-4o-mini": { "prompt": 0.001, "completion": 0.002 },
    "gpt-4.1":     { "prompt": 0.010, "completion": 0.030 },
    "o3-mini":     { "prompt": 0.001, "completion": 0.002 }
  }
}
```

Z.ai:
```
{
  "zai": {
    "<model-id>": { "prompt": 0.000, "completion": 0.000 }
  }
}
```

Moonshot (Kimi):
```
{
  "moonshot": {
    "kimi-k2": { "prompt": 0.000, "completion": 0.000 }
  }
}
```

Cohere (Command family):
```
{
  "cohere": {
    "command":   { "prompt": 0.0005, "completion": 0.0012 },
    "command-r": { "prompt": 0.0015, "completion": 0.0030 }
  }
}
```

Minimax:
```
{
  "minimax": {
    "<model-id>": { "prompt": 0.000, "completion": 0.000 }
  }
}
```

## Validation & Troubleshooting

- After editing, call: `POST /api/v1/admin/llm-usage/pricing/reload`.
- Verify in WebUI → Providers tab, or via `GET /api/v1/llm/providers`.
- If a model appears only in `config.txt`, it’s listed but costs may be “estimated.” Add it here for exact rates.
