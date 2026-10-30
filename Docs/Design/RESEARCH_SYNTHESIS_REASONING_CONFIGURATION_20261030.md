# Deep Research synthesis reasoning configuration

TASK-13260.278.18.83.45. S04 remains open.

A real direct diagnostic using the actual provider prompt and approved owned inputs exhausted 2,500 tokens on reasoning and returned no final content. The parser then saw the outer completion envelope and reported a missing outline. This does not recover the discarded native response. Explicit temperature 0 is also replaced with 0.2.

Add one optional operator setting, Search-Agent.deep_research_llamacpp_enable_thinking. Blank preserves provider defaults; standard ConfigParser booleans are strict. Per-run overrides cannot change it. Persist the resolved synthesis policy and use the existing llama.cpp extra_body.chat_template_kwargs.enable_thinking control only for that provider. Other providers and trusted endpoint/credential policy stay unchanged. Preserve explicit temperature 0. A completion ending with length fails explicitly as synthesis_output_truncated before parsing; keep existing fallback and strict content/citation validation. Do not increase token or time budgets.

Causal tests precede implementation: configured boolean policy, zero/default temperature, exact llama.cpp extension and other-provider behavior, truncated empty/partial completions and normal structured responses. Then affected resolver checks, security, review, package and live model-backed synthesis are required.

ADR assessment: no new durable integration rule. ADR-025 provider adapter routing and ADR-030 configured local origins remain governing; this uses their existing request extension path and introduces no service, dependency or endpoint.

The subsequent normal-worker run retained the false thinking policy and temperature zero, then reached a final structured response with qualitative confidence "high". Its prompt had named the confidence field without specifying its numeric type. State the numeric zero-to-one probability contract explicitly; keep qualitative values rejected rather than invent a conversion. Causal outgoing-contract and negative parsing controls precede this narrow correction; actual model validation remains required.

The next normal worker reached another fallback: the recorded dictionary-update error is consistent with the parser receiving a non-object summary, although the discarded model reply cannot be recovered. The request named summary without its object type and left nested arrays/citations unspecified. State those existing parser shapes explicitly, including source_id citation objects, and reject non-object summary values with a stable field error. Keep missing/null summary compatibility, all evidence checks and the existing fallback. New causal request/parser tests plus affected valid synthesis controls precede native validation; no token or runtime increase.
