# Deep Research local full-text configuration

Task: TASK-13260.278.18.83.45

Local collection currently forces vector retrieval and constructs the Media vector adapter before query configuration is applied. An FTS deployment therefore cannot avoid embedding/model work through its normal Deep Research configuration.

Add `Search-Agent.deep_research_local_use_vector`, default `true`, parsed with Python's standard ConfigParser boolean values. Invalid values fail descriptively. The operator setting is resolved before per-run overrides; an unrecognized per-run `local.use_vector` key cannot override it.

LocalResearchProvider creates one RetrievalConfig with FTS enabled and the resolved vector policy. It supplies that policy to MultiDatabaseRetriever's keyword-only `media_config` argument and to retrieval. The existing MediaDBRetriever then avoids vector initialization and querying when disabled. Other constructor callers retain their existing default.

Validate the resolver, shared constructor/query policy, real SQLite FTS with vector initialization/query tripwires, existing provider behavior, and native PostgreSQL ownership. No mock-provider UAT or model download substitutes for FTS acceptance.

ADR assessment: no new architecture decision. Existing ADR-017 content authority and the current retrieval configuration/lifecycle contracts govern this narrow option.
