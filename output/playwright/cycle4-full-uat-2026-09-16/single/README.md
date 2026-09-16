# Cycle4 single-user UAT evidence — running checkpoint

Product freeze: `7c9409fad2`. Branch `codex/fresh-install-uat-fixes` contains verified dev tip `2e1a5e58d3`; the original task did not start from that tip. See [running matrix](RUNNING_TRACKER.md) and the repository's global UAT tracker.

This is fresh configuration, databases and browser state using existing project dependencies. It does not certify clean-machine dependency installation. Real inference uses the model discovered from local endpoint9099; no mock inference counts as acceptance.

The `evidence` directory contains credential-scanned CLI snapshots, safe request/response bodies and read-only API records. Some `.json` files are CLI output envelopes rather than raw JSON. Read their contents and accompanying matrix: a command acknowledgment alone does not prove its intended outcome. Empty failed-observer outputs are excluded. `evidence-manifest.json` records SHA256 hashes and exclusions. This bundle will be extended as the run continues.

## Key evidence

- Setup: first-chat-response, first-chat-ready, home, provider-required.
- Ingest: ingest-results, media-first, aster-media, minimize-control, wiki-actual-review and wiki-result.
- Chat: default-chat-502, configured-chat-400/request, provider-restart, recovered-chat-visible/network, two-turns-visible, normal-server-messages and chat-settled-reload.
- Notes/cards: note-saved, note-more, flashcards-loaded/result/saved, five-cards, study-complete/reloaded, review-2-through-7 and early-end-result.
- Accessibility: create-another-stuck is the initial hypothesis filename; create-button-semantics disproves a stuck submission. Invisible loading-icon accessibility text changes the name of otherwise enabled buttons. Native activation succeeds.

## Scope and corrections

The API18300 restart is an explicit workaround for UAT107, with unchanged config/code hash provenance. Original errors remain failures. Console connection errors observed during this planned outage are not new unexplained product failures.

Recorded automation corrections include a hidden provider suggestion, a card observer matching both hidden Manage content and visible Study content, AntD radio input interception by its clickable label, and a Cram observer expecting a schedule-review POST. Cram is observed to advance without changing scheduled totals. The seven-card due review uses actual confirmed transitions and seven distinct cards; early End uses a separate two-card fixture and records one successful rating before End200.

No full single-user or multi-user acceptance claim is made at this checkpoint.
