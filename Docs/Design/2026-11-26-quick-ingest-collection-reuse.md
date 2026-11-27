# Resume Quick Ingest collection setup

A user can retry an interrupted metadata import in the original collection. Reopening the same draft or retrying it from a second tab preserves the rows already saved. Starting a fresh import creates its own collection, including when it contains the same URLs.

## Setup and recovery

The wizard supplies a persistent plan key and a source key for each queue row. The collections API derives ownership from authentication. Its plan endpoint receives the collection metadata and the requested source rows in one write. A new source takes its metadata from that request. A previously bound source returns its current ID, metadata, order and processing state. A lost acknowledgement is recovered by repeating the same keys. The browser can resume a subset without removing other sources.

An unsuccessful setup request stops dispatch. A successful response must identify each requested source exactly once before the browser uses it. The receipt for a saved source opens that Media record. A source with an active job reconnects to that job. Queued submissions carry a stable logical idempotency key into Jobs. The pre-acknowledgement cancellation change in #3488 gives each native upload a distinct transport nonce so stopping a new attempt cannot cancel an earlier accepted job.

Two clients may also reach a direct scrape or synchronous upload fallback together. The server decides which caller may dispatch that side effect. If delivery or completion is uncertain, the receipt reports that uncertainty. A stale browser cannot replace an accepted or saved receipt with a setup failure.

## Verification

The database tests exercise a failed write, concurrent resumes, reordered sources, later edits, separate owners and both fresh and upgraded SQLite and PostgreSQL databases. The receiver tests cover Jobs, direct scraping and the synchronous upload fallback.

In Chromium at 1280 and 390 pixels, the server fixture committed two named source rows and dropped the setup response. Retrying preserved both row IDs and titles, dispatched each URL once and showed both saved Media links. [Desktop receipt](assets/quick-ingest/batch-metadata/collection-setup-recovered-desktop.png) · [Phone receipt](assets/quick-ingest/batch-metadata/collection-setup-recovered-mobile.png)

A separate browser run used an isolated HTTP server and media worker. The first request committed the two source rows for a pair of native text files before its acknowledgement was discarded. After the user retried, the collection database contained the original two rows. Jobs contained two completed jobs. Each row identified the job and Media record containing its own file text.

ADR required because this introduces a durable client-to-server collection identity. [ADR-103](../ADR/103-quick-ingest-collection-plan-identity.md)
