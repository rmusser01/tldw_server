# Source Chat targeted acceptance — 2026-09-16

Frozen source `223591ac4fb2520290634c74310351ce7d8b18ae`; existing isolated SQLite single-user profile, actual local model, synthetic public Rowan/Cedar fixtures. This is targeted repair acceptance, not a fresh full workflow matrix or native PostgreSQL certification.

- Home's actual file flow recognizes the already-ingested identical Rowan document and enables its starter in the saved Cedar conversation. Initial media creation was actual ingestion request21; Home's repeated file is correctly deduplicated. No claim of a new Home-created media row.
- A controlled abort of the one actual selected-source RAG request58 produces a local diagnostic pair. Reload shows five UI rows and three canonical rows. A subsequent ordinary genuine turn leaves seven UI rows and five canonical rows, without promoting or duplicating the local pair.
- That ordinary request79 reveals **UAT163**: reload lost retrieval activation, so it contains Cedar history and a question mentioning Rowan but no retrieved Rowan facts. Keep this failure separate from the historical013 wrong answer despite supplied facts.
- Reapplying the actual Home starter makes scoped RAG request124 with media1, HTTP200 after **14,588ms**. Completion request126 contains real returned Rowan facts and answers correctly. This exceeds the former ten-second Settings timeout without transport fakery or retry.
- Final reload has seven canonical rows and nine visible rows. The one local diagnostic pair remains visible and absent from canonical/provider history. The genuine source question precedes its answer, with exact canonical IDs in `acceptance-summary.json`; no extra fallback message writes occur.

`final-source-reloaded.png` was visually inspected. `real-source-answer.png` captures the viewport before scroll settlement and does not show the full final answer; the settled reload image and snapshot do. Streamed completion response bodies are unavailable to this CDP observer, with terminal ERR_ABORTED events. Do not infer raw SSE acknowledgements from these logs: completed text and identities are established by canonical200 reads and settled UI.

Text copies normalize only trailing whitespace and extra EOF blank lines. `retention-manifest.json` retains original and resulting hashes; binary screenshots are unchanged. Private credentials, headers, profile files and service logs are excluded. Read the independent review for acceptance scope and remaining limitations.
