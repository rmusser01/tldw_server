# UAT157 per-row server-save receipts

TASK13260.95. Root reviewed both source/test diffs: explicit non-empty current-turn user and assistant canonical IDs independently suppress only their matching fallback write. Existing ownership and mode guards remain. Missing opposite-role receipts preserve ordinary fallback. Seven new permanent cases fail4/pass3 before repair; independent final219tests/13suites pass. Author compiler90/90 and lint0errors/17unchangedwarnings. No Python change; Bandit not applicable.

Native fresh image Send/reload and required PostgreSQL acceptance remain pending. No claim of solving genuinely lost-ACK ambiguity or changing legacy no-ACK fallback image handling. Source/test hashes, exact commands and behavior evidence retained; trailing whitespace normalized in copied text only.
