# UAT209 / UAT210 — reviewed targeted native acceptance

Independent bounded verdict: native TASK13260.147 AC3 and TASK13260.148 AC3 pass, combined with their previously retained implementation/regression review. See audit/REVIEW209-210-native.md for exact evidence and limits.

Alice2/Bob3 have separate actual Notes/keyword catalogues, bidirectional foreign GET/PATCH404, own save/read200, original Alice source unchanged, Alice browser autosavev3/manualSavev4/fullreload cataloguev4, and Bob fullreload cataloguev2. Both reloaded editors show New note; no reopened-editor claim.

Administrator1 focused/all graph HTTP200 returns its one note+tag+edge. The initial `passed:false` graph receipt is intentionally retained: it expected the frontend note: prefix on raw API UUIDs. The corrected read-only continuation passes using the same note. Alice graph403 and its UAT221 UX remain separate and open in this audit.

Native provenance: restart source6d06aae9bd, API76778; all3668 before/after startup source hashes matched. Runtime service role remains privileged/BYPASSRLS. This is application ownership acceptance, not native restricted-role/RLS proof or graph browser visualization acceptance.

Only synthetic fixture contents are retained. Notes event extracts keep exact event objects since08:15UTC for identity or Notes endpoints and include original source hashes/rules. Original cumulative event/source-manifest files are additionally secret-scanned but not copied. Prior source/static/fixture proof remains in adjacent followup209-210-217-reviewed-supplemental and followup210-graph-compatibility-reviewed packages.

Original evidence remains unchanged. Explicit allowlist only; raw private logs, configs, credentials and native databases are excluded. Original and normalized text were both scanned against known runtime API/JWT/hash/account passwords, vision key and PostgreSQL passwords, plus JWT/PEM patterns: zero matches. Full cumulative captures and source manifests remain in the private packet; scoped unmodified events and exact original-source hashes preserve attribution. Embedded hashes in original reports refer to original bytes; manifest.json records every retained-byte hash.
