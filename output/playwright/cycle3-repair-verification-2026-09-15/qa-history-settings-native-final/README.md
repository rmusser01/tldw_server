# QA history and Settings native account controls

Existing TASK13260.27 / UAT086 and TASK13260.37 / UAT096, under parent TASK13260. This records targeted native passes; the full fresh single-user/multi-user UAT matrix remains pending.

## Sequence and supported outcomes

Root captured these actions in the existing multi-user browser on port 18281 against server 18201. The retaining reviewer read and validated the artifacts without operating the browser or generating new inference.

1. **Alice identity and own history:** at 23:31:46UTC, actual /api/v1/auth/me responses identify user 2, cycle3_multi_alice. QA shows her Indigo and Cedar Recent entries. The existing Cedar row's settled snapshot shows the saved cited answer, one source and one citation, including Jonah Patel, nine reading tables and the book return beside the north entrance. Root reports activating the existing row; the retained artifact is its resulting snapshot, not a recording of a new model call.
2. **Normal Alice Settings logout:** at 23:32:30UTC, actual Logout button action returns POST /auth/logout 200 and Login Required. Its bounded observers record zero console errors, zero matching disconnected-Form warnings and no pageerror events.
3. **Bob login outcome, then verified identity:** at 23:33:14UTC, the normal Login click reaches Logged In, but principals[] is empty. Root used its safe helper to select Bob's credentials; this capture alone does not independently verify Bob. At23:33:59UTC, actual QA reload /auth/me responses identify user 3, cycle3_multi_bob. No previous QA sessions, Alice Cedar/Indigo question text, or Alice answer are visible. The bounded response observer records no /chat/conversations/ responses during that reload. This is not a claim of zero network requests or indefinite absence of foreign requests.
4. **Bob settled source health:** initial health is checking/unavailable; the later settled snapshot explicitly shows Sources ready: 6 of 8 and still no previous QA sessions. The transient state is not presented as the final result.
5. **Normal Bob Settings logout:** at 23:35:47UTC, Logout returns POST /auth/logout 200 and Login Required, with zero console errors and zero matching Form warnings during the capture. Bob's probe does not separately collect pageerror events.
6. **Alice returns:** following root's normal UI login, actual /auth/me responses at 23:36:22UTC again identify user 2 and her Recent history returns, including the existing Cedar cited-answer entry. The raw login form is excluded because its username field was populated; no credential-entry values are retained.

## Observation limits

The logout listeners stop at the recorded UI outcomes (Alice also waits for DOM content loaded). These successful observations do not prove indefinite clean-console behavior. Whole-session browser warning counts remain in the raw capture and are not equated with Form warnings. All 12 candidate native captures had zero tool-level “### Error” blocks; 11 safe outcome captures are retained, with the populated raw login form omitted. No screenshots were supplied in this bounded artifact set; this bundle makes no new visual geometry claim.

Native targeted history separation, same-account restoration and normal logout pass. No new query/inference or source mutation was performed by the retaining reviewer. No backend authorization conclusion beyond the observed principal IDs and scoped UI/response outcomes is inferred. The full fresh acceptance matrix and unrelated Header/mobile/scene workflows remain outside this bundle.

## Source checkpoint and provenance

Root identifies QA commit d05c13ecc0 and Settings commit a246441257 as unchanged for this native run. Retention checked HEAD 98a97eea7adf0080189c1fbb44c8b0213182ec9b and verified both commits are ancestors. The two primary owner modules exactly match their stated committed versions; source-provenance.json records their hashes. This is not a claim that every mixed shared file from those commits is unchanged. Uncommitted Header CSS under TASK13260.39 is unrelated and not certified here. No browser, runtime, source, test or global documentation change occurred during packaging; official task notes are the tracking updates.

## Integrity

INDEX.md lists every retained file; retention-manifest.json records original paths/sizes/hashes and retained hashes. Native text only permits trailing-whitespace/final-newline normalization; JSON is generated provenance/audit data. SHA256SUMS covers every file except itself. All input and generated text was checked before writing against 14 known isolated runtime credential values plus JWT/private-key patterns, with zero matches. Runtime-private files and credential values are not included. Later independent SHA validation and task notes record completion.

Raw CLI .playwright-cli links refer to original temporary capture locations. Their inline result/snapshot content is the durable evidence retained here. Older automated regression logs matching the broad final-* prefix are excluded because this is the native outcome bundle.
