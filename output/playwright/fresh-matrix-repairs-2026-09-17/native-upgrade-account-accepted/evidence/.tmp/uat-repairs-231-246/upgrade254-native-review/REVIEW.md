# UAT254 actual startup and preservation review

**CLEAR — corrected startup and the reviewed original-record readbacks are supported. No actionable gaps in this bounded scope.**

Audit completed: 2026-09-17T21:53:23.433Z. Source candidate: `a7d3155a567afb25982eb360ea24b973cc3249c9`. Original profiles: `repairs231-250-targeted-20260917`. Corrected copy-run: `repairs251-254-upgrade2-20260917`.

## Actual startup and provenance

- Live API PIDs **64579 / 64596** and Next parent PIDs **64737 / 64765** exactly match the recorded executable/arguments and working directories. Next listener children **64740 / 64768** belong to those parents. Both frontends independently returned **HTTP 200** on **18782 / 18783**. Backend startup-complete and Next ready lines are present in the hashed private logs. HTTP 401 health observations are treated only as authentication boundaries.
- Actual API PYTHONPATH points to the corrected source copies; API cwd, config, environment-file and user-data paths remain under the original profiles. Actual Next cwd/CLI use the new copies and the dedicated `.next-live-tier-upgrade-*` directories, with the original internal API origins. Current launcher/helper hashes match both immutable bindings.
- Released gate, completion, source-manifest and Python-reuse proof hashes match the bindings. Copied backend entry, Next config and networking validator bytes match their source manifests. The actual Next config hash remains `55f31677133e8fc03759d961e6bca7534c1a6d007fd0554723ee54d5bd5e3188`.

## Original identity and data

- Both original **profile, initialization and holder files are byte-identical to the hashes in both the first failed-run bindings and the corrected-run bindings**. Initialization remains the original completed records from **19:52:08.427 / 19:52:11.537 UTC**. Original PostgreSQL fixture holders **18859 / 18878** remain alive and retain their original run/source identity. This supports preserving initialization without reset across this retry.
- Single-user **Media 1** returned **GET 200**, content length **1,914**, and original version creation **2026-09-17 20:32:22.772 UTC**. The earlier observer-error artifact remains preserved. Content SHA256: `a94b1e966d89b7b94e0cd69dafe9ab1c554dc81accf43e57957276b08294225c`.
- Bob's original chat `e3755001-1cd4-40a4-aaaf-edd4b67b3418` reloaded under authenticated Bob identity (user 3), with six successful canonical message responses. The **two records present in the pre-upgrade canonical capture match whole records exactly**. The third assistant record retains its pre-upgrade timestamp and the same content displayed by the original terminal capture. This distinction avoids implying that the earlier canonical response already contained all three records.
- Browser evidence records an automation tab at **about:blank**, followed by normal navigation to the original chat in the authenticated context. The captured recovery sequence contains no login request. This proves recovered access to the original chat, **not continuous survival of the same tab**. Later explicit Bob logout/Alice login artifacts remain separate.

## Evidence and limits

The audit contains **44 evidence hashes**, safe process/source provenance and all comparison results. Audit SHA256: `be0127da527763cc46b450c3885b8de84c8ebc6dcdc2491cda7bf9873f793be1`. Key native evidence hashes:

| Evidence | SHA256 |
| --- | --- |
| Media upgraded read v2 | `f8dd8bc0ef2d9fafd77a05abde1ba331f6443962aae7afdaf41a4b5b236f5970` |
| Bob upgraded capture | `0d246af6591a44573f77bcec4fb1e533395ba51538511cd6b85792a9faa1a5dc` |
| Bob original canonical reload | `9efac2a01c1e77cb0f9cab977e9e00a3d7812b28d567a06c3754621fcfa97eb8` |

No tests were repeated. This reviewer used read-only process inspection, local HTTP GETs and retained native evidence; no browser/DB mutations, starts/stops, resets, source copies, Git, Backlog or tracker edits. Only this three-file review packet was written. No secrets or full private logs were printed or retained. The process sandbox initially blocked ps; the approved read-only escalation supplied the live checks.

This is corrected startup plus bounded original-data preservation, not full database equivalence, fresh-install acceptance or full-matrix UAT. All 35,159 tracked source files per cell and the archive were not independently rehashed in this audit; source-proof identities and the stated entry/config bytes were checked.
