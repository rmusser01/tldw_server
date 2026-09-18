# UAT236 native acceptance review — TASK13260.178

## Verdict

**ACCEPTED for the picker/provider/model-availability repair.** The retained PG-single sequence shows the configured qualified model as healthy, opens the configured-model picker, and after reselecting the same choice renders it as a healthy **LLaMa.cpp** selection. The retained flow then enters Character mode, sends, and reloads the resulting saved conversation. That is the UAT236 boundary: the `llama` catalogue identity, picker/provider identity, and selected-model owner agree without changing the suffix.

The frozen implementation is also adequate on its own terms. The only provider-resolution additions normalize the closed `llama`, `llama-cpp`, and `llama_cpp` aliases to `llama.cpp`; availability adds the corresponding `llama` alias; Settings replaces the disconnected storage setter with the existing live selected-model owner. The independent focused suite passed **82 tests in 3 files** (0 skips): suffix preservation, configured availability, unconfigured/catalog-only blocking, and a mounted Settings owner that survives remount.

## Native evidence

| Evidence | Finding | SHA-256 |
| --- | --- | --- |
| `model236-picker.txt` | The configured qualified choice is present and healthy in the picker. | `ca38013…ba25e` |
| `model236-reselected.txt` | Reselecting it presents the same model as healthy `LLaMa.cpp`, rather than the stale generic grouping. | `24810e…28ae` |
| `model236-sent.txt` and `model236-reloaded-output.txt` | The retained Character flow sent and reloaded after that selection. | `ec40bc…46d6`; `0ac938…d572` |
| UAT236 canonical projection | The conversation contains the requested user/assistant pair; output semantics are assessed separately below. | `52a206…0a82` |

The author’s frozen 16-path manifest is `437130…97c8c`; its freeze verification is `05d880…d0d0e`. The three UAT236 production snapshots are `resolve-api-provider.ts` `4228aa…0db1`, `chat-model-availability.ts` `54482c…aa00`, and Models Settings `099bd3…1b05`.

## Strict separation from UAT261 and UAT263

This does **not** claim exact TestBot success. The UAT236 canonical projection marks its exact-output predicate false because the final message contains additional text. The UAT261 canonical projection independently retains the same non-exact/no-final outcome; its retry-branch projection has an exact child response, but a child branch does not establish the original canonical conversation. These remain UAT261/model-output findings, not picker availability proof.

UAT263’s reviewed `nextChatId` route replacement protects a stale-URL/store race after a Character retry branch is accepted. Its current source hash is the complete `ec6ddb62f75bf7c12eceae7d8db1d2db7c1cb79e3d0941662cb2a2d5d012d3d0`; the author report omitted the final `0`. That implementation review does not retroactively turn the retained UAT236/UAT261 capture into a native replay of the stale-route path.

## Provenance and limits

The browser profile is the preserved `repairs231-250-targeted-20260917` PG-single profile. Its safe manifest identifies PostgreSQL single-user mode and original source revision `86458ab88ce3fa62e6518c9d813c3860254ddb2c` (`c6b146…abee`); the original source manifest is `b90438…c4ef`. The later runtime source manifest is `a7d3155a567afb25982eb360ea24b973cc3249c9` (`b79010…22fb`). Therefore this native evidence is **not** asserted as a replay at the frozen `2787043410fc918b2c280d90f753d8fd02b6b35b` source baseline. Frozen-278 is bound by the reviewed UAT263/UAT264 retention packets, not by relabelling the native run.

No browser, runtime, model/provider configuration, product source, task record, or Git state was changed for this review. `AUDIT.json` contains the complete safe input-hash inventory and command/result; no provider reasoning, credentials, private profile contents, request bodies, or runtime logs were retained here.
