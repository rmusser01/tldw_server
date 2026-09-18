# UAT236 timestamp-specific provenance supplement

This supplement **supersedes the provenance-limit paragraph only** in `REVIEW.md` and `AUDIT.json`. It preserves the picker-only verdict and every UAT261/UAT263 limitation.

The retained time/order binding supports the UAT236 picker sequence as an explicit frozen-`2787043410fc918b2c280d90f753d8fd02b6b35b` runtime upgrade, not merely an earlier `a7d315…` capture:

| Receipt | Safe binding fact | SHA-256 |
| --- | --- | --- |
| `retry256-worldbook255-gate.json` | Released targeted existing-profile upgrade; revision `278704…`; original run `repairs231-250-targeted-20260917`. | `005697aa…f1e3` |
| `retry256-startup-safe.json` | Revision `278704…`; second identical launcher calls started the retained cells without profile/data changes. | `b267bcd5…748c` |
| PG-single `binding.private.json` | Private input inspected by key selection only: cell `pg-single`, source commit `278704…`. | `b4012e7a…1172` |
| PG-single frontend process receipt | Frozen `278704…`, started `2026-09-17T23:45:09.480Z`. | `16f9a8f0…e401` |
| PG-single backend process receipt | Frozen `278704…`, started `2026-09-17T23:46:45.806Z`. | `d761c147…9379` |
| `model236-picker.txt` | Picker capture at `23:54:19.166Z`. | `ca38013d…ba25e` |
| `model236-reselected.txt` | Reselection capture at `23:55:27.168Z`. | `24810e5d…28ae` |
| `model236-sent.txt` | Character-mode send at `23:57:24.318Z`. | `ec40bc78…46d6` |
| `model236-canonical-final-projection.json` | Canonical readback at `2026-09-18T00:00:33.575Z`, HTTP 200. | `52a2060a…0a82` |

The ordered timestamps bind the picker/reselection/send/readback to the frozen-278 Retry256 runtime. The original source profile and its data remain the preserved targeted profile; no provider/model/browser configuration was changed for this sequence. The separately retained `a7d315…` native captures are valid only under their own upgrade source and are not used for this conclusion.

The `retry256-canonical-final-projection.json` independently records seven canonical rows and `exactFinalExpected: true` for the Retry256 scenario (`6d91c0cb…d70a`). That is useful runtime continuity evidence, but it is not substituted for UAT236’s model-output predicate. UAT236 remains accepted only for picker/provider/model availability; the model236 canonical projection still has a non-exact output, and the UAT261 outcome remains open.

No private receipt contents, credentials, provider reasoning, request bodies, or logs are reproduced here.
