# Server Buddy temporary-chat rebase verification

```json
{
  "tested_source": "3a671a4cb05b8845fd7190826a91bd295b360790",
  "dev": "c867287210d4e85314b00ff22e7d30d0474030a0",
  "previous_published_head": "cefc51ae26cd6c2933d5d7efc730f7cc6ca88b7f",
  "incoming_pr": 3064,
  "rebase": {
    "prior_commits": 5,
    "range_diff": "All five prior commits are patch-identical",
    "runtime_change": "Incoming temporary-chat read-only sidebar and implicit-feedback controls; existing Buddy repair unchanged"
  },
  "verification": {
    "results": [
      {
        "passed": 141,
        "duration_seconds": 27.61,
        "raw_log_sha256": "a44631806cb36a3bc60b6d5bae733f607a0f8407ea71c21fa5d43fed58e5af39"
      },
      {
        "passed": 11,
        "duration_seconds": 0.962,
        "raw_log_sha256": "82662869bc71d55f6d6a345cce716044838f5a404c82916e83d0d0106cbbe61b"
      }
    ],
    "total_passed": 152,
    "incoming_temporary_chat_cases": 50,
    "workspace_contract_cases": 37,
    "shared_workspace_cases": 14,
    "redirect_security_cases": 13,
    "buddy_component_cases": 37,
    "route_lifecycle_cases": 1,
    "node": "26.0.0",
    "vitest": "4.0.18",
    "ci_node_major": 20,
    "ci_status": "Await exact published-head CI; local Node26 results do not replace Node20 gates",
    "bandit": "Not applicable: no Python changes",
    "diff_check": "passed"
  },
  "limits": {
    "full_suite_run": false,
    "paid_provider_run": false,
    "raw_logs": "Retained locally only; not published",
    "native_terminal": "open",
    "installed_extension": "open",
    "upgraded_webui": "open",
    "physical_voice": "open"
  },
  "prior_evidence": "Original WebUI, extension build and FastAPI receipts retain original source attribution"
}
```
