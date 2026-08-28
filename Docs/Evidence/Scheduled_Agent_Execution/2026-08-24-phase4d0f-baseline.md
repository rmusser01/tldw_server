# Scheduled Agent Execution Feasibility Evidence

| Field | Value |
| --- | --- |
| Evidence ID | `sha256:1df8024b73472ea0a02a323fbad0d2f864d8b5f604611cb01bf49478f60a5874` |
| Deployment class | `sha256:76a1074c303c74cd6db3f6823f391133e44437a0da019f99f5b02b95b2cb3337` |
| Source commit | `a43949d4b06a5a633619c0c8227ecb9771ddde28` |
| Created | `2026-08-27T03:13:27.817481+00:00` |
| Valid until | `2026-08-28T03:13:27.817481+00:00` |
| Outcome | `draft_only` |

## Reasons

- `adapter_dispatch_recovery_missing`
- `adapter_dispatch_recovery_unverified`
- `authoritative_receipt_missing`
- `brokered_credentials_and_mediation_missing`
- `brokered_credentials_and_mediation_unverified`
- `deployment_identity_unverified`
- `hostile_boundary_missing`
- `hostile_boundary_unverified`
- `isolation_attestation_missing`
- `isolation_attestation_unverified`
- `isolation_profile_identity_unverified`
- `monotonic_execution_evidence_missing`
- `monotonic_execution_evidence_unverified`
- `operational_fail_closed_missing`
- `operational_fail_closed_unverified`
- `scheduled_transcript_non_disclosure_missing`
- `scheduled_transcript_non_disclosure_unverified`

## Requirements

| Requirement | State | Verification | Evidence |
| --- | --- | --- | --- |
| `isolation_attestation` | `missing` | `repository_characterization` | `sha256:7a5b003949395c6c2ef2a6496dceea781b8f6389a0705eaaea8289df0d2cf481` |
| `hostile_boundary` | `missing` | `repository_characterization` | `sha256:cc184783ba5a2d774f6d05a1bbab7ab869e100652764ae5d5be18a2cdc87bfdc` |
| `scheduled_transcript_non_disclosure` | `missing` | `repository_characterization` | `sha256:46ecbb64e917dbf6d692daaaf0956f2337b45dd522375c65e17b99904f23dbd4` |
| `adapter_dispatch_recovery` | `missing` | `repository_characterization` | `sha256:d53e28686dd45800f19b68fab3a7e05645ddf12367a20a0ae0d28665a7fdbbd8` |
| `monotonic_execution_evidence` | `missing` | `repository_characterization` | `sha256:12c4b0382054cc98b45f4e3c00df5925be7332a591e16c493dfa0ae0ed83eacd` |
| `brokered_credentials_and_mediation` | `missing` | `repository_characterization` | `sha256:cbc36034e35fb166ea862630fa0af90a1235d91be684c48eadb1c87f9ed39aae` |
| `operational_fail_closed` | `missing` | `repository_characterization` | `sha256:f775c9a7bc1e5f803764eb1f19cbc29100270bd8703916a146333241a2cade2f` |

## Repository-Static Runtime Eligibility

This appendix is derived from typed runtime metadata. It is not deployment certification evidence.

| Runtime | Default outcome | Primary reason |
| --- | --- | --- |
| `docker` | `draft_only` | `required_server_verified_evidence_incomplete` |
| `firecracker` | `draft_only` | `required_server_verified_evidence_incomplete` |
| `lima` | `draft_only` | `required_server_verified_evidence_incomplete` |
| `vz_linux` | `draft_only` | `required_server_verified_evidence_incomplete` |
| `vz_macos` | `unsupported` | `runtime_not_untrusted_eligible, runtime_strict_deny_all_unavailable` |
| `seatbelt` | `unsupported` | `runtime_not_untrusted_eligible, runtime_strict_deny_all_unavailable` |
| `worktree` | `unsupported` | `runtime_not_untrusted_eligible, runtime_strict_deny_all_unavailable` |

Repository characterization is not deployment certification. Host-gated raw evidence is retained outside the repository.
