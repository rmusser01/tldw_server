# tldw-agent Migration

- Source repo: `../../../tldw-agent`
- Upstream commit: `2fa0bef9d5e1d3fb5c5949762bfef80ef3c14b68`
- Migration date: `2026-03-10`
- Migration rule: preserve existing behavior before adding `vz_linux` guest mode

## Notes

- This directory is now the in-repo source snapshot used for `vz_linux` helper and guest-agent work.
- The first migration slice intentionally avoids package renames or behavioral rewrites.
- Follow-up work should preserve existing host/native-messaging and ACP behavior before extending the agent for VM guest use.
