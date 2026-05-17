# Getting Started (Self-Hosting Profiles)

Choose exactly one base setup profile and follow it end-to-end.

Recommended default:
- Run `make quickstart` from the repo root for the Docker single-user + WebUI path. That is the shortest alias for `make setup-docker-single`, `make start-docker-single`, and `make verify-docker-single`.
- Use `Docker multi-user + Postgres` when you need JWT auth, a first admin account, and bundled Postgres.
- Use `Local single-user` for development, debugging, or contributor workflows.

Deployment mode chooser:

| Goal | Start here | Use when | Follow-up docs |
| --- | --- | --- | --- |
| Local single-user | [Local single-user](./Profile_Local_Single_User.md) | You are developing, debugging, or running a local contributor setup. | Add audio only after the base profile verifies. |
| Docker single-user + WebUI | [Docker single-user + WebUI](./Profile_Docker_Single_User.md) | You want the shortest self-hosted path with the bundled WebUI and API key auth. | [Minimal deployment](../Deployment/minimal-deploy.md), [reverse proxy examples](../Deployment/Reverse_Proxy_Examples.md), and [CDN/static assets](../Deployment/cdn-static-assets.md). |
| Docker multi-user + Postgres | [Docker multi-user + Postgres](./Profile_Docker_Multi_User_Postgres.md) | You need JWT auth, a first admin account, and bundled Postgres. | [Postgres migration guide](../Deployment/Postgres_Migration_Guide.md) and [long-term admin guide](../Deployment/Long_Term_Admin_Guide.md). |
| Production or horizontal scaling | [Docker multi-user + Postgres](./Profile_Docker_Multi_User_Postgres.md) first, then [horizontal scaling](../Deployment/horizontal-scaling.md) | You are preparing a shared or multi-node deployment. | [Resource requirements](../Deployment/resource-requirements.md), [sidecar workers](../Deployment/Sidecar_Workers.md), and monitoring/operations docs. |
| Offline or air-gapped | [Offline and air-gapped deployment](../Deployment/offline-air-gapped.md) | You need controlled egress, preloaded models, or disconnected operation. | Validate provider, model, and package assumptions before first production use. |
| Sidecar workers | [Sidecar workers](../Deployment/Sidecar_Workers.md) | You want background workers split from the API/WebUI process. | Use the profile guide first, then add worker compose or service units after the base stack verifies. |
| Audio or GPU setup | [CPU audio setup](./First_Time_Audio_Setup_CPU.md) or [GPU/accelerated audio setup](./First_Time_Audio_Setup_GPU_Accelerated.md) | Speech, transcription, TTS, or diarization is part of day-one setup. | Start from a healthy base profile, then add CPU or GPU audio prerequisites. |
| Monitoring and operations | `make monitoring-up` plus `Docs/Deployment/Monitoring/README.md` | You need Prometheus, Grafana, Alertmanager, or operator runbooks. | Monitoring published output is generated as top-level `Docs/Published/Monitoring`; do not move it by hand. |

Canonical base profiles:

1. [Docker single-user + WebUI](./Profile_Docker_Single_User.md)
   - Prepare: `make setup-docker-single`
   - Start: `make start-docker-single`
   - Verify: `make verify-docker-single`
2. [Docker multi-user + Postgres](./Profile_Docker_Multi_User_Postgres.md)
   - Prepare: export generated `ADMIN_USERNAME` / `ADMIN_PASSWORD`, then `make setup-docker-multi`
   - Start: `make start-docker-multi`
   - Verify: `make verify-docker-multi`
3. [Local single-user](./Profile_Local_Single_User.md)
   - Install: `make install-local`
   - Prepare: `make setup-local-single`
   - Start: `make start-local-single`
   - Verify: `make verify-local-single`
   - No-`make` shortcuts: `./quick-launch.sh`, `quick-launch.command`, or `.\quick-launch.ps1`

Generated multi-user admin bootstrap:

```bash
export ADMIN_USERNAME=tldw-admin
export ADMIN_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
make setup-docker-multi
```

Optional add-ons:

- [First-time audio setup: CPU systems](./First_Time_Audio_Setup_CPU.md)
- [First-time audio setup: GPU/accelerated systems](./First_Time_Audio_Setup_GPU_Accelerated.md)
- [GPU/STT Add-on](./GPU_STT_Addon.md) (legacy pointer to the accelerated guide)

## How To Use These Guides

- Pick the profile that matches your target environment.
- For most users, start with the `quickstart-docker-webui` path via `make quickstart`.
- Treat LAN/custom-host browser access as advanced configuration and stay on the default same-origin browser API requests through the WebUI proxy unless you specifically need another device or origin to reach the API.
- Complete the guide sections in order: prepare, start, verify, first value, audio path, troubleshoot, and optional add-ons.
- Do not mix setup commands from other docs unless the guide explicitly links to them.
- Apply add-ons only after your chosen base profile is healthy.
- If speech is part of day-one setup, switch to the CPU or GPU/accelerated audio guide after the base profile is healthy instead of starting with the older STT/TTS reference pages.

## Notes

- This page is the onboarding index for self-hosting.
- For legacy/deeper reference material, use linked docs from each profile guide.

## Migration Disposition (2026-02-28)

Onboarding setup content was consolidated into these canonical guides.

| Path | Action | Replacement |
| --- | --- | --- |
| `README.md` | migrated | `Docs/Getting_Started/README.md` |
| `Docs/Deployment/First_Time_Production_Setup.md` | redirected | `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md` |
| `Docs/User_Guides/Server/CLI_Reference.md` | redirected | `Docs/Getting_Started/README.md` |
