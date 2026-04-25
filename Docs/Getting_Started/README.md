# Getting Started (Self-Hosting Profiles)

Choose exactly one base setup profile and follow it end-to-end.

Recommended default:
- Run `make quickstart` from the repo root for the Docker single-user + WebUI path. That is the shortest alias for `make setup-docker-single`, `make start-docker-single`, and `make verify-docker-single`.
- Use `Docker multi-user + Postgres` when you need JWT auth, a first admin account, and bundled Postgres.
- Use `Local single-user` for development, debugging, or contributor workflows.

Canonical base profiles:

1. [Docker single-user + WebUI](./Profile_Docker_Single_User.md)
   - Prepare: `make setup-docker-single`
   - Start: `make start-docker-single`
   - Verify: `make verify-docker-single`
2. [Docker multi-user + Postgres](./Profile_Docker_Multi_User_Postgres.md)
   - Prepare: `ADMIN_USERNAME=admin ADMIN_PASSWORD='replace-with-a-long-password' make setup-docker-multi`
   - Start: `make start-docker-multi`
   - Verify: `make verify-docker-multi`
3. [Local single-user](./Profile_Local_Single_User.md)
   - Install: `make install-local`
   - Prepare: `make setup-local-single`
   - Start: `make start-local-single`
   - Verify: `make verify-local-single`

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
