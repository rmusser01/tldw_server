# Ergo IRC Community Migration Design

Backlog task: `TASK-12165`
Date: 2026-07-06
Status: Draft for spec review

## Goal

Make Ergo IRC the default communications area for the tldw project/community while preserving a permanent Discord compatibility path. The launch should be simple enough to operate on one small VPS, explicit about public support logging, and structured so Discord is useful without becoming the canonical community home.

## Decisions

- Run the community stack on a new small VPS.
- Use Docker Compose for the deployable unit.
- Use Ergo as the IRC server.
- Use Kiwi IRC as the web client.
- Use Caddy for TLS, static web serving, and WebSocket proxying.
- Use Matterbridge for Discord compatibility.
- Use `chat.<project-domain>` for the browser entry and `irc.<project-domain>` for native IRC clients.
- Launch channels: `#tldw`, `#support`, `#dev`, and `#announcements`.
- Launch with no-email IRC account self-registration, then enable email verification in milestone 2.
- Require registered IRC accounts to speak in public IRC channels.
- Make `#announcements` public read-only for normal users; only project operators or explicitly voiced announcement accounts may speak.
- Keep Discord permanently bridged, but document IRC as canonical.
- Make `#support` public, searchable, archived for 365 days, and clearly disclosed at every entry point.
- Treat `<project-domain>` as an intentional deploy-time placeholder. The implementation plan must replace it with the real tldw community domain before deployment.

## Architecture

The launch stack has five services:

- `ergo`: IRC server and canonical community state.
- `kiwiirc`: static browser IRC client served at `chat.<project-domain>`.
- `caddy`: HTTPS endpoint, static file server, and reverse proxy for Ergo's WebSocket listener.
- `matterbridge`: bridge between Discord and selected IRC channels.
- `support-archive`: small `#support`-only log collector and static archive generator.

DNS:

- `chat.<project-domain>` points to the VPS and serves Kiwi over HTTPS.
- `irc.<project-domain>` points to the VPS for native IRC over TLS port `6697`.
- `https://chat.<project-domain>/support/` is the default public support archive route.

Network exposure:

- Public: `443/tcp` for web and IRC-over-WebSocket.
- Public: `6697/tcp` for native IRC over TLS.
- Admin-only: SSH.
- Not public: plaintext IRC `6667`; keep loopback-only or disabled.

## Channel Model

`#tldw` is the general community channel. It is bridged two-way with Discord.

`#support` is the user support channel. It is bridged two-way with Discord, logged by `support-archive`, and published as a public searchable archive with 365-day retention.

`#dev` is for contributor and development discussion. It is IRC-only at launch to preserve an IRC-native area and avoid making every important space Discord-backed.

`#announcements` is canonical on IRC and mirrored one-way into Discord. It is publicly readable but not publicly writable. Normal registered users must not be able to post there; use a moderated/read-only channel setup where only IRC operators or approved announcement accounts can speak.

Matterbridge configuration must make direction explicit:

- `#tldw`: IRC `inout` plus Discord `inout`.
- `#support`: IRC `inout` plus Discord `inout`.
- `#announcements`: IRC `in`, Discord `out`.
- `#dev`: no bridge.

## Moderation And Abuse Controls

Launch registration policy is open but gated:

- Users can self-register IRC accounts without email.
- Unregistered IRC users can join and read public channels.
- Public channel speaking requires a registered IRC account.
- Ergo registration throttling, login throttling, connection limits, nickname reservation, and account/IP/network bans are enabled.

Accepted exception: Discord users in bridged Discord channels can speak into `#tldw` and `#support` without an IRC account. Abuse control for those messages depends on Discord channel permissions and the bridge kill switch.

Required operator controls:

- Register the Matterbridge IRC account.
- Do not grant the Matterbridge account IRC operator/founder powers.
- Limit Matterbridge to the bridged channels only.
- Keep `#announcements` write access restricted to project operators or approved announcement accounts so one-way Discord mirroring cannot become an accidental public broadcast path.
- Give the Discord bot only the permissions needed for the mapped channels.
- Configure Matterbridge message delay and queue limits so Discord bursts can drop bridge messages instead of flooding IRC.
- Document a kill switch for disabling all Matterbridge traffic or one gateway.

Milestone 2 enables email verification for new IRC registrations. This should remain a configuration milestone, not a custom identity system.

## Public Support Archive

`#support` needs a real public archive because Ergo's client history replay is not a public searchable support knowledge base.

The launch archive should be deliberately boring:

- A small IRC logbot or sidecar records only `#support`.
- Static daily pages are generated.
- A static search index is generated from retained messages.
- Message entries include timestamp, source protocol (`irc` or `discord`), display name, and text.
- Discord-origin messages must be tagged before they enter IRC, using Matterbridge remote nick formatting or an equivalent gateway-specific prefix such as `[discord] <display-name>`. The archive collector identifies Discord-origin entries from the Matterbridge IRC account plus that configured source marker; all other `#support` messages are recorded as IRC-origin.
- Attachments are not downloaded, rehosted, or embedded. Text links may be shown.
- Entries expire after 365 days.
- Raw logs, generated pages, search index, and backups all follow the same 365-day retention policy.

Public archive launch posture:

- The archive is public and locally searchable.
- Add `noindex` at launch while redaction and moderation operations are proven.
- Revisit search engine indexing after real use and after the redaction process has been exercised.

Redaction requirements:

- Provide a moderator runbook for deleting or redacting an archive entry.
- Rebuild static pages and the search index after redaction.
- Document how users request removal of accidental secrets or personal information.
- Treat pasted secrets as compromised; archive deletion does not make a secret safe again.

Disclosure is a launch blocker. The public archive warning must appear in:

- `#support` topic.
- Discord `#support` channel description.
- Kiwi welcome text or MOTD.
- Public archive header.
- Community/support docs.
- Bridge bot/app description where practical.

Suggested warning:

```text
#support is public. Messages here may be bridged between IRC and Discord and published in a searchable support archive for up to 365 days. Do not post secrets, tokens, private logs, or personal data.
```

## Discord Policy And Platform Considerations

The Discord bridge must have a clear stated function: compatibility access between the tldw Discord and the IRC-first community. The Discord bot/app description and channel descriptions must say that messages in bridged channels may be relayed to IRC, and `#support` messages may be archived publicly for 365 days.

The bridge likely needs Discord Message Content Intent to relay normal channel text. Setup must include enabling the required intent and confirming the bot remains within Discord's policy and review requirements. If the app grows toward Discord's review thresholds, reassess whether permanent bridging is still worth the operational and policy burden.

Discord message content must not be used for AI/model training, profiling, advertising, or any purpose outside bridge and support archive functionality.

Sources:

- Ergo README and operator docs: https://github.com/ergochat/ergo
- Ergo Docker docs: https://raw.githubusercontent.com/ergochat/ergo/stable/distrib/docker/README.md
- Matterbridge config sample: https://raw.githubusercontent.com/42wim/matterbridge/master/matterbridge.toml.sample
- Discord Developer Policy: https://discord.com/developers/docs/policies-and-agreements/developer-policy
- Discord Privileged Intents: https://support-dev.discord.com/hc/en-us/articles/6207308062871-What-are-Privileged-Intents

## Backup And Restore

Back up only the state that matters:

- Ergo datastore and config.
- Caddy config.
- Matterbridge config and secrets.
- `support-archive` raw logs, generated pages, and search index.

Do not rely on container state as the backup boundary. Restore testing must prove that a staging directory can recover the IRC datastore and support archive without requiring production secrets to be exposed in docs.

Backup retention for support archive content must not exceed the public archive retention. If `#support` expires at 365 days, backups containing support logs must also age out.

## Operations Runbooks

The implementation plan must create concise operator docs for:

- Start, stop, and restart the stack.
- Rotate Discord bot token.
- Rotate Matterbridge IRC password.
- Disable Matterbridge globally.
- Disable a single Matterbridge gateway.
- Redact/delete a support archive entry and rebuild search.
- Restore Ergo datastore from backup.
- Restore support archive from backup.
- Upgrade Ergo, Matterbridge, Kiwi, and Caddy images.

## Launch Checks

- DNS resolves for `chat.<project-domain>` and `irc.<project-domain>`.
- `https://chat.<project-domain>` loads Kiwi.
- Kiwi connects through `wss://chat.<project-domain>/webirc`.
- Kiwi joins `#tldw` and `#support`.
- A native IRC client connects to `irc.<project-domain>:6697` with TLS.
- Public plaintext IRC is not exposed.
- Unregistered IRC users can join and read public channels.
- Unregistered IRC users cannot speak in public channels.
- Registered IRC users can speak.
- Discord to IRC works for `#tldw`.
- IRC to Discord works for `#tldw`.
- Discord to IRC works for `#support`.
- IRC to Discord works for `#support`.
- IRC to Discord works for `#announcements`.
- Discord to IRC does not work for `#announcements`.
- `#dev` is IRC-only.
- `#support` archive displays the public archive warning.
- `#support` archive search works.
- `#support` archive labels `irc` vs `discord` origin.
- Discord-origin support messages carry the configured bridge source marker before archive collection.
- `#support` archive excludes attachments.
- `#support` archive entries expire after 365 days.
- Support archive backups do not outlive 365-day retention.
- Bridge flood controls are configured.
- Matterbridge kill switch is documented and tested.
- Redaction runbook is tested on a staging archive entry.
- Backups restore in a staging directory.

## Milestone 2

- Enable email verification for new IRC registrations.
- Revisit whether Discord should remain permanently writable.
- Revisit public search engine indexing for the `#support` archive.
- Revisit `#support` archive search quality after real support traffic exists.
- Revisit whether `#dev` should remain IRC-only.

## Non-Goals

- No custom tldw chat server.
- No custom bridge service unless Matterbridge fails in practice.
- No Matrix, Slack, Telegram, or additional bridge targets at launch.
- No archive database or admin web app unless static files and static search fail.
- No Discord category/channel mirror beyond the selected channel mapping.
- No rehosting Discord attachments.
