# Calm Context and Runtime rails

Task: TASK-13924. Mode: Operate. User confirmed two separate rails, simplified and clarified, as a separate piece after the composer redesign.

Context explains what shapes the next reply: prompt and assistant identity together, one source inventory with existing individual and bulk actions, conversation/persistence details, then the full request preview behind a disclosure. Runtime explains how the reply runs: status and run/recovery controls together first; one provider/model block with settings overrides and disclosed defaults; then MCP availability and tool controls.

Use the incumbent theme and typography with flat sections, full-row disclosure targets, quieter secondary actions, readable source rows and fewer repetitive idle/default badges. Keep warnings beside their recovery actions. Both rails retain independent saved visibility, focus mode and mobile access. Preserve callbacks, source-specific clearing, persona/character distinction, scoped provider settings, model readiness, disabled reasons, privacy rules and history ownership. The composer, backend and global design tokens stay outside this change.

ADR required: no. The change moves presentation and existing callbacks within the shared Chat UI; it establishes no new API, storage, provider, auth or ownership rule. ADR-049 continues to govern chat history ownership.

Validation: failing-first behavior tests, existing context/runtime/assistant/composition and integrated shell tests, TypeScript and scoped lint, then one desktop/mobile light/dark browser capture matrix with axe, persisted visibility and focus checks. Run Bandit for applicability, a fresh code/design review and a documentation check. Live provider or microphone use is outside the visual fixture.

Final validation (2026-10-23): TypeScript passed; scoped lint introduced no findings across 19 files. Focused tests passed 128 cases with two unchanged baseline failures. Five offline browser cases passed across desktop/mobile and light/dark, covering scoped axe (zero violations), rail visibility persistence, focus restoration and draft preservation; the six final captures and manifest are in `.impeccable/review/rails/`. The one-time design detector found no issues. Fresh code and visual reviews recommend shipping within that offline capture scope. Bandit could not parse the TypeScript scope (no Python changed); live providers, microphone use and a production bundle were not validated.
