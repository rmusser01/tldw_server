# Bundled pixel-migu

Fresh per-user SQLite workspaces include **pixel-migu** in the character picker,
with a portrait and an active Shared Visual Identity pack containing 18 expression
slots. Select the character normally: the existing chat portrait, mood resolver,
expression picker and `/emote` flow use this binding. No import or provider call
is needed. This does not change the selected character or conversation.

For Buddy, open the selected Persona's visual starter catalog, choose
**pixel-migu**, then review and activate the copied draft using the normal Buddy
workflow. It contains 64 transparent PNGs, twelve four-frame sequences, sixteen
static poses and 31 state mappings. Buddy remains disabled/unselected until the
user configures it. **Migu Marker Basic** remains a separate starter.

Startup installs the character only once. Renaming, editing, unbinding or deleting
it (or editing/deleting its pack) does not cause startup to restore the original.
An existing character with the same name is preserved and the seed is skipped.
Each user's expressions are copied into their private Visual Identity storage.
PostgreSQL is unchanged: the existing Shared Visual Identity repository supports
SQLite only.

Artwork provenance is recorded beside both bundled resource directories as
`PROVENANCE.md` with `LicenseRef-User-Supplied`; no original artist name was
provided. Code licensing remains unchanged.

Ownership and rollback decision: `backlog/decisions/003-pixel-migu-bundled-character-seeding.md`.
