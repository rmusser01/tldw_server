## Stage 1: Regression Coverage

**Goal**: Capture the expected fullscreen focus-mode behavior before implementation.
**Success Criteria**: Tests fail because focus mode does not request shell chrome hiding and still renders the top cockpit controls instead of a dedicated exit control.
**Tests**: Targeted Vitest coverage for shared layout shell overrides and Playground focus mode.
**Status**: Complete

## Stage 2: Shared Shell Override Hook

**Goal**: Reuse the existing OptionLayout shell override channel from route content without rendering a nested layout shell.
**Success Criteria**: Focus-mode content can request `hideHeader` and `hideSidebar` across WebUI and extension shells.
**Tests**: Shared layout shell override tests pass.
**Status**: Complete

## Stage 3: Focus Mode UX

**Goal**: Make focus mode hide non-chat chrome and provide a single clear escape hatch.
**Success Criteria**: Focus mode shows chat transcript, composer, and an `Exit focus` control only; clicking the control returns to cockpit mode.
**Tests**: Playground cockpit-control tests pass.
**Status**: Complete

## Stage 4: Verification And PR Prep

**Goal**: Validate the focused UI in tests and browser, then prepare the PR update.
**Success Criteria**: Targeted tests pass, browser screenshot shows fullscreen focus mode, task notes are updated, and changes are committed/pushed.
**Tests**: Targeted Vitest command plus visual browser check.
**Status**: Complete
