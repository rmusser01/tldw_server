# UAT091 targeted header repair

Follow-up to [the before-fix Notes screenshot](../notes-home-round3/notes-desktop-before-header-fix.png). Same isolated multi-user profile, existing fourth synthetic note; no content mutation or fresh-install claim.

At1280×720, the entire heading is218px wide, Saved21mago remains on one17px-high line, and actions use their own row. At1024×720, compact buttons wrap within530px. At390×844, the mobile44px action row and full note title/status remain visible. Document width equals viewport width in all3checks. All3after PNGs were visually inspected.

Root code scope: NotesEditorHeader, NotesSaveStatus, existing touch-layout expectation. Before RED1/6; after14tests/2suitespass. ESLint0errors2unchangedwarnings. See13260.18 and the tracker for independent review and commit status. Full fresh single/multi acceptance remains pending.
