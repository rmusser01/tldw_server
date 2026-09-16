# UAT112 / TASK13260.52 — Flashcard action names

Bounded independent read-only diagnosis. No source/test/browser/runtime edits. Three inspected source/test paths match frozen `7c9409`; see `/private/tmp/uat112-frozen-source-comparison.json`.

## Confirmed boundaries

1. `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx:2882–2894` wraps an icon-only Ant Button in a Tooltip but gives the button no text, aria-label, or title. Tooltip copy is a description, not a reliable accessible name for the unfocused button. Native `/private/tmp/uat-cycle4-single-create-another-stuck.json` records exactly an unnamed BUTTON. A private mounted actual ManageTab + real Ant control probe reproduces its empty accessible name.
2. `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx:425–443` delegates both Create action loading icons to Ant and has no explicit accessible action label or busy attribute. Native `/private/tmp/uat-cycle4-single-create-button-semantics.json` captures an enabled Create button with a retained leave-animation span at width0/opacity0/scale0. Its inner Ant icon is role=img aria-label=loading, so its accessible name remains `loading Create` even though the button is no longer pending. The native evidence explicitly shows enabled Create actions; this is not evidence of a stuck create request.

Installed `apps/packages/ui/node_modules/antd/es/button/DefaultLoadingIcon.js` uses CSSMotion/removeOnLeave and a LoadingOutlined icon during the leave animation. This explains where the retained accessible icon originates; the capture does not prove why that animation node remained for minutes. Do not expand this diagnosis into a global animation or dependency change.

## Private probes

`/private/tmp/uat112-accessible-name.config.ts` inserts only private cases into the existing ManageTab.empty-state test fixture, leaving product code unchanged. It also replays the exact retained native button outerHTML to test accessible-name calculation.

Command from `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run --config /private/tmp/uat112-accessible-name.config.ts --testNamePattern UAT112 --maxWorkers=1 --no-file-parallelism
```

`/private/tmp/uat112-accessible-name-red.log`: **2 expected RED, 3 unrelated tests filtered**.

- Actual mounted ManageTab FAB: expected `Create card`, actual empty accessible name.
- Captured native completed-button DOM replay: enabled and width0/opacity0 controls pass; expected `Create`, actual `loading Create` fails.

The second probe certifies the semantics of the retained DOM, not the animation lifecycle, network completion, or a live browser rerun.

## Minimal repair contract

Give the FAB a translated descriptive aria-label. Give the two create actions stable translated accessible labels independent of the decorative spinner; expose actual pending state separately with aria-busy and existing disabled/loading behavior. If a custom spinner is used, hide its decorative icon from assistive naming. Keep handler/state semantics unchanged and do not report completed controls as pending merely because the exit icon remains in DOM.

Permanent controls should query the real buttons by role/name before, during, and after held create resolution; ensure busy/disabled matches the mutation and exact names remain stable through icon exit. Native keyboard and accessibility-tree follow-up remains required. No general Flashcards audit or full-UAT acceptance is claimed.
