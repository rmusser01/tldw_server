# Model settings close lifecycle — TASK-13666

Actual Save updates the settings but loses focus because its parent removes the editor before AntD finishes closing. Keep controlled `open=false` immediate so editor reads and writes retire normally. Keep the opened editor present until its own Modal/Drawer close-completion event, then unmount it and restore its valid opener.

Use one model-settings-specific state hook in all three owners: full Form, compact `useComposerEvents`, and Layout. It retains presence through closure and captures an opening/closing revision so an old callback cannot unmount a reopened editor. No general overlay framework or fixed delays are needed. Shared focus-return behavior remains unchanged.

The full Form clears the settings scope on the matching close completion. Every default settings opener also initializes scope and old focus provenance before reopening; the explicit external event retains its requested scope. Its scheduled focus must retain the original draft/history/account/model selection and parent lifetime, and recheck the same close revision inside the scheduled callback. Save, Cancel, Escape, reopen and source-retirement paths retain existing dirty/pending/save protections.

ADR assessment:no new durable rule or protocol. This repairs the existing controlled-dialog and source-ownership contract, using installed AntD close events. Separate TASK-13667 follows the human decision to disable unsupported tracked-persona choices; it introduces no native persona authority.
