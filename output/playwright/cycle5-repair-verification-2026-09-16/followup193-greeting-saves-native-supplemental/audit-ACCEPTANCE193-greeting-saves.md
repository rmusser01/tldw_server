# Independent native acceptance addendum — UAT193 greeting saves

## Disposition

**UAT193 / TASK13260.131 literal AC3 now passes; recommend closing UAT193 together with its already accepted AC1/AC2 source review and required-PostgreSQL evidence.** The prior accepted audit establishes fresh Alice character Chat creation and controlled failed-provider Retry followed by canonical reload. This addendum supplies the missing ordinary greeting Save Note and Save Flashcard actions in that same chat.

The exact AC3 is: “Native fresh character Chat creates its conversation, then controlled failed provider Retry and greeting save actions complete successfully.” Both new actions returned 201 and displayed their success messages. No new saved-resource GET or completed post-save Chat reload is claimed. Those are not extra requirements added to AC3.

This is UAT193 only. It does not close the broader Chat task, certify other acceptance criteria, prove a fresh full matrix, or erase separately tracked issues. AC1/AC2 were not rerun by this native-only audit; the official task records their reviewed repair and 247 combined independent tests with zero skips.

## Exact native chain

All times are UTC on 2026-09-17. The cumulative observer contains earlier Bob activity; this audit checks the most recent identity before **each** save, rather than treating the whole capture as one actor. The most recent identity is Alice, user 2 at 07:32:20.348, with no intervening actor switch before either save.

- Conversation: `471f52fa-22c5-4eba-a3d6-ee5e6982884f`.
- Original greeting message: `1e23f18b-4842-4799-931f-df2b2ea75448`.
- The canonical history response at 07:32:21.099 is 200 and contains that greeting exactly once. Its text is: “Hello! I'm your Helpful AI Assistant. How can I support you today?”
- Both action menus were opened from the actual first assistant message. The request snippet and Flashcard answer match that canonical greeting exactly. The operator filled only the question field for the card; the answer was already populated. No model substitution is used as save-action evidence.

| Action | Request / response | Verified result |
| --- | --- | --- |
| Save Note | POST `/api/v1/chat/knowledge/save` 07:33:39.200 → **201** 07:33:39.236 | `make_flashcard:false`; returned note `72e2e192-c099-4829-af4b-c935e8ecf993`; same conversation/greeting IDs; UI says **Saved to Notes**. |
| Save Flashcard | POST same endpoint 07:37:42.619 → **201** 07:37:42.662 | `make_flashcard:true`; returned note `029a74d1-1de1-4283-99ff-5a816f275fa7` and card `082ec63a-124e-4c47-ac23-c11a2deba70b`; same conversation/greeting IDs; UI says **Saved to Flashcards**, and review dialog closes. |

The card question is “What greeting did the Helpful AI Assistant use in this UAT conversation?” Its answer and snippet equal the greeting above. The returned IDs are server-confirmed save results; this audit does not infer unreturned database owner fields.

## Persistence and reload limits

The accepted `ACCEPTANCE193-212.md` already proves the Retry completion's persisted assistant and the same ordered three canonical message IDs after the earlier full reload. Its 18 original native inputs and two task snapshots still match their retained hashes. That earlier successful reload remains the reload evidence supporting the combined UAT193 chain.

The new `alice193-after-saves-reload.txt` records an actual `page.reload()`. The new events file is captured at 07:40:22.307 during startup. It contains fresh successful Alice 2 identity receipts at 07:40:21.857 and07:40:22.106, but no completed post-reload Chat history response. The accompanying shell snapshot likewise does not establish a settled conversation. It must not be reported as a successful post-save history reload.

No GET of the new note72e2e192, companion note029a74d1 or card082ec63a is present in these supplied files. Therefore no independent saved-content readback or resource reopening claim is made here. Extra canonical card/Note checks can be retained separately without changing this task's literal criterion.

## Evidence and reproducibility

`greeting-input-manifest.json` binds all 15 new native input files, the prior accepted audit/manifest and a fresh official CLI snapshot of TASK13260.131. No original native bytes were normalized or changed. `audit_greeting_saves.py` reads those files only and verifies:

1. Prior accepted input hashes remain unchanged.
2. Two exact save requests/responses, status 201 and same original conversation/greeting IDs.
3. Alice 2 identity immediately preceding each save.
4. Request snippets and prefilled card answer match canonical greeting content.
5. Correct returned resource IDs, success text and closed card review dialog.
6. Fresh reload identity and the explicit absence of new resource/history readbacks.

The check passed; reduced, non-secret facts are in `greeting-audit-facts.json` and `greeting-audit-check.log`. No authorization/session material, model reasoning text or private helper was copied into the report. No browser, runtime, database, production/test source change, task mutation or git action was performed. Runtime source attribution remains the parent-owned frozen backend 47e23/PID 56113 receipt documented by the earlier audit; these new UI captures do not independently establish a process startup manifest.
