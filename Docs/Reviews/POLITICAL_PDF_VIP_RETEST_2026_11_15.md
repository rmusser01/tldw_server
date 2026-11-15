# Political PDF remediation and end-to-end retest

Task: TASK-15505. Status: implementation in progress; no post-fix acceptance pass yet. Baseline: [original UAT](LARGE_POLITICAL_PDF_UAT_2026_11_15.md), commit a1ec6cc5d1. [Approved remediation design](../Design/POLITICAL_PDF_VIP_FIXES_2026_11_15.md).

## Evaluation method

Agent-operated walkthrough and code/trace review, using [NN/g's ten usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/). Severity follows [NN/g's 0–4 scale](https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/), considering frequency, impact and persistence: 0 not a problem, 1 cosmetic, 2 minor, 3 major, 4 release-blocking for the evaluated task. Ratings below are provisional evaluator judgments, not participant measurements or NN/g certification. Independent review and actual retest outcomes will be recorded separately.

Heuristic key: H1 system status; H2 familiar language/concepts; H3 control/freedom; H4 consistency; H5 error prevention; H6 recognition over recall; H7 efficiency; H8 minimalism; H9 error diagnosis/recovery; H10 help.

## UX issue ledger

| ID / prior finding | Task and observed evidence | Heuristics | Severity and rationale | Owner | Fix / verification status |
|---|---|---|---|---|---|
| UX-PDF-01 / F1 | Request book summary; only first6,000characters sent, with no visible coverage contract | H1,H5 | 4: repeats on every large source; user can mistake front matter for complete analysis | .3 | Open; exact sections, coverage and complete batching planned |
| UX-PDF-02 / F2 | Upload succeeded but original-file endpoint404 | H1,H4,H9 | 3: blocks evidence review after apparent success; cannot recover original through app | .1 | Open; shared retention before staging cleanup |
| UX-PDF-03 / F3 | Citation preview opens beginning/offsets, no PDF page | H2,H6,H7 | 3: repeated manual searching makes verification impractical | .1/.4 | Open; page map, exact chunk evidence and viewer links |
| UX-PDF-04 / F4 | Selected PDF question asks what document is meant; UI reports search complete with0sources though retrieval never ran | H1,H4,H9 | 3: natural first action fails and suggests wrong recovery | .2 | Open; scope-aware default and distinct clarification outcome |
| UX-PDF-05 / F5 | Long retrieval, partial relevant evidence, question restored without answer | H1,H3,H7,H9 | 3: central task fails; delay/cancellation state unclear | .2 | Open; focused persisted retrieval and overall deadline |
| UX-PDF-06 / F6 | Full-source mode places question after huge document; query truncation can omit question (code-confirmed) | H2,H5 | 3: selected mode can defeat the user's question without a useful explanation | .2 | Open; separate retrieval question and bounded evidence context |
| UX-PDF-07 / F7 | Default800output tokens fail with DeepSeek; retries expose separate failures | H5,H9 | 3: default primary action fails and requires technical tuning | .3 | Open; reasoning-compatible defaults and stage-specific recovery |
| UX-PDF-08 / F8 | Source annotation survives refresh but absent from workspace export | H4,H5,H10 | 3: repeated evidence work fails to travel with export; exclusion undisclosed there | .4 | Open; validated annotation snapshot roundtrip and accurate copy |
| UX-PDF-09 / F8 | Existing Media outline is disconnected from summary/comparison scope; comparison requires two source files | H2,H6,H7 | 3: researcher cannot directly compare positions inside one book | .3 | Open; reviewed section selection and within-document comparison |
| UX-PDF-10 / new | Navigation estimates page→text ranges and falls back to whole source when unresolved (code-confirmed) | H1,H5 | 4: chapter coverage could be falsely claimed even after adding a chapter selector | .1/.3 | Open; exact core resolution and explicit unavailable status |
| UX-PDF-11 / mobile |390×844 status strips compressed; setup and technical controls crowd composer | H1,H8 | 2: task remains reachable but recognition/space use suffers | .4 | Pending post-fix desktop/phone review |

## Verification ledger

- Design and work tracking created before edits. Four implementation units assigned; original checkout preserved in managed worktree.
- New regression tests and actual results will be recorded here as each unit completes. Existing baseline tests are documented in the original UAT and do not constitute post-fix verification.
- Real PDF/DeepSeek retest, chapter coverage, citation opening, export/import, keyboard/phone checks and independent UX review remain pending.

## Acceptance results

No post-fix result claimed yet. Record real browser outcomes separately from controlled API diagnostics and automated tests. In particular, a downloaded ZIP is not an export/import pass, a saved artifact is not a complete-coverage pass, and model citations are not verified until opened against the PDF.
