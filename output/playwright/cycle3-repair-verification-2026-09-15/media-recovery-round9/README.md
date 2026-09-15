# Targeted Media recovery and Trash title checks

TASK13260.17 / TASK13260.20, integrated-dev repair branch, 2026-09-15. This is targeted verification, not a fresh full UAT pass.

## Observed passes

- Alice's actual Media Analysis version selector renders the original Markdown analysis as strong text and six list items, with all Cedar facts visible. The ordinary user's Delete control is disabled with a permission explanation.
- Actual Settings logout and admin login verified browser /auth/me identity1, cycle3_multi_admin. The only admin source is synthetic uat-cycle3-multi-bob-admin-source, media1 in the admin's own library.
- Actual Delete confirmation sent DELETE /media/1 and received204. After the development build interruption and rebuild, the settled empty library retained a visible Trash action after the Undo toast was gone.
- Actual Trash navigation showed the source and the correct deletion timestamp. Actual Restore completed with POST /media/1/restore200, as confirmed by the retained backend access lines. The first action capture ended early at Restoring and captured no response; the subsequent settled capture proves Trash empty.
- Independent admin GET200 before deletion and after restoration contains exactly equal content objects (281 text characters). Selecting the actual result shows the original source/sentinel. A normal reload retains the selected source and exact content. The restored-source and empty-library screenshots were visually inspected.
- UAT058 also affected the Trash wrapper. Existing real Next Head regression expanded by one route:1 failed/9 passed before the minimal wrapper correction;10 passed afterward. Scoped ESLint exits0 with empty output. Actual Media Trash click afterward yields Trash | tldw. Bandit is inapplicable to these TypeScript-only changes.

## Limits and interrupted observations

- The first delete/empty-state follow-up was interrupted by a Next development missing-export build overlay while an adjacent Chat module changed. DELETE already succeeded. No product success is inferred from the interrupted body; only the later settled captures count. Test data was not reinitialized.
- A first direct /media?id=1 navigation after restoration settled at /media with no selected item. Selecting the actual restored result and a subsequent normal reload passed. Retained restore-final-current records that initial observation; direct-entry stability needs the remaining source-link matrix, and is not certified here.
- Analysis/outer-page scrolling and leaving Media produced no progress PUT in the observation window. The short source did not establish an actual scroll of the content container. Live zoom/progress persistence remains unverified; no new progress bug is inferred from that check. A raw aria-ref analysis count in the first capture is an invalid DOM selector result, not absence of the rendered analysis.
- Admin optional storage/quota responses require email verification and returned403. No verification or permission policy was bypassed. Logout form warning occurred amid HMR; stable reproduction remains pending.
- No permanent deletion was performed. Undo behavior and all fresh single/multi workflow rows remain separate acceptance work.

Text logs normalize trailing whitespace; JSON and PNG bytes are retained. All entries were scanned for the14 known isolated runtime credential values and token/private-key patterns and are indexed by SHA256SUMS. Synthetic usernames and public fixture text are intentional.
