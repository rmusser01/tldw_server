# Correlated native UAT108 recheck

Repair4bd4e2dfda; preserved multi-user Alice runtime. Parent intentionally restarted API18301 PID22744 and frontend18381 PID22787. Initial browser restart-error cleared by normal navigation after readiness; no reset/config/product edits.

| Control | Result |
|---|---|
| New saved ordinary Chat | PASS: c244a779-afcd-4f3d-8d89-a63065971a9b |
| Actual unavailable Ollama502 | PASS negative: one user, correlation pa_eeb3-27ac-b5b-513c |
| Pre-Retry canonical open + ordinary reload | PASS: system + one canonical user e69bfae5-f386-4b8e-8616-eeeefdcbdd06, matching correlation; UI oneuser |
| Actual configured Gemma Retry | PASS:200, one outbound user, same correlation + explicit failed-turn intent, no displayerrorJSON |
| Final canonical ordinary reload | PASS: exactly system + sameuser + assistant1b4b6005-f5fb-439a-9a2b-2bf3e5b826f5, CEDAR RECOVERED |
| Local error retention | Observation: prior local error reappears afterreload;4UIvs3canonical, no duplicateuser |
| Save successful answer to Notes | PASS:1eea11bb-c31d-448e-a83d-74004f7d540e, correct source conversation/message metadata |
| Note Open conversation | FAIL: two explicit menu clicks close menu but remain/notes, no newtab, no console error |

Gemma lease released after Retry completed; parent subsequently regranted for separate117check. No more108generation. Previous failed conversations untouched. Not a full fresh UAT/signoff; no image Retry claim.
