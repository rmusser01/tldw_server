# Targeted multi-user native controls —111,121,117 PASS

Source checkpoint e7bf2184d3. Root restarted preserved isolated API18301 PID76485 and Next18381 PID76326. Same Alice account/profile; no reset, config edit, product edit, seeded response, mock, or direct database mutation. Natural login refreshed the long-idle session. This is targeted repair verification, not a full fresh UAT.

##111: retained-error Note backlink

Selected the preserved ordinary conversation c244a779-afcd-4f3d-8d89-a63065971a9b from visible Recent conversations. Confirmed actual UI has system, one original user, retained local provider-error bubble and recovered CEDAR RECOVERED answer. Navigated to Notes and opened existing Note1eea11bb-c31d-448e-a83d-74004f7d540e using its actual list entry. More actions → Open conversation navigated from /notes to /chat. Canonical GET349 returned200 and exact original three server rows:

- System8f69af6a-30fb-47a5-a46d-b4d677842eb5.
- User e69bfae5-f386-4b8e-8616-eeeefdcbdd06.
- Answer1b4b6005-f5fb-439a-9a2b-2bf3e5b826f5.

The historical local error remains visible, but no longer blocks navigation. Actual retained-error scenario passes. Automated text/image draft guards remain separate coverage; no native draft mutation performed. Existing note and chat records preserved.

Artifacts: retained-error.txt, note-menu.txt, backlink111.txt, backlink-canonical.txt, backlink111.png. Screenshot visually inspected. All filenames use this report prefix.

##121: actual current-chat controls reach complete-v2

Created a genuinely new saved Cedar Character chat through Characters → Chat as Cycle4 Cedar Guide. Visible Current Chat Model Settings initially blank. Set Max Tokens16, Temperature0, Top P0.9 and Repeat Penalty1 and clicked Save. Exact synthetic input:

    Cycle4 bounded reasoning117 20260916_0651: carefully reason through a three-step plan for Project Cedar before giving the final answer.

One Send at approximately06:50:54UTC. Actual complete-v2 request522 returned200 and contains max_tokens16, temperature0, top_p0.9, repetition_penalty1, stream:true. Model is the existing configured Gemma path. Body retained without headers in actual-request121.txt. No backend configuration change. The neutral repetition1 is supported and may normalize downstream; this verifies its requested field, not a claim about its numerical provider effect.

##117: newly completed actual reasoning-only negative

New conversation416f935c-4848-4725-9d3f-29cd6348bf34. This first bounded actual completion produced only reasoning, not a final answer. The UI explicitly reports “No final answer was generated. Retry or continue to request an answer.” Retry same model, Switch model, provider fallback and Continue from partial are offered. Expanded reasoning is readable. The UI preserves it through the normal assistant persistence request523, which returned201 with content:

    <think>*   User Input: "Cycle4 bounded reasoning117</think>

Normal browser reload returned canonical messages GET793200 with exactly three rows, no duplicate user or assistant:

- Greeting d832e4ee-d8a4-4f2a-b872-c899ab6c3337.
- User81dfd2b8-1964-4014-9637-72518ca74ff7.
- Reasoning assistant fd744aba-1098-4e87-84ee-553bf88fb52b.

The same missing-answer guidance and readable reasoning remain after reload; reloaded-reasoning117.png visually inspected. No Retry/Continue click or second generation was needed. This now covers an actual newly completed reasoning-only case; it does not claim every provider's stream shape. Exact stream body was not separately captured; request, UI, persisted assistant body and canonical reload provide the evidence.

The sole9099 lease was released immediately after terminal completion was observed at~06:51UTC, before reload checks. All four controls were cleared to their original blank state through the visible dialog and Save before reload. Current-chat settings intentionally have in-memory lifetime. No inference after release.

## Startup observations and limits

Long-idle startup auth/me returned401; normal Alice sign-in restored API access. Direct /chat?chatId=c244 initially showed the previously active276 Character transcript even after reauth. Selecting the actual visible history row loaded c244 correctly. No request or generation was made from the mismatched view. Retained startup evidence is an observation, not a repaired or accepted route behavior. Also the ordinary recovered Chat screenshot retains a Character mode/Choose character banner from the preceding Character context; no ordinary send was attempted in that state. Parent notified; no broader route exploration.

An initial harness navigation used an incorrect conversation query name and was replaced with the observed chatId route while first compilation was pending, producing ERR_ABORTED for that initial navigation. No product failure is inferred from this harness mistake.

Final reload console capture has0 errors and0 warnings. Native data impact is only the new416f conversation and three canonical rows; prior accounts, conversations and Note unchanged. Runtime/process lifecycle belongs to root. Credential scan and exact artifact hashes accompany this report.
