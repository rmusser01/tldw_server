import fs from 'node:fs';
import cp from 'node:child_process';
import crypto from 'node:crypto';
const root='/Users/macbook-dev/Documents/GitHub/tldw_server2';
const ui=JSON.parse(fs.readFileSync('/private/tmp/cycle4-uat108-persistence-paths.json','utf8'));
const py=['tldw_Server_API/app/core/Chat/chat_service.py','tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py','tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py'];
const task='backlog/tasks/task-13260.49 - Exclude-failed-Chat-display-placeholders-from-retry-model-context.md';
const code=[...py,...ui];
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const timestamp=new Date().toISOString();
const entries=[...code,task].map(path=>{const b=fs.readFileSync(root+'/'+path);const old=cp.spawnSync('git',['show','8097d672d5:'+path],{cwd:root,maxBuffer:8*1024*1024});return {path,kind:path===task?'task':path.includes('/tests/')||path.includes('/__tests__/')?'test':'production',bytes:b.length,sha256:digest(b),baselineSha256:old.status===0?digest(old.stdout):null};});
const manifest={task:'TASK-13260.49',frozenAt:timestamp,sourceBaseline:'8097d672d5',checkpointHead:cp.execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),production:8,tests:7,entries};
fs.writeFileSync('/private/tmp/cycle4-uat108-persistence-owned-manifest.json',JSON.stringify(manifest,null,2)+'\n');
fs.writeFileSync('/private/tmp/cycle4-uat108-persistence-production-manifest.json',JSON.stringify({...manifest,entries:entries.filter(x=>x.kind==='production')},null,2)+'\n');
const diff=cp.spawnSync('git',['diff','--check','--',...code,task],{cwd:root,encoding:'utf8'});
fs.writeFileSync('/private/tmp/cycle4-uat108-persistence-diff-check.txt',`Exit: ${diff.status}\n${diff.stdout}${diff.stderr}`);
if(diff.status!==0)throw Error('Owned diff-check failed');
const report=`# UAT108 retry persistence — implementation ready for independent review

Task: TASK-13260.49. Source baseline: 8097d672d5. Frozen: ${timestamp}. Checkpoint HEAD: ${manifest.checkpointHead}.

## Root cause and bounded correction

The normal saved Chat service persists the user before invoking the provider. An early provider HTTP502 therefore leaves an unanswered user row, with no saved assistant error envelope. Existing overlap handling recognized a saved error envelope, so a correct Retry request containing the intended user once still appended a second canonical user. Native evidence is retained under output/playwright/cycle4-repair-verification-2026-09-16/native-multi/retry-canonical-reload-108.txt and retry-reloaded-108.txt.

The existing regeneration path now marks a decoded assistant display-error retry explicitly, through pageAssistModel → ChatTldw → TldwChat, in app request metadata only: tldw_retry_failed_turn:true. Ordinary sends and regeneration of legitimate answers do not opt in. No schema validation, auth, provider inventory or provider-message format was relaxed.

The server reads the actual owned conversation tail independently of the requested history window/order. An explicit retry reuses the canonical ID of an exactly matching unanswered user (raw text plus substantive image content), or the user preceding a recognized legacy error envelope. A conflicting unanswered tail or already answered matching turn returns409 before invoking the provider. A failure before user persistence can still create its intended user. Ordinary repeated text remains a distinct user turn. Image detail:auto is normalized because the request schema supplies that default; different substantive images remain a conflict. The reused user is projected once into model history even when it falls outside the loaded context window. Both streaming and nonstreaming success acknowledgements identify the original user.

The pipeline resolves the existing user ID only on a decoded failed-turn retry. The existing save-message contract carries an optional retryFailedTurn flag, false by default; both success and error helper ACK gates require true, so ordinary successful-answer regeneration retains its former behavior. Its confirmed canonical ACK updates the visible row and existing local user row under the existing scoped transaction; it does not insert a second local user or replace draft fields. The local helper matches history, user role and unchanged content, rejects a contradictory existing ACK, and preserves timestamp, images and metadata. Newer text edits remain unacknowledged. Existing captured-owner, cancellation and A→B→A controls remain active.

## Final verification

- Backend: **76 passed**, six existing warnings. /private/tmp/cycle4-uat108-persistence-backend-broader.log.
- Frontend: **113 passed / 10 files** after independent-review correction. /private/tmp/cycle4-uat108-persistence-corrected-final.log.
- Bandit, touched production Python: **0 findings / 0 errors**. /private/tmp/cycle4-uat108-bandit.json.
- Explicit repository-root frontend ESLint: **0 errors / 35 unchanged warnings**, 12 paths. /private/tmp/cycle4-uat108-persistence-lint-comparison.json.
- Ruff, three touched Python paths: **15 baseline / 15 current findings, 0 added / 0 removed**. /private/tmp/cycle4-uat108-ruff-comparison.json.
- Owned diff check: clean. /private/tmp/cycle4-uat108-persistence-diff-check.txt.

Counts overlap earlier focused runs and must not be added together. Root completed the final combined TypeScript checkpoint: exit2 with exactly90 existing diagnostics and no added/removed signatures or multiplicity changes. Evidence: /private/tmp/cycle4-followup-combined-typecheck-final.log and /private/tmp/cycle4-followup-combined-typecheck-final-comparison.json. This is baseline parity, not a clean typecheck. No real model, browser, runtime restart, staging or commit was performed by this repair unit.

## Independent review correction

The first freeze exposed a P2: successful-answer regeneration received a new ordinary-send ACK and repointed the original local user. Reviewer actual pipeline probe /private/tmp/cycle4-uat108-successful-regeneration-probe.test.tsx reproduced it. Permanent controls at /private/tmp/cycle4-uat108-regeneration-gate-actual-red.log reproduced **3 failures / 1 positive** (actual successful regeneration and both-helper omitted/false opt-in). After the bounded failed-only gate, the targeted run passed5, final suite passed113/10, and the unchanged original reviewer probe passed1 (/private/tmp/cycle4-uat108-successful-regeneration-original-green.log). The backend was unchanged and not redundantly rerun. Only the existing SaveMessageBase declaration path was added. Whitespace-only alignment of forwarded property lines followed the passing run; the final parent-requested correction aligns the single retryFailedTurn property near line947 from eight spaces to six. A whitespace-stripped before/after comparison verified no behavioral source change. Tests were not redundantly repeated for formatting; independent review remains valid.

## Meaningful RED and correction trail

- /private/tmp/cycle4-uat108-persistence-backend-final-red.log: **3 failed / 2 passed** on real endpoint/DB failure→retry controls before production correction (two transport duplicate-user failures and conflicting-tail acceptance).
- /private/tmp/cycle4-uat108-persistence-ui-red.log: **3 failed / 65 passed** before intent plumbing at actual action/model/request boundaries.
- /private/tmp/cycle4-uat108-retry-mirror-red.log: **2 failed / 54 passed** before existing-local-user ACK correction.
- /private/tmp/cycle4-uat108-invoke-red.log: **1 failed / 10 passed** discovered missing nonstream option forwarding; corrected and covered by final113.
- /private/tmp/cycle4-uat108-newer-draft-red.log: **1 failed / 5 passed** before unchanged-content ACK protection; corrected and covered by final113.
- /private/tmp/cycle4-uat108-valid-envelope-actual-red.log and matching baseline config: final actual-owner test replay against unchanged8097d672d5 production gives **1 expected failure / 47 unselected tests**, missing [false,true] retry intent. The final valid-i18n test is green in the final113 run.

Harness qualifications: the first backend RED log included two request-object JSON serialization fixture errors, corrected before final3RED. Additional controls initially included invalid history_limit:0, which existing strict request validation rejected; the invalid fixture was removed without relaxing validation. The same additional run found meaningful schema-default-image and legacy-error-ACK failures, both fixed. One intermediate frontend failure came from uninitialized test i18n producing an invalid display envelope; its fixture now returns the same fallback strings used by initialized production i18n. One initial review-control command ran from the wrong working directory and found no tests; it is not the actual RED. The first two optional baseline-replay config attempts failed at startup and are not product RED evidence. The final replay above ran the actual test. Standard Node localStorage warnings and pytest teardown cleanup warning remain visible in logs.

## Exact commands

Backend, repository root:

\`\`\`sh
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py -q
python -m bandit tldw_Server_API/app/core/Chat/chat_service.py -f json -o /private/tmp/cycle4-uat108-bandit.json
\`\`\`

Frontend, apps/packages/ui:

\`\`\`sh
./node_modules/.bin/vitest run \\
  src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx \\
  src/hooks/chat-helper/__tests__/saveMessageOnSuccess.scope.test.ts \\
  src/models/__tests__/pageAssistModel.mcp-tools.test.ts \\
  src/services/tldw/__tests__/TldwChat.abort.test.ts \\
  src/hooks/chat-modes/__tests__/chatModePipeline.conversation-id.test.ts \\
  src/db/dexie/__tests__/helpers.user-acknowledgement.test.ts \\
  src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts \\
  src/models/__tests__/ChatTldw.stream-metadata.test.ts \\
  src/models/__tests__/ChatTldw.abort-signal.test.ts \\
  src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts \\
  --maxWorkers=1 --no-file-parallelism
\`\`\`

Lint reproduction, repository root: node /private/tmp/cycle4-uat108-persistence-static-check.mjs (uses the explicit frontend ESLint config and records baseline/current signatures). Frontend path inventory: /private/tmp/cycle4-uat108-persistence-paths.json. Ruff baseline/current/signature comparison JSON is retained with the same prefix.

## Owned files and freeze

${entries.map(e=>'- '+e.kind+': '+e.path).join('\n')}

Exact hashes and baseline hashes: /private/tmp/cycle4-uat108-persistence-owned-manifest.json. Production-only subset: /private/tmp/cycle4-uat108-persistence-production-manifest.json. Eight production files, seven tests and one official task record; unrelated working-tree changes are excluded.

## Limits and review focus

This is a forward retry correction. It does not delete or silently coalesce historical duplicate rows. Legacy callers without explicit intent retain their existing behavior. This is not a general request-idempotency protocol or a concurrency transaction redesign. Permanent endpoint tests use the real route and database with a mocked provider; no inference is claimed. The local Dexie helper test exercises the actual modify callback through a controlled adapter, not native browser IndexedDB. The actual action harness likewise uses its existing scoped DB adapter. Native fresh failed-provider→Retry→canonical reload remains parent-owned and required before marking the UAT acceptance complete.

Review should check failed-turn-only activation, ordinary repeat/regeneration behavior, exact tail matching and legacy compatibility, streaming/nonstream ACK, unchanged local draft preservation, and the existing captured-owner/cancel guards. No success is claimed for stale unrelated retry intents beyond the documented canonical-tail validation or for cleanup of already damaged conversations.
`;
fs.writeFileSync('/private/tmp/cycle4-uat108-retry-persistence-implementation.md',report);
console.log(JSON.stringify({frozenAt:timestamp,production:8,tests:7,task:1,diffCheck:diff.status,report:'/private/tmp/cycle4-uat108-retry-persistence-implementation.md',manifest:'/private/tmp/cycle4-uat108-persistence-owned-manifest.json'},null,2));
