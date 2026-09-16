import fs from 'node:fs';
import cp from 'node:child_process';
import crypto from 'node:crypto';
const root='/Users/macbook-dev/Documents/GitHub/tldw_server2';
const {ui,py}=JSON.parse(fs.readFileSync('/private/tmp/cycle4-uat108-correlation-paths.json'));
const task='backlog/tasks/task-13260.49 - Exclude-failed-Chat-display-placeholders-from-retry-model-context.md';
const sourceBaseline='7dcee3d72c', frozenAt=new Date().toISOString();
const digest=x=>crypto.createHash('sha256').update(x).digest('hex');
const entries=[...ui,...py,task].map(path=>{const bytes=fs.readFileSync(root+'/'+path);const old=cp.execFileSync('git',['show',sourceBaseline+':'+path],{cwd:root,maxBuffer:15e6});return {path,kind:path===task?'task':path.includes('/tests/')||path.includes('/__tests__/')?'test':'production',bytes:bytes.length,sha256:digest(bytes),baselineSha256:digest(old)};});
const manifest={task:'TASK-13260.49',sourceBaseline,frozenAt,checkpointHead:cp.execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),production:entries.filter(x=>x.kind==='production').length,tests:entries.filter(x=>x.kind==='test').length,entries};
for(const [suffix,value]of[['owned',manifest],['production',{...manifest,entries:entries.filter(x=>x.kind==='production')}]])fs.writeFileSync('/private/tmp/cycle4-uat108-correlation-'+suffix+'-manifest.json',JSON.stringify(value,null,2)+'\n');
const diff=cp.spawnSync('git',['diff','--check','--',...entries.map(x=>x.path)],{cwd:root,encoding:'utf8'});
fs.writeFileSync('/private/tmp/cycle4-uat108-correlation-diff-check.txt',`Exit: ${diff.status}\n${diff.stdout}${diff.stderr}`);if(diff.status!==0)throw Error('diff-check failed');
const report=`# UAT108: failed-user correlation through the mounted loader

Task: TASK-13260.49. Baseline ${sourceBaseline}; frozen ${frozenAt}. Exact byte manifests: cycle4-uat108-correlation-owned-manifest.json and production-manifest.json in /private/tmp. Eight production paths, seven test paths, one task record. This follow-up supersedes the native acceptance of the prior retry-persistence repair; it does not claim native GREEN.

## Evidence and cause

The preserved native initial request contains one user and returns502. The UI already contains two identical users before Retry. Retry sends two users; after Retry, the canonical conversation contains two user IDs and its final assistant. Native captures did not include a pre-Retry canonical GET, so no pre-Retry server row-count claim is made. Evidence: /private/tmp/cycle4-uat108-native-multi-final-report.md and its21-artifact manifest.

Source and the controlled mounted reproduction establish the missing interaction: the server saves the user before provider execution; the early502 returns no ACK. The real server loader then sees a canonical user while the local row is still unacknowledged. Previous mirror identity recovery required an acknowledged assistant anchor, which cannot exist yet. It preserves the unmatched local row, resulting in two visible users. Explicit Retry then submitted both; the old overlap rule appended another canonical user.

## Correction

The already-generated local user ID travels through the existing pipeline/model/client options as app-only metadata tldw_client_message_id. Only the final user actually being persisted receives a validated, bounded client_message_id in its DB metadata, in the existing atomic transaction. Provider messages and arguments do not receive it. Correlation is added after content-placeholder classification, preserving empty-content behavior.

The mirror links only a single exact correlation claim to an unacknowledged same-history user with identical role, content and substantive images. Existing canonical identity, foreign-history and captured-authority checks remain. Equal text with another local ID remains separate. No text-only deduplication or deletion was added. Normal/persona message loads use the existing render_placeholders=false option so literal stored text can match; tracked-character loads retain rendered placeholders. Both listing shapes retain correlation metadata.

Explicit Retry rejects residual unmatched extra users409 before any new write/provider call. If saved correlation exists, a contradictory local identity also rejects409. Missing/null/malformed legacy metadata remains compatible with the pre-existing exact-tail retry validation. Ordinary repeated sends and successful-answer regeneration remain distinct from failed Retry.

## Verification and RED trail

- Permanent mounted actual action + useServerChatLoader + real ChatTldw regression reproduced1 expected failure before repair: /private/tmp/cycle4-uat108-mounted-real-model-red.log. It now verifies initial failure, one linked local user before Retry, one outbound user, successful canonical answer, and remount retaining both IDs/content. Transport/provider and storage adapters are controlled; this is not native IndexedDB or real inference.
- Correlation model/transport/mirror boundary RED:3 failed/51 passed across3 files, cycle4-uat108-correlation-ui-red.log. Real endpoint/SQLite DB RED:3 failed/23 deselected, cycle4-uat108-correlation-backend-red.log.
- Raw normal/persona versus tracked-character compatibility RED:3 failed/7 skipped, cycle4-uat108-placeholder-read-red.log. Character's missing explicit query flag was contract scaffolding; its existing default rendered behavior was already correct. Tests now also assert the resulting content.
- Additional review controls reproduced a null-extra error (1 failed/7 passed, cycle4-uat108-correlation-boundary-actual.log) and unrelated empty-content placeholder regression (1 failed, cycle4-uat108-empty-content-red.log) before their bounded corrections.
- Author expanded UI:117 passed/5 files, cycle4-uat108-correlation-ui-expanded.log. Final changed owner fixture:2 passed/50 skipped, cycle4-uat108-correlation-owner-final.log. Final loader scope suite including two actual mapper attachment-safety controls:12 passed, cycle4-uat108-correlation-mapper-safety-final.log. Counts overlap and must not be summed.
- Author backend:87 passed/2 files, cycle4-uat108-correlation-backend-expanded.log; final added/affected endpoint/DB controls:7 passed/24 deselected, cycle4-uat108-correlation-extra-boundaries.log. These overlap; final reviewer runs all92 tests across the same two files. Tests exercise transactional metadata failure rollback, actual pre-provider502 persistence, both listing shapes, metadata decoding failure, and no-partial-write409.
- Root-config frontend ESLint:0 errors/22 unchanged warnings over11 frontend paths. cycle4-uat108-correlation-lint-{current,baseline,comparison}.json. The initial test-only unreachable-yield finding was corrected with a behavior-equivalent rejecting async generator, then affected cases/lint rerun.
- Ruff:18 existing findings/18 current,0 added/removed across4 Python paths. cycle4-uat108-correlation-ruff-comparison.json.
- Bandit:0 findings/0 errors across2 touched production Python paths. cycle4-uat108-correlation-bandit.json.
- Owned diff-check clean. Whole TypeScript and combined broader runs are parent-owned and not claimed here before completion.

## Exact commands

UI from apps/packages/ui:
\
./node_modules/.bin/vitest run src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx src/db/dexie/__tests__/server-chat-mirror.test.ts src/models/__tests__/pageAssistModel.mcp-tools.test.ts src/services/tldw/__tests__/TldwChat.abort.test.ts src/hooks/__tests__/useServerChatLoader.scope.test.tsx --maxWorkers=1 --no-file-parallelism

Backend from repository root:
\
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py -q

Static exact commands/paths are retained in /private/tmp/cycle4-uat108-correlation-lint.mjs and cycle4-uat108-correlation-paths.json; ESLint uses explicit repository-root apps/tldw-frontend/eslint.config.mjs. Bandit: source .venv/bin/activate && python -m bandit tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py -f json -o /private/tmp/cycle4-uat108-correlation-bandit.json.

## Limits and ownership

- Historically ambiguous local/server rows without the new marker are preserved. Existing acknowledged-reply recovery remains available; unresolved duplicate Retry fails409 rather than silently discarding work or adding another canonical user. No migration/deletion is attempted.
- The existing standard listing/client omits user attachment bytes and the loader maps ordinary images to[]. Image-only stored text can be a generated attachment placeholder. Therefore successful attachment recovery is NOT established by this text-workflow repair. The handcrafted mirror image case tests exact-equality safety only. Permanent actual mapper controls prove image-only/text+image local content and bytes remain unchanged and unacknowledged when identity cannot be safely established. This pre-existing limitation was explicitly scoped out by root; no weakened match was introduced.
- No browser/runtime/inference calls, staging, commit, global tracker or shared plan edits performed by this unit. Parent owns stable-runtime native recheck and full fresh UAT. Independent review is a separate report; final native108 is still pending.

## Owned paths

${entries.map(x=>'- '+x.path+' ('+x.kind+')').join('\n')}
`;
fs.writeFileSync('/private/tmp/cycle4-uat108-correlation-implementation.md',report.trimEnd()+'\n');console.log(JSON.stringify({frozenAt,production:manifest.production,tests:manifest.tests,entries:entries.length,diffCheck:diff.status}));
