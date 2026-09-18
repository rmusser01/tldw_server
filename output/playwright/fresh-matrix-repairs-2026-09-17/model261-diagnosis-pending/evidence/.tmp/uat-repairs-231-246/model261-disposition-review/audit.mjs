// Read-only, hash-only disposition audit for TASK13260.203 / UAT261.
// It never serializes provider stream bodies, reasoning, or credentials.
import crypto from 'node:crypto';
import fs from 'node:fs';

const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const read = file => fs.readFileSync(file);
const text = file => read(file).toString();
const check = (name, pass) => checks.push({ name, pass: Boolean(pass) });
const checks = [];
const inputs = [];
const tracked = (path, { privateHashOnly = false } = {}) => {
  const bytes = read(path);
  inputs.push({ path, bytes: bytes.length, sha256: sha(bytes), privateHashOnly });
  return bytes.toString();
};

const workflowPath = 'apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts';
const endpointPath = 'tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py';
const presetPath = 'tldw_Server_API/app/core/Character_Chat/modules/character_prompt_presets.py';
const emotePath = 'tldw_Server_API/app/core/Character_Chat/emote_directives.py';
const taskPath = 'backlog/tasks/task-13260.203 - Diagnose-TestBot-exact-response-deviation-after-valid-model-selection-UAT-261.md';
const streamReviewPath = '.tmp/uat-repairs-231-246/native-stream246-review/REVIEW.md';
const freshReviewPath = '.tmp/uat-repairs-231-246/native236-fresh-review/REVIEW.md';
const freshAuditPath = '.tmp/uat-repairs-231-246/native236-fresh-review/audit.json';
const repeatSendPath = '.tmp/uat-repairs-231-246/native-targeted/pg-single/testbot261-repeat-send.js';
const retryProjectionPath = '.tmp/uat-repairs-231-246/native-targeted/pg-single/testbot261-retry-branch-projection.json';
const staleProjectionPath = '.tmp/uat-repairs-231-246/native-targeted/pg-single/testbot261-retry-reloaded-projection.json';

const workflow = tracked(workflowPath);
const endpoint = tracked(endpointPath);
const presets = tracked(presetPath);
const emotes = tracked(emotePath);
const task = tracked(taskPath);
const streamReview = tracked(streamReviewPath);
const freshReview = tracked(freshReviewPath);
const freshAudit = JSON.parse(tracked(freshAuditPath));
const repeatSend = tracked(repeatSendPath);
const retryProjection = JSON.parse(tracked(retryProjectionPath));
const staleProjection = JSON.parse(tracked(staleProjectionPath));

check('Maintained journey creates a timestamp-named character',
  workflow.includes('const characterName = `E2E-TestBot-${Date.now()}`'));
check('Maintained journey sends a distinct description field',
  workflow.includes('description: "E2E test character for journey spec"'));
check('Maintained journey exercises library selection and complete-v2 streaming',
  workflow.includes('selectCharacter(characterName)')
  && workflow.includes('sendMessage("Hello, who are you?")')
  && workflow.includes('include_character_context: true')
  && workflow.includes('stream: true'));
check('Maintained journey only requires successful completion status, not exact canonical answer',
  workflow.includes('expect(completeStatus).toBeLessThan(300)')
  && !/expect\s*\([^)]*BEEP BOOP|to(?:Be|Contain|Equal)\s*\([^)]*BEEP BOOP/.test(workflow));
check('Tagged UAT261 helper opens existing decorated card and requires a fresh conversation',
  repeatSend.includes('Chat as E2E-TestBot — Repair PGsingle 20260917')
  && repeatSend.includes("Expected fresh character conversation")
  && repeatSend.includes('Hello, who are you?'));
check('Default prompt builder makes character name a prompt input before character fields',
  presets.includes('f"You are {resolved_char}."')
  && presets.includes('_safe_replace(character.get("system_prompt")'));
check('Complete-v2 builds character context then appends the generic emote instruction',
  endpoint.includes('sys_text = _build_system_prompt_for_preset(')
  && endpoint.includes('sys_text = append_character_emote_prompt_instruction(sys_text, character)')
  && emotes.includes('When the character expression should change'));
check('Retained source record explicitly withholds outbound provider prompt/messages',
  task.includes('Original outbound provider prompt/messages absent'));
check('UAT261 record retains both original deviation and later no-final-answer outcome',
  task.includes('Original extra-introduction')
  && task.includes('No final answer was generated'));
check('Same tagged-card sequence contains an exact retry child final without exposing reasoning',
  retryProjection.status === 200
  && retryProjection.rows?.length === 2
  && retryProjection.rows?.[1]?.final === 'BEEP BOOP.'
  && typeof retryProjection.rows?.[1]?.sha256 === 'string'
  && !Object.hasOwn(retryProjection.rows?.[1] ?? {}, 'content'));
check('Original stale parent projection is distinct and records no final answer',
  staleProjection.status === 200
  && staleProjection.uiHasNoFinalError === true
  && staleProjection.rows?.[1]?.final === ''
  && staleProjection.rows?.[1]?.chars > 0);
check('Independent UAT246 review records exact fresh turns using the existing TestBot card in both PostgreSQL modes',
  streamReview.includes('existing TestBot card')
  && streamReview.includes('Both native requests send exactly')
  && streamReview.includes('BEEP BOOP.'));
check('Fresh UAT236 review explicitly scopes its static E2E-TestBot as distinct from tagged UAT261',
  freshReview.includes('E2E-TestBot')
  && freshReview.includes('differs from the earlier tagged UAT261 character')
  && freshAudit.summary?.character?.name === 'E2E-TestBot'
  && freshAudit.summary?.canonical?.exactFinalBEEP === true);

const result = {
  task: 'TASK13260.203 UAT261 read-only disposition review',
  verdict: checks.every(entry => entry.pass) ? 'DISPOSITION: no application defect established; task remains open pending a causal boundary' : 'GAPS',
  checks,
  counts: { checks: checks.length, inputs: inputs.length },
  inputHashes: inputs.map(({ path, bytes, sha256, privateHashOnly }) => ({ path, bytes, sha256, privateHashOnly })),
  findings: {
    workflowAndTaggedFixtureDifferMaterially: true,
    materialDifferences: [
      'workflow creates a timestamp-named character; tagged helper opens a pre-existing decorated-name card',
      'workflow supplies a description field; retained tagged-card fields are not a byte-identical workflow creation receipt',
      'workflow has status/recovery assertions but no exact canonical-final assertion',
    ],
    decoratedNamePromptRelevant: true,
    decoratedNameCausalityEstablished: false,
    reason: 'The default builder incorporates the name, but retained UAT246 exact fresh turns used the existing TestBot card and UAT261 retry child on the tagged card reached the exact final.',
    applicationPromptDefectEstablished: false,
    reasonPrompt: 'The retained artifacts do not contain the outbound provider messages or a prompt fingerprint for the failed turn; source augmentation is therefore not comparable at the required request boundary.',
  },
  justifiedNextEvidence: [
    'Do not perform blind provider retries or tune the model.',
    'If UAT261 is advanced, collect one paired, same-tagged-card/same-provider request-boundary record that hashes the assembled message sequence and normalized request settings before provider dispatch, then pair it with canonical final-only metadata. Retain no raw prompt, provider reasoning, or credentials.',
    'Use the current UAT261 exact-final requirement in the acceptance harness: canonical final after think-block removal, not the maintained workflow’s present status-only completion assertion.',
    'A product edit is unjustified unless the paired records show an application-controlled prompt/request difference or a deterministic source-level invariant that conflicts with the character instruction.',
  ],
  limitations: [
    'No browser, provider, database, runtime, source, test, Git, Backlog, or tracker action was performed.',
    'The audit reads only source, safe projections, and metadata; it does not emit raw provider output, reasoning, prompt bodies, or credentials.',
    'Fresh UAT236 establishes a separate static-card success and cannot erase UAT261 failures.',
  ],
};
const out = '.tmp/uat-repairs-231-246/model261-disposition-review/audit.json';
fs.writeFileSync(out, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ verdict: result.verdict, checks: checks.length, failed: checks.filter(c => !c.pass).map(c => c.name), inputs: inputs.length }));
