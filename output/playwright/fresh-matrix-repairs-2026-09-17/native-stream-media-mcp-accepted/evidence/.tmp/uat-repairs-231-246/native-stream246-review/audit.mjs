import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

// Offline, read-only evidence projection; writes only this directory's audit.json.
const outputDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(outputDir, '../../..');
const base = '.tmp/uat-repairs-231-246';
const native = `${base}/native-targeted`;
const packet = 'output/playwright/fresh-matrix-repairs-2026-09-17/character-stream-buffering-reviewed';
const evidence = new Map();
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
function read(relative) {
  const bytes = fs.readFileSync(path.resolve(root, relative));
  evidence.set(relative, { path: relative, bytes: bytes.length, sha256: sha(bytes) });
  return bytes.toString('utf8');
}
function nativeFile(cell, name) {
  const text = read(`${native}/${cell}/${name}`);
  const line = text.split('\n').find(line => line.startsWith('{'));
  return { text, data: line ? JSON.parse(line) : null };
}
const checks = [];
const check = (name, pass) => checks.push({ name, passed: Boolean(pass) });
const pathname = e => e.url ? new URL(e.url).pathname : e.path;
const stable = value => JSON.stringify(value, (_, v) => v && !Array.isArray(v) && typeof v === 'object'
  ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => a.localeCompare(b))) : v);
const compactMessage = m => ({ id: m.id, parent: m.parent_message_id, sender: m.sender,
  timestamp: m.timestamp, contentLength: m.content.length, contentSha256: sha(m.content) });
const inputs = [
  { cell: 'pg-single', id: '5ceb72a9-61f1-42af-bc7a-434cb8c7052b', character: 3,
    old: 'fdea51b6-6f24-462e-b505-7d4a5495b61b', user: '65ee4abb-c9b9-4e45-8dd2-cca141253e3d', assistant: 'pa_d31a-2759-0f8-e609' },
  { cell: 'pg-multi', id: '335f57df-78c2-4831-b7de-5275f376bd69', character: 4,
    old: 'd817d54a-cc28-456b-b773-8340db5495ad', user: '87cae18f-debc-42e1-98c7-fefde2e32d04', assistant: 'pa_73d1-eb28-e30-4fbc' }
];
const results = [];
const priorAttempts = [];
for (const c of inputs) {
  const prefix = name => `${c.cell}: ${name}`;
  const send = nativeFile(c.cell, 'testbot-fresh-exact-send.txt').data;
  const terminal = nativeFile(c.cell, 'testbot-fresh-exact-terminal.txt').data;
  const reload = nativeFile(c.cell, 'testbot-fresh-exact-reloaded.txt').data;
  const initial = nativeFile(c.cell, 'testbot-created-entry.txt').data;
  const originalTerminal = nativeFile(c.cell, 'testbot-terminal.txt').data;
  for (const n of ['testbot-upgraded-library.txt', 'testbot-upgraded-library-settled.txt',
    'testbot-fresh-entry.txt', 'testbot-create.js', 'testbot-upgraded-exact-send.txt',
    'testbot-upgraded-exact-terminal.txt']) nativeFile(c.cell, n);
  if (c.cell === 'pg-single') nativeFile(c.cell, 'testbot-fresh-entry-settled.txt');
  else nativeFile(c.cell, 'testbot-fresh-entry.js');
  const oldReload = nativeFile(c.cell, 'testbot-upgraded-exact-reloaded.txt').data;
  const req = terminal.timing.find(e => e.event === 'request' && e.path === `/api/v1/chats/${c.id}/complete-v2`);
  const timing = terminal.timing.filter(e => e.requestId === req.requestId);
  const header = timing.find(e => e.event === 'headers');
  const bytes = timing.filter(e => e.event === 'body-bytes');
  const end = timing.at(-1);
  const firstDelayMs = (bytes[0].timestamp - header.timestamp) * 1000;
  // Header observer intentionally omits request identifiers. Correlate unique header
  // timestamp with the selected CDP request and confirm no competing stream exists.
  const headerMatches = terminal.headers.filter(e => Math.abs(Date.parse(e.at) - Date.parse(header.observedAt)) <= 5);
  const responseHeaders = headerMatches[0];
  const selected = terminal.events.filter(e => pathname(e)?.startsWith(`/api/v1/chats/${c.id}`));
  const requests = selected.filter(e => e.event === 'request' && e.method === 'POST');
  const userPost = requests.find(e => pathname(e).endsWith('/messages'));
  const complete = requests.find(e => pathname(e).endsWith('/complete-v2'));
  const persist = requests.find(e => pathname(e).endsWith('/completions/persist'));
  const persistAck = selected.find(e => e.event === 'response' && pathname(e).endsWith('/completions/persist'));
  const creation = terminal.events.find(e => e.status === 201 && e.body?.id === c.id);
  const pages = reload.events.filter(e => e.status === 200 && pathname(e) === `/api/v1/chats/${c.id}/messages` && e.body?.messages);
  const messages = pages.at(-1).body.messages;
  const originalCard = initial.events.find(e => e.event === 'response' && e.body?.id === c.character && e.body?.system_prompt)?.body;
  const currentCard = terminal.events.findLast(e => e.event === 'response' && pathname(e) === `/api/v1/characters/${c.character}`)?.body;
  const cardFields = ['id', 'version', 'name', 'system_prompt', 'first_message', 'personality', 'scenario',
    'post_history_instructions', 'message_example', 'alternate_greetings', 'character_version', 'updated_at'];
  const cardProjection = card => Object.fromEntries(cardFields.map(k => [k, card[k]]));
  const originalCompletion = originalTerminal.events.find(e => e.method === 'POST' && pathname(e).endsWith('/complete-v2'));
  check(prefix('fresh UI sends exact original prompt from an empty draft'), send.previousDraft.length === 0 && userPost.body.content === 'Hello, who are you?');
  check(prefix('fresh conversation is created empty with original character'), creation.body.message_count === 0 && creation.body.character_id === c.character && c.id !== c.old);
  check(prefix('one user write, one provider completion and one persistence dispatch'), requests.length === 3 && userPost && complete && persist);
  check(prefix('no rewrite or deletion of previous failed conversation during fresh turn'), !terminal.events.some(e => e.at >= send.start && ['POST','PUT','PATCH','DELETE'].includes(e.method) && pathname(e)?.startsWith(`/api/v1/chats/${c.old}`)));
  check(prefix('original card instructions and version remain unchanged'), stable(cardProjection(originalCard)) === stable(cardProjection(currentCard)) && currentCard.system_prompt === 'You are E2E-TestBot. Always respond with exactly: BEEP BOOP.');
  check(prefix('request settings match original native TestBot completion exactly'), stable(originalCompletion.body) === stable(complete.body));
  check(prefix('native SSE response is 200, unencoded and no-transform'), headerMatches.length === 1 && responseHeaders.status === 200 && responseHeaders.contentEncoding === null && responseHeaders.contentType.startsWith('text/event-stream') && responseHeaders.cacheControl === 'no-cache, no-transform');
  check(prefix('first body arrives under 100ms and before terminal'), firstDelayMs >= 0 && firstDelayMs < 100 && bytes[0].timestamp < end.timestamp);
  check(prefix('multiple nonempty body events precede completion'), bytes.length > 2 && bytes.every(e => e.dataLength > 0));
  check(prefix('settled UI shows exact visible BEEP BOOP reply and no timeout'), terminal.settled === true && terminal.ui.includes('paragraph: BEEP BOOP.') && !terminal.ui.includes('button "Stop Streaming"') && !terminal.ui.includes('Your chat timed out') && !terminal.ui.includes('No final answer'));
  check(prefix('canonical persistence acknowledgement is successful and IDs correlate'), persistAck.status === 200 && persistAck.body.saved === true && persist.body.user_message_id === c.user && persist.body.assistant_message_id === c.assistant);
  check(prefix('reload begins after verified terminal UI and successful persistence'), reload.start > terminal.at && reload.start > persistAck.at);
  check(prefix('fresh post-reload canonical pages consistently contain exactly two records'), pages.length > 0 && pages.every(e => e.at >= reload.start && e.body.total === 2 && e.body.has_more === false && stable(e.body.messages.map(compactMessage)) === stable(messages.map(compactMessage))));
  check(prefix('canonical acknowledged user and assistant retain parent link'), messages[0].id === c.user && messages[0].content === 'Hello, who are you?' && messages[1].id === c.assistant && messages[1].parent_message_id === c.user);
  check(prefix('reload displays two articles with exact acceptance marker'), reload.articles.length === 2 && reload.articles[0].includes('Hello, who are you?') && reload.articles[1].includes('BEEP BOOP.') && !reload.articles[1].includes('No final answer'));
  if (c.cell === 'pg-multi') check(prefix('multi-user native turn identifies Alice'), terminal.events.some(e => e.at >= send.start && e.identity?.id === 2 && e.identity.username === 'alice'));
  const oldRequest = terminal.timing.find(e => e.event === 'request' && e.path.includes(c.old) && e.observedAt >= '2026-09-17T21:47:00.000Z');
  const oldTiming = terminal.timing.filter(e => e.requestId === oldRequest.requestId);
  const oldPages = oldReload.events.filter(e => e.body?.messages && pathname(e) === `/api/v1/chats/${c.old}/messages`);
  const oldTail = oldPages.at(-1).body.messages.at(-1);
  check(prefix('prior existing-history no-final-answer outcome is preserved'), oldReload.articles.at(-1).includes('No final answer') && !oldReload.articles.at(-1).includes('BEEP BOOP.'));
  priorAttempts.push({ cell: c.cell, conversationId: c.old, requestId: oldRequest.requestId,
    requestAt: oldRequest.observedAt, endAt: oldTiming.at(-1).observedAt,
    durationMs: (oldTiming.at(-1).timestamp - oldRequest.timestamp) * 1000,
    canonicalTail: compactMessage(oldTail), reloadAt: oldReload.at,
    outcome: 'reasoning retained, no final answer; precise generation cause unestablished' });
  results.push({ cell: c.cell, conversationId: c.id, characterId: c.character, requestId: req.requestId,
    requestAt: req.observedAt, headersAt: header.observedAt, responseHeaders, headerCorrelation: 'unique observer timestamp within 5ms of selected CDP response',
    firstObservedByteAt: bytes[0].observedAt, firstByteDelayMs: Number(firstDelayMs.toFixed(3)),
    observedDecodedBytes: bytes.reduce((sum, e) => sum + e.dataLength, 0), bodyEventCount: bytes.length,
    streamEnd: { at: end.observedAt, event: end.event, canceled: end.canceled, errorText: end.errorText },
    persistenceAcknowledgedAt: persistAck.at, terminalUICapturedAt: terminal.at, reloadStartedAt: reload.start,
    reloadCapturedAt: reload.at, canonicalPages: pages.length, canonicalMessages: messages.map(compactMessage),
    model: complete.body.model, provider: complete.body.provider, cardVersion: currentCard.version,
    cardInstructionHash: sha(currentCard.system_prompt), originalRequestSettingsSha256: sha(stable(originalCompletion.body)) });
}
check('fresh single completes and reloads before fresh multi request', results[0].reloadCapturedAt < results[1].requestAt);
check('earlier single actually ended before multi began; no overlap attribution', priorAttempts[0].endAt < priorAttempts[1].requestAt);

const wrongSend = nativeFile('pg-single', 'testbot-upgraded-send.txt').data;
const wrongTerminal = nativeFile('pg-single', 'testbot-upgraded-terminal.txt').data;
const wrongUser = wrongTerminal.events.find(e => e.at >= wrongSend.start && e.method === 'POST' && pathname(e).endsWith('/messages') && e.body?.role === 'user');
check('earlier collapsed handoff actual 1971-character request is excluded from exact scenario', wrongUser.body.content.length === 1971 && wrongUser.body.content !== 'Hello, who are you?');
nativeFile('pg-single', 'testbot-upgraded-reloaded.txt');
nativeFile('pg-multi', 'testbot-upgraded-exact-terminal-v2.txt');
const correctedNotes = read(`${native}/UPGRADED_NATIVE_NOTES.md`);
check('current notes explicitly supersede incorrect overlap and reload-cause claim', correctedNotes.includes('initial controller claim') && correctedNotes.includes('incorrect') && correctedNotes.includes('21:47:59.038'));
read('backlog/tasks/task-13260.188 - Diagnose-native-TestBot-stream-timeout-in-fresh-PostgreSQL-multi-user-UAT.md');
const instrumentation = ['observe-character-timing-runtime.js', 'observe-upgraded-stream.js', 'settle-upgraded-stream.js'].map(p => read(`${native}/${p}`));
check('reviewed instrumentation is passive and does not route or fulfill responses', instrumentation.every(s => !/\.route\(|\.fulfill\(|\.abort\(/.test(s)) && instrumentation[0].includes("session.send('Network.enable')"));

const provenance = JSON.parse(read(`${base}/upgrade254-native-review/audit.json`));
read(`${base}/upgrade254-native-review/REVIEW.md`);
check('independent corrected runtime provenance is clear for both source copies', provenance.verdict === 'CLEAR' && provenance.candidate === 'a7d3155a567afb25982eb360ea24b973cc3249c9' && provenance.results.length === 2 && provenance.problems.length === 0);
read(`${packet}/review246-buffering/REVIEW246.md`);
read(`${packet}/stream246-native-audit/REPORT.md`);
read(`${packet}/manifest.json`);
const sourceAudit = JSON.parse(read(`${packet}/review246-buffering/final-hash-audit.json`));
const sourceHashes = sourceAudit.files.map(file => ({ path: file.path, expected: file.sha256,
  copies: ['.', ...provenance.results.map(r => path.relative(root, r.sourceRoot))].map(prefix => {
    const p = path.join(prefix, file.path); const actual = sha(read(p));
    return { path: p, sha256: actual, matches: actual === file.sha256 };
  }) }));
check('all three reviewed repair files match workspace and both corrected runtime copies', sourceHashes.every(f => f.copies.every(c => c.matches)));
const transport = read('apps/packages/ui/src/services/background-proxy.ts');
check('stream parser returns at DONE and cancels reader in finally', transport.includes('if (data === "[DONE]")') && transport.includes('await reader.cancel()'));
const endpoint = read('tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py');
check('endpoint keeps no-transform header map on exactly three response branches', endpoint.includes('"Cache-Control": "no-cache, no-transform"') && (endpoint.match(/headers=sse_headers/g) ?? []).length === 3);

read(path.relative(root, fileURLToPath(import.meta.url)));
if (fs.existsSync(path.join(outputDir, 'REVIEW.md'))) read(path.relative(root, path.join(outputDir, 'REVIEW.md')));
const failed = checks.filter(c => !c.passed);
const audit = { at: new Date().toISOString(), task: 'TASK-13260.188', finding: 'UAT246',
  verdict: failed.length ? 'REMAINING_GAP' : 'CLEAR', scope: 'Fresh native exact TestBot scenario and repaired browser byte delivery in PG single/multi only',
  candidate: provenance.candidate, checks, results, priorAttempts,
  excludedWrongPrompt: { at: wrongUser.at, actualUserContentLength: wrongUser.body.content.length, contentSha256: sha(wrongUser.body.content) },
  sourceHashes, limits: [
    'The original 45-second failure attribution remains unproven. Controlled installed-Next tests establish compression buffering; native repaired byte flow does not retrospectively prove the original cause.',
    'CDP measures browser receipt, not provider generation timestamps. Header observer records no request ID; unique timestamps correlate it to the selected CDP request.',
    'Terminal ERR_ABORTED alone is neither failure nor success; normal parser reader cancellation is compatible with it, while visible answer and canonical persistence establish acceptance.',
    'Canonical assistant content includes retained reasoning; exact BEEP BOOP refers to the visible final answer, not equality of the entire stored content field.',
    'Both prior existing-history exact attempts ended without a final answer. Their precise cause remains unestablished; successful fresh first turns do not erase them.',
    'The earlier claim of request overlap/reload-caused cancellation was superseded by complete CDP timestamps; the earlier single ended before the multi request and reload.',
    'Profiles and databases were reused through the independently reviewed corrected source upgrade. This is not fresh initialization or full 48-cell matrix acceptance.',
    'No tests, runtime, browser, provider, source, Git, tracker, or database mutations were performed. Prior 43-test implementation review is supporting retained evidence, not a new replay.'
  ], reviewedFiles: [...evidence.values()].sort((a,b) => a.path.localeCompare(b.path)) };
fs.writeFileSync(path.join(outputDir, 'audit.json'), `${JSON.stringify(audit,null,2)}\n`);
console.log(JSON.stringify({ verdict: audit.verdict, passed: checks.length-failed.length, failed, reviewedFiles:evidence.size,
  auditSha256:sha(fs.readFileSync(path.join(outputDir,'audit.json'))) },null,2));
if(failed.length) process.exitCode=1;
