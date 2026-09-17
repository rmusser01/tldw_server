import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

// Offline evidence audit only: no browser, network, runtime, DB, or Git access.
const outputDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(outputDir, '../../..');
const base = '.tmp/uat-repairs-231-246';
const native = `${base}/native-targeted/pg-multi`;
const reviewed = new Map();
const sha256 = value => crypto.createHash('sha256').update(value).digest('hex');
function read(relative) {
  const bytes = fs.readFileSync(path.resolve(root, relative));
  reviewed.set(relative, { path: relative, bytes: bytes.length, sha256: sha256(bytes) });
  return bytes.toString('utf8');
}
function result(name) {
  const text = read(`${native}/${name}`);
  const line = text.split('\n').find(line => line.startsWith('{'));
  return { text, data: line ? JSON.parse(line) : null };
}
const checks = [];
function check(name, passed) { checks.push({ name, passed: Boolean(passed) }); }
const names = [
  'account-boundary-prepared.txt', 'account-boundary-logout.txt',
  'account-boundary-current.txt', 'account-boundary-logout-v2.txt',
  'account-boundary-bob-login.txt', 'account-boundary-auth-current.txt',
  'account-boundary-after-bob.txt', 'alice-cancel-history-readback.txt',
  'bob-ordinary-chat-open.txt', 'bob-chat-settled.txt', 'bob-new-saved-chat.txt',
  'bob-ordinary-send.txt', 'bob-ordinary-terminal.txt', 'bob-ordinary-reloaded.txt',
  'bob-upgrade-readback.txt', 'bob-upgrade-navigate.txt',
  'bob-upgrade-settled.txt', 'bob-upgrade-captured.txt',
  'upgrade-current.txt', 'upgrade-tabs.txt', 'upgrade-observer-installed.txt',
  'bob-upgrade-logout.txt', 'alice-upgrade-login.txt'
];
const files = Object.fromEntries(names.map(name => [name, result(name)]));
const data = name => files[name].data;
const text = name => files[name].text;
for (const name of ['account-boundary-prepare.js', 'account-boundary-send-logout.js',
  'account-boundary-send-logout-v2.js', 'bob-upgrade-capture.js']) read(`${native}/${name}`);
const aliceId = 'd817d54a-cc28-456b-b773-8340db5495ad';
const bobId = 'e3755001-1cd4-40a4-aaaf-edd4b67b3418';
const cancelledUserId = 'cbd509c8-39e8-4227-97cf-346306a80644';
const pathname = event => event.url ? new URL(event.url).pathname : event.path;
const messageResponses = packet => packet.events.filter(e => e.status === 200 && Array.isArray(e.body?.messages));
const compactMessage = message => ({
  id: message.id, parent: message.parent_message_id, sender: message.sender,
  timestamp: message.timestamp, contentLength: message.content.length,
  contentSha256: sha256(message.content)
});

const failedAttempt = data('account-boundary-current.txt');
check('first harness attempt actually completed and persisted before logout',
  failedAttempt.events.some(e => e.method === 'POST' && pathname(e).endsWith('/completions/persist')) &&
  failedAttempt.authEvents.length === 0);
check('failed harness receipts retained',
  text('account-boundary-prepared.txt').includes('strict mode violation') &&
  text('account-boundary-logout.txt').includes('TimeoutError') &&
  text('account-boundary-bob-login.txt').includes("reading 'filter'"));

const boundary = data('account-boundary-logout-v2.txt');
const afterBob = data('account-boundary-after-bob.txt');
const start = boundary.start;
const timings = afterBob.timing.filter(e => e.observedAt >= start);
const streamRequest = timings.find(e => e.event === 'request');
const requestTimings = timings.filter(e => e.requestId === streamRequest.requestId);
const headers = requestTimings.find(e => e.event === 'headers');
const aborted = requestTimings.find(e => e.event === 'failed');
const logout = afterBob.authEvents.find(e => e.path === '/auth/logout');
const login = afterBob.authEvents.find(e => e.path === '/auth/login');
const bodyBytes = requestTimings.filter(e => e.event === 'body-bytes').reduce((sum, e) => sum + e.dataLength, 0);
const userPost = boundary.events.find(e => e.method === 'POST' && pathname(e).endsWith('/messages'));
const userAck = boundary.events.find(e => e.event === 'response' && pathname(e).endsWith('/messages') && e.status === 201);
check('actual complete-v2 request belongs to Alice tracked conversation',
  streamRequest.path === `/api/v1/chats/${aliceId}/complete-v2` &&
  boundary.events.some(e => e.identity?.id === 2 && e.identity.username === 'alice'));
check('user acknowledgement precedes streaming headers', userAck.at < headers.observedAt);
check('normal logout follows headers and precedes cancellation',
  headers.status === 200 && logout.status === 200 &&
  headers.observedAt < logout.at && logout.at < aborted.observedAt);
check('CDP observed cancelled ERR_ABORTED with no body bytes for this request',
  aborted.canceled === true && aborted.errorText === 'net::ERR_ABORTED' && bodyBytes === 0);
check('original page becomes Signed out', boundary.chatAfterLogout.includes('heading "Signed out"'));
check('normal second-tab Bob login succeeds', login.status === 200 && login.at > logout.at &&
  afterBob.authEvents.some(e => e.path === '/auth/me' && e.status === 200 && e.identity?.id === 3 && e.identity.username === 'bob'));
const lateWrites = afterBob.events.filter(e => e.at > logout.at && e.method === 'POST' && pathname(e).startsWith(`/api/v1/chats/${aliceId}/`));
check('no later Alice message or completion write in retained observer interval', lateWrites.length === 0);

const aliceReadback = data('alice-cancel-history-readback.txt');
const alicePages = messageResponses(aliceReadback);
const aliceMessages = alicePages.at(-1).body.messages;
check('later readback authenticated as Alice and targets original conversation',
  aliceReadback.events.some(e => e.identity?.id === 2 && e.identity.username === 'alice') &&
  new URL(aliceReadback.url).searchParams.get('chatId') === aliceId);
check('all canonical Alice message readbacks contain five rows with no next page',
  alicePages.length > 0 && alicePages.every(e => e.body.total === 5 && e.body.messages.length === 5 && e.body.has_more === false));
check('canonical Alice rows contain initial pair, failed-harness pair, cancelled user only',
  aliceMessages[0].sender === 'user' && aliceMessages[1].parent_message_id === aliceMessages[0].id &&
  aliceMessages[2].sender === 'user' && aliceMessages[3].parent_message_id === aliceMessages[2].id &&
  aliceMessages[4].id === cancelledUserId && aliceMessages[4].sender === 'user' &&
  aliceMessages[4].content === userPost.requestBody.content);
check('cancelled user has no assistant successor in canonical readback',
  !aliceMessages.some(m => m.parent_message_id === cancelledUserId) &&
  !aliceMessages.some(m => m.sender !== 'user' && m.timestamp >= aliceMessages.at(-1).timestamp));
check('canonical conversation tail is cancelled user and count is five',
  aliceReadback.events.some(e => e.body?.id === aliceId && e.body.message_count === 5 && e.body.tail?.message_id === cancelledUserId));

const bobTerminal = data('bob-ordinary-terminal.txt');
const bobReload = data('bob-ordinary-reloaded.txt');
const bobUpgrade = data('bob-upgrade-captured.txt');
const ordinaryPost = bobTerminal.events.find(e => e.method === 'POST' && pathname(e) === '/api/v1/chat/completions');
check('Bob normal completion uses ordinary endpoint for new conversation',
  ordinaryPost.body.conversation_id === bobId &&
  bobTerminal.events.some(e => e.identity?.id === 3) &&
  bobTerminal.events.some(e => e.event === 'response' && pathname(e) === '/api/v1/chat/completions' && e.status === 200));
check('newly created ordinary conversation has no character',
  bobTerminal.events.some(e => e.body?.id === bobId && e.body.character_id === null));
check('settled ordinary completion shows answer and Standard chat without reload',
  bobTerminal.url === 'http://127.0.0.1:18783/chat' &&
  bobTerminal.body.includes('Standard chat') && bobTerminal.body.includes('BOB ORDINARY CHAT OK.') &&
  !bobTerminal.body.includes('Character Chat') && !bobTerminal.body.includes('Choose a character'));
check('early reload contradictory label retained and canonical entries identified as stale',
  bobReload.body.includes('Character Chat') && bobReload.body.includes('Choose a character') &&
  bobReload.canonical.every(e => e.at < '2026-09-17T21:20:00.000Z'));
const bobPages = messageResponses(bobUpgrade);
const bobMessages = bobPages.at(-1).body.messages;
check('upgraded normal navigation retains authenticated Bob identity',
  bobUpgrade.events.some(e => e.identity?.id === 3 && e.identity.username === 'bob') &&
  text('bob-upgrade-navigate.txt').includes('await page.goto('));
check('upgraded settled UI shows saved ordinary answer and Standard chat',
  bobUpgrade.ui.includes('Standard chat') && bobUpgrade.ui.includes('BOB ORDINARY CHAT OK.') &&
  !bobUpgrade.ui.includes('Character Chat') && !bobUpgrade.ui.includes('Choose a character') &&
  text('bob-upgrade-settled.txt').includes('Standard chat'));
check('all upgraded canonical Bob reads contain same three rows and no next page',
  bobPages.length > 0 && bobPages.every(e => e.body.total === 3 && e.body.has_more === false &&
    JSON.stringify(e.body.messages.map(compactMessage)) === JSON.stringify(bobMessages.map(compactMessage))));
check('canonical Bob rows are system, user and one exact assistant reply',
  bobMessages.map(m => m.sender).join(',') === 'system,user,assistant' &&
  bobMessages[2].content === 'BOB ORDINARY CHAT OK.' && new Set(bobMessages.map(m => m.id)).size === 3);
check('upgraded canonical Bob conversation remains ordinary',
  bobUpgrade.events.some(e => e.body?.id === bobId && e.body.character_id === null && e.body.source === 'webui-chat' && e.body.message_count === 3));
check('automation continuation began about blank and initial readback timed out',
  text('upgrade-current.txt').includes('about:blank') && text('upgrade-tabs.txt').includes('about:blank') &&
  text('bob-upgrade-readback.txt').includes('TimeoutError'));

for (const p of [
  'backlog/tasks/task-13260.185 - Align-newly-saved-ordinary-Chat-presentation-after-account-switch.md',
  'backlog/tasks/task-13260.190 - Retire-Character-stream-and-persistence-after-actual-account-invalidation.md',
  `${base}/review248/REVIEW.md`, `${base}/review248/REVIEW.json`,
  `${base}/review-chat/REVIEW.md`, `${base}/review-chat/REVIEW.json`,
  `${base}/review-chat/source-after.json`, `${base}/native-upgrade-review/REVIEW196.md`
]) read(p);
const reviewedSource = JSON.parse(read(`${base}/review248/source-after.json`));
const sourceRoots = [
  '.', '.tmp/uat-next-matrix-20260916/repair-sources/repairs231-250-targeted-20260917/pg-multi',
  '.tmp/uat-next-matrix-20260916/repair-sources/repairs251-253-upgrade-20260917/pg-multi',
  '.tmp/uat-next-matrix-20260916/repair-sources/repairs251-254-upgrade2-20260917/pg-multi'
];
const sourceComparisons = reviewedSource.files.filter(f => f.kind === 'production').map(file => ({
  path: file.path, reviewedSha256: file.sha256,
  copies: sourceRoots.map(sourceRoot => {
    const relative = path.join(sourceRoot, file.path);
    const actual = sha256(read(relative));
    return { path: relative, sha256: actual, matchesReviewed: actual === file.sha256 };
  })
}));
check('all three production files match independently reviewed bytes in all four source locations',
  sourceComparisons.every(file => file.copies.every(copy => copy.matchesReviewed)));
const hook = read('apps/packages/ui/src/hooks/chat/useChatActions.ts');
check('combined hook retains ordinary publication guard and account lease signal',
  hook.includes('useStoreMessageOption.getState().serverChatId !== payloadConversationId') &&
  hook.includes('const executionSignal = servicePromptSnapshot?.scopeSignal ?? signal'));
read(path.relative(root, fileURLToPath(import.meta.url)));
if (fs.existsSync(path.join(outputDir, 'REVIEW.md'))) read(path.relative(root, path.join(outputDir, 'REVIEW.md')));
const failed = checks.filter(c => !c.passed);
const audit = {
  generatedAt: new Date().toISOString(), scope: 'Offline PostgreSQL multi-user native acceptance review; UAT243 and UAT248 only',
  tasks: ['TASK-13260.185', 'TASK-13260.190'],
  sourceRevisionsSuppliedByController: {
    initial: '86458ab88ce3fa62e6518c9d813c3860254ddb2c',
    upgraded: 'a7d3155a567afb25982eb360ea24b973cc3249c9',
    note: 'Commit identities supplied by controller; reviewer verified relevant file bytes across retained copies without Git access.'
  },
  verdicts: { UAT243: failed.length ? 'REMAINING_GAP' : 'CLEAR', UAT248: failed.length ? 'REMAINING_GAP' : 'CLEAR' },
  checks,
  boundary: {
    conversationId: aliceId, requestId: streamRequest.requestId, userAckAt: userAck.at,
    canonicalUserTimestamp: aliceMessages.at(-1).timestamp, headersAt: headers.observedAt,
    logoutAt: logout.at, abortedAt: aborted.observedAt, observedBodyBytes: bodyBytes,
    millisecondsFromLogoutResponseToObservedAbort: Date.parse(aborted.observedAt) - Date.parse(logout.at),
    loginAt: login.at, observationEnd: afterBob.at, lateWriteRequestsObserved: lateWrites.length,
    canonicalReadbackAt: aliceReadback.at, messages: aliceMessages.map(compactMessage)
  },
  ordinaryChat: {
    conversationId: bobId, ordinaryRequestAt: ordinaryPost.at, preReloadSettledAt: bobTerminal.at,
    earlyReloadAt: bobReload.at, earlyReloadCanonicalLastAt: bobReload.canonical.at(-1).at,
    upgradedReadbackAt: bobUpgrade.at, canonicalResponseCount: bobPages.length,
    messages: bobMessages.map(compactMessage)
  },
  limitations: [
    'First wrong-path waiter timed out; that turn completed and persisted normally and is excluded from cancellation acceptance.',
    'Second-turn zero body bytes are CDP observations. This proves browser request retirement before delivery, not server rollback or post-logout provider work cessation.',
    'No late assistant was present at the canonical Alice readback approximately 32 minutes later. This is a bounded observation, not a claim over arbitrary future time.',
    'The 21:20 early reload snapshot shows Character Chat but retained only pre-reload canonical events. It does not establish settled reload acceptance or a proven transient root cause.',
    'Automation continuation began about:blank. Successful later evidence uses normal goto with surviving authentication and then a normal Bob logout/Alice login; it is not uninterrupted-page continuity.',
    'Native actions occurred on initial targeted source; later canonical reads used upgraded source with identical scoped frontend files. No fresh full 48-cell matrix acceptance is inferred.',
    'Prior independent implementation tests and security/lint receipts are context; this audit did not rerun tests or make a new compiler, Bandit, or full-source attestation.'
  ],
  sourceComparisons, reviewedFiles: [...reviewed.values()].sort((a, b) => a.path.localeCompare(b.path))
};
fs.writeFileSync(path.join(outputDir, 'audit.json'), `${JSON.stringify(audit, null, 2)}\n`);
console.log(JSON.stringify({ verdicts: audit.verdicts, passed: checks.length - failed.length, failed,
  reviewedFiles: reviewed.size, output: path.join(outputDir, 'audit.json'), auditSha256: sha256(fs.readFileSync(path.join(outputDir, 'audit.json'))) }, null, 2));
if (failed.length) process.exitCode = 1;
