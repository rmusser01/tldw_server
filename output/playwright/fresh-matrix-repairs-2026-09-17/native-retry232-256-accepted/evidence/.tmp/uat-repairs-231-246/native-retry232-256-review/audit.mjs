import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const evidence = path.join(root, '.tmp/uat-repairs-231-246/native-targeted/pg-single');
const preparation = path.join(root, '.tmp/uat-repairs-231-246/native-upgrade-preparation');
const output = path.join(root, '.tmp/uat-repairs-231-246/native-retry232-256-review');
const conversationId = 'cb561345-2d7f-43eb-93e2-ef60d5d3e07f';
const failedClientId = 'pa_556d-145c-831-c040';
const newUserId = '2ca9137f-71f3-43c8-99c8-076960e22a81';
const newAssistantId = 'b647981f-69d1-45e1-b1ad-8af35bc5c5c6';
const revision = '2787043410fc918b2c280d90f753d8fd02b6b35b';

const checks = [];
const inputs = [];
const sha256 = (value) => crypto.createHash('sha256').update(value).digest('hex');
function check(name, pass) { checks.push({ name, pass: Boolean(pass) }); }
function read(relative) {
  const target = path.join(evidence, relative);
  const bytes = fs.readFileSync(target);
  inputs.push({ path: path.relative(root, target), bytes: bytes.length, sha256: sha256(bytes) });
  return bytes.toString('utf8');
}
function result(relative) {
  const text = read(relative);
  const match = text.match(/### Result\n([\s\S]*?)(?:\n### |$)/);
  if (!match) throw new Error(`Missing structured result: ${relative}`);
  return JSON.parse(match[1]);
}
function parseSseTerminalFrames(body) {
  if (typeof body !== 'string') return [];
  return body.split('\n').flatMap((line) => {
    if (!line.startsWith('data:')) return [];
    const payload = line.slice('data:'.length).trim();
    if (payload === '[DONE]') return [{ type: 'done' }];
    try {
      const frame = JSON.parse(payload);
      return frame?.type === 'error' ? [{ type: 'error' }] : [];
    } catch {
      return [];
    }
  });
}

const started = result('retry256-started.txt');
const observed = result('retry256-observed.txt');
const canonical = result('retry256-canonical-reload.txt');
const retryScript = read('retry256-retry.js');
const reloadScript = read('retry256-canonical-reload.js');
const chatOpen = read('retry256-chat-open.txt');
const chatSettled = read('retry256-chat-settled.txt');
const startupPath = path.join(preparation, 'retry256-startup-safe.json');
const startupBytes = fs.readFileSync(startupPath);
const startup = JSON.parse(startupBytes);
inputs.push({ path: path.relative(root, startupPath), bytes: startupBytes.length, sha256: sha256(startupBytes) });

const completionResponses = observed.events.filter((event) =>
  event.event === 'response' && /\/chat\/completions$/.test(event.url || '')
);
const terminalFrames = completionResponses.flatMap((event) => parseSseTerminalFrames(event.body));
const messageResponses = canonical.events.filter((event) =>
  event.event === 'response' && event.status === 200 && Array.isArray(event.body?.messages)
);
const latestMessages = messageResponses.at(-1)?.body?.messages;
const ids = Array.isArray(latestMessages) ? latestMessages.map((message) => message?.id) : [];
const senders = Array.isArray(latestMessages) ? latestMessages.map((message) => message?.sender) : [];
const newUser = Array.isArray(latestMessages) ? latestMessages.find((message) => message?.id === newUserId) : null;
const newAssistant = Array.isArray(latestMessages) ? latestMessages.find((message) => message?.id === newAssistantId) : null;
const chatResponses = canonical.events.filter((event) =>
  event.event === 'response' && event.status === 200 && event.body?.id === conversationId
);
const chatRecord = chatResponses.at(-1)?.body;
const processRoles = startup.processes.map((process) => {
  const label = `${process.cell || ''} ${process.action || ''}`.toLowerCase();
  const database = label.includes('pg-single') ? 'pg-single' : label.includes('pg-multi') ? 'pg-multi' : 'other';
  const component = label.includes('backend') ? 'backend' : label.includes('frontend') ? 'frontend' : 'other';
  return `${database}:${component}`;
});
const retryRequest = started.events.find((event) =>
  event.event === 'request' && /\/chat\/completions$/.test(event.url || '')
);
const retryRequestBody = retryRequest?.body;

check('Retry capture includes one chat-completion request', Boolean(retryRequest));
check('Retry request is bound to the original conversation', retryRequestBody?.conversation_id === conversationId);
check('Retry request preserves the failed client message identity', JSON.stringify(retryRequestBody).includes(failedClientId));
check('Retry observed a successful completion response', completionResponses.length === 1 && completionResponses[0].status === 200);
check('Retry stream contains exactly one terminal frame with no error frame', terminalFrames.length === 1 && terminalFrames[0].type === 'done');
check('Canonical reload has exactly seven messages', Array.isArray(latestMessages) && latestMessages.length === 7);
check('Canonical reload contains the new user and assistant rows', ids.includes(newUserId) && ids.includes(newAssistantId));
check('Canonical rows preserve user then assistant ordering', ids.indexOf(newUserId) < ids.indexOf(newAssistantId) && senders[ids.indexOf(newUserId)] === 'user' && senders[ids.indexOf(newAssistantId)] === 'assistant');
check('Canonical chat record reports seven messages', chatRecord?.message_count === 7);
check('Startup receipt has the approved revision', startup.revision === revision);
check('Startup receipt covers backend and frontend for both immutable PostgreSQL upgrades', processRoles.length === 4 && new Set(processRoles).size === 4 && processRoles.every((role) => /^(pg-single|pg-multi):(backend|frontend)$/.test(role)));
check('Every upgrade process is sourced from the approved immutable revision', startup.processes.every((process) => process.sourceCommit === revision && typeof process.sourceRoot === 'string' && process.sourceRoot.includes('retry256-worldbook255-upgrade-20260917')));
check('Retained UI captures are present for the pre-retry and settled states', chatOpen.length > 0 && chatSettled.length > 0);
check('Retry and reload helper artifacts are retained with the evidence', retryScript.length > 0 && reloadScript.length > 0);

const audit = {
  issue: 'UAT232/256 independent native retry review',
  status: checks.every((entry) => entry.pass) ? 'pass' : 'fail',
  checks,
  summary: {
    originalConversationVerified: retryRequestBody?.conversation_id === conversationId,
    failedClientIdentityVerified: JSON.stringify(retryRequestBody).includes(failedClientId),
    completionStatus: completionResponses.map((event) => event.status),
    terminalFrameTypes: terminalFrames.map((frame) => frame.type),
    canonicalMessageCount: Array.isArray(latestMessages) ? latestMessages.length : null,
    canonicalNewRowOrder: ids.indexOf(newUserId) < ids.indexOf(newAssistantId),
    startupProcessRoles: processRoles,
    sourceRevisionParity: startup.processes.every((process) => process.sourceCommit === revision),
  },
  limits: [
    'This review uses retained UI/API evidence and immutable startup receipts only.',
    'No raw stream frames, provider reasoning, private configuration, runtime logs, or message contents are emitted.',
    'The textual UI captures are retained and hashed, but this safe audit does not emit or independently classify their visible control labels.',
    'The review confirms the single retained retry path and canonical reload; it does not establish all provider, cancellation, or multi-user retry behavior.',
  ],
  inputs,
};
fs.writeFileSync(path.join(output, 'audit.json'), JSON.stringify(audit, null, 2) + '\n');
console.log(JSON.stringify({ status: audit.status, passed: checks.filter((entry) => entry.pass).length, failed: checks.filter((entry) => !entry.pass).map((entry) => entry.name), inputCount: inputs.length }));
if (audit.status !== 'pass') process.exitCode = 1;
