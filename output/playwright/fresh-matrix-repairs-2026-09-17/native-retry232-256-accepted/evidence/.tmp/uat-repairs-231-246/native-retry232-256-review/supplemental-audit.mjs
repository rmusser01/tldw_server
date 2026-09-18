import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const targeted = path.join(root, '.tmp/uat-repairs-231-246/native-targeted/pg-single');
const preparation = path.join(root, '.tmp/uat-repairs-231-246/native-upgrade-preparation');
const output = path.join(root, '.tmp/uat-repairs-231-246/native-retry232-256-review');
const revision = '2787043410fc918b2c280d90f753d8fd02b6b35b';
const servicePath = 'tldw_Server_API/app/core/Chat/chat_service.py';
const checks = [];
const inputs = [];

const sha256 = (value) => crypto.createHash('sha256').update(value).digest('hex');
const digest = (value) => sha256(Buffer.from(typeof value === 'string' ? value : JSON.stringify(value)));
const check = (name, pass) => checks.push({ name, pass: Boolean(pass) });
const capture = (file) => {
  const bytes = fs.readFileSync(file);
  inputs.push({ path: path.relative(root, file), bytes: bytes.length, sha256: sha256(bytes) });
  return bytes;
};
const json = (file) => JSON.parse(capture(file).toString('utf8'));
const result = (file) => {
  const text = capture(file).toString('utf8');
  const match = text.match(/### Result\n([\s\S]*?)(?:\n### |$)/);
  if (!match) throw new Error(`Missing structured result: ${path.basename(file)}`);
  return { text, value: JSON.parse(match[1]) };
};
const latestMessages = (captureResult) => captureResult.value.events
  .filter((event) => event.event === 'response' && event.status === 200 && Array.isArray(event.body?.messages))
  .at(-1)?.body.messages;
const messageProof = (messages, later) => messages.map((message) => {
  const replacement = later.find((candidate) => candidate.id === message.id);
  return {
    idSha256: digest(message.id),
    contentSha256: digest(message.content),
    recordSha256: digest(message),
    exactRecordMatch: JSON.stringify(message) === JSON.stringify(replacement),
  };
});
const selectedModelUnavailable = (text) => /selected\s+model[\s\S]{0,200}(?:unavailable|not\s+available)/i.test(text);

const modelError = result(path.join(targeted, 'model232-upgraded-error-captured.txt'));
const modelReload = result(path.join(targeted, 'model232-upgraded-reload.txt'));
const oldReload = result(path.join(targeted, 'model232-upgraded-retry-reloaded.txt'));
const retryReload = result(path.join(targeted, 'retry256-canonical-reload.txt'));
const startup = json(path.join(preparation, 'retry256-startup-safe.json'));

const oldMessages = latestMessages(oldReload);
const newMessages = latestMessages(retryReload);
if (!Array.isArray(oldMessages) || !Array.isArray(newMessages)) throw new Error('Missing canonical message projection');
const oldMessageProof = messageProof(oldMessages, newMessages);

check('Model232 capture displays selected-model unavailable guidance', selectedModelUnavailable(modelError.text));
check('Model232 error capture retains the retry-same-model action before the later retry', /retry\s+same\s+model/i.test(modelError.text));
check('Retry256 canonical reload shows the expected success markers', /ORBIT-742/.test(retryReload.text) && /Saved/.test(retryReload.text));
check('Retry256 canonical reload has no active selected-model-unavailable or retry control', !selectedModelUnavailable(retryReload.text) && !/retry\s+same\s+model/i.test(retryReload.text));
check('Model232 settled reload has no active unavailable guidance', !selectedModelUnavailable(modelReload.text));
check('Canonical recovery grows exactly from five rows to seven rows', oldMessages.length === 5 && newMessages.length === 7);
check('All five original canonical rows remain byte-identical after recovery', oldMessageProof.length === 5 && oldMessageProof.every((item) => item.exactRecordMatch));

const matrixChecks = [];
for (const cell of ['pg-single', 'pg-multi']) {
  const cellProcesses = startup.processes.filter((process) => `${process.cell || ''} ${process.action || ''}`.includes(cell));
  const sourceRoot = cellProcesses[0]?.sourceRoot;
  if (cellProcesses.length !== 2 || !sourceRoot) throw new Error(`Missing ${cell} startup receipts`);
  const run = path.basename(path.dirname(sourceRoot));
  const matrix = path.resolve(sourceRoot, '../../..');
  const bindingPath = path.join(matrix, 'targeted-upgrades', run, cell, 'binding.private.json');
  const bindingBytes = capture(bindingPath);
  const binding = JSON.parse(bindingBytes.toString('utf8'));
  const profile = json(binding.originalProfile);
  const initializationPath = path.join(profile.root, 'initialized.private.json');
  const initialization = json(initializationPath);
  const holder = json(profile.pgReceiptPath);
  const manifestPath = path.join(path.dirname(sourceRoot), 'preparation', `${cell}-source-manifest.json`);
  const manifestBytes = capture(manifestPath);
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  const sourceBytes = capture(path.join(sourceRoot, servicePath));
  const sourceHash = sha256(sourceBytes);
  const manifestEntry = manifest.files.find((entry) => entry.path === servicePath);
  const bindingHash = sha256(bindingBytes);
  const profileHash = sha256(fs.readFileSync(binding.originalProfile));
  const initializationHash = sha256(fs.readFileSync(initializationPath));
  const holderHash = sha256(fs.readFileSync(profile.pgReceiptPath));
  const receipts = cellProcesses.map((process) => json(process.receipt));
  const sourceBinding = receipts.every((receipt) =>
    receipt.bindingHash === bindingHash &&
    receipt.sourceRoot === binding.sourceRoot &&
    receipt.sourceCommit === revision &&
    receipt.status === 'started'
  );
  const originalRecords =
    profileHash === binding.originalProfileHash &&
    initializationHash === binding.originalInitializationHash &&
    holderHash === binding.originalHolderHash &&
    initialization.status === 'completed' &&
    initialization.preparationHash === digest(profile) &&
    holder.status === 'held' &&
    holder.source_root === profile.sourceRoot &&
    holder.source_commit === profile.sourceCommit;
  const immutableSource =
    binding.sourceCommit === revision &&
    binding.sourceRoot === sourceRoot &&
    manifest.revision === revision &&
    manifest.root === sourceRoot &&
    binding.proof?.sourceManifest === sha256(manifestBytes) &&
    manifestEntry?.sha256 === sourceHash;
  check(`${cell} startup receipts bind the actual immutable binding`, sourceBinding);
  check(`${cell} binding preserves profile initialization and holder identities`, originalRecords);
  check(`${cell} immutable chat service matches the revision-bound manifest`, immutableSource);
  matrixChecks.push({
    cell,
    bindingSha256: bindingHash,
    sourceSha256: sourceHash,
    sourceManifestSha256: sha256(manifestBytes),
    sourceManifestMatches: immutableSource,
    originalProfileMatches: profileHash === binding.originalProfileHash,
    originalInitializationMatches: initializationHash === binding.originalInitializationHash,
    originalHolderMatches: holderHash === binding.originalHolderHash,
    receiptBindingsMatch: sourceBinding,
  });
}
check('Both immutable copies have identical chat-service bytes', matrixChecks.length === 2 && matrixChecks[0].sourceSha256 === matrixChecks[1].sourceSha256);

const audit = {
  issue: 'UAT232/256 supplemental independent native retry review',
  status: checks.every((entry) => entry.pass) ? 'pass' : 'fail',
  checks,
  summary: {
    oldCanonicalRows: oldMessages.length,
    recoveredCanonicalRows: newMessages.length,
    oldCanonicalProof: oldMessageProof,
    matrix: matrixChecks,
  },
  limits: [
    'This audit reads synthetic UI text and private receipts only in memory; its output contains labels as check names, hashes, counts, and booleans only.',
    'No raw UI capture, provider reasoning, credentials, database data, runtime output, or private receipt fields are emitted.',
    'The immutable source conclusion is manifest- and binding-based. This review does not invoke Git or modify a native profile, process, database, or source file.',
  ],
  inputs,
};
fs.writeFileSync(path.join(output, 'supplemental-audit.json'), JSON.stringify(audit, null, 2) + '\n');
console.log(JSON.stringify({
  status: audit.status,
  passed: checks.filter((entry) => entry.pass).length,
  failed: checks.filter((entry) => !entry.pass).map((entry) => entry.name),
  oldRows: oldMessages.length,
  recoveredRows: newMessages.length,
  immutableCopies: matrixChecks.length,
  inputCount: inputs.length,
}));
if (audit.status !== 'pass') process.exitCode = 1;
