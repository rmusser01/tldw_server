// Retention-only audit for the frozen UAT236 fresh-profile packet.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const packet = 'output/playwright/fresh-matrix-repairs-2026-09-17/native-model236-fresh-accepted';
const sourceReview = '.tmp/uat-repairs-231-246/native236-fresh-review';
const out = '.tmp/uat-repairs-231-246/native236-fresh-retention-review';
const expectedManifest = '25dbb7aa812039b8f695a948837e10d75b208b6710d9e77a808f6fccf78e781f';
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const read = file => fs.readFileSync(file);
const json = file => JSON.parse(read(file));
const check = (name, pass) => checks.push({ name, pass: Boolean(pass) });
const checks = [];

const manifestPath = path.join(packet, 'manifest.json');
const manifestBytes = read(manifestPath);
const manifest = JSON.parse(manifestBytes);
check('Declared manifest checksum matches', sha(manifestBytes) === expectedManifest);
check('Packet is bounded to UAT236 acceptance',
  Array.isArray(manifest.acceptedFindings)
  && manifest.acceptedFindings.length === 1
  && manifest.acceptedFindings[0] === 236
  && manifest.fullMatrixAccepted === false);
check('Manifest secret counters are zero',
  manifest.secretScan?.knownVariants === 122
  && manifest.secretScan?.matches === 0
  && manifest.secretScan?.jwtMatches === 0);

const files = manifest.files.map(entry => {
  const packaged = read(path.join(packet, entry.path));
  const source = read(entry.source);
  return {
    packagedMatchesManifest: sha(packaged) === entry.sha256 && packaged.length === entry.bytes,
    sourceMatchesManifest: sha(source) === entry.sourceSha256 && source.length === entry.sourceBytes,
    sourceByteEqual: packaged.equals(source),
    packagedSha256: sha(packaged),
  };
});
check('Every packaged payload matches its source bytes and manifest', files.every(file =>
  file.packagedMatchesManifest && file.sourceMatchesManifest && file.sourceByteEqual));

const checkpointEntries = read(path.join(packet, 'CHECKPOINT_SHA256SUMS'))
  .toString()
  .split(/\r?\n/)
  .filter(Boolean)
  .map(line => {
    const [digest, relative] = line.split(/\s+/, 2);
    const bytes = read(path.join(packet, relative));
    return { matches: sha(bytes) === digest, sha256: sha(bytes) };
  });
check('Every packet checkpoint matches', checkpointEntries.length > 0 && checkpointEntries.every(entry => entry.matches));

const sourceAudit = json(path.join(sourceReview, 'audit.json'));
const sourceInputs = sourceAudit.inputs;
const omitted = manifest.omitted;
check('Source reviewer recorded a complete passing 21-check audit',
  sourceAudit.verdict === 'CLEAR bounded fresh-profile UAT236 AC3'
  && sourceAudit.checks.length === 21
  && sourceAudit.checks.every(entry => entry.pass === true)
  && sourceInputs.length === 43);
const inputChecks = sourceInputs.map(input => {
  const bytes = read(input.path); // Includes private inputs only for comparison; no value is retained.
  return {
    hashMatches: sha(bytes) === input.sha256,
    sizeMatches: bytes.length === input.bytes,
    privateHashOnly: input.privateHashOnly === true,
    sha256: sha(bytes),
  };
});
check('All 43 source inputs still match their recorded hashes and sizes', inputChecks.length === 43 && inputChecks.every(input => input.hashMatches && input.sizeMatches));
const stripReason = entry => ({
  path: entry.path,
  bytes: entry.bytes,
  sha256: entry.sha256,
  privateHashOnly: entry.privateHashOnly,
});
check('Packet omission records are the complete 43-input hash projection',
  Array.isArray(omitted)
  && omitted.length === 43
  && omitted.every(entry => typeof entry.reason === 'string' && entry.reason.length > 0)
  && JSON.stringify(omitted.map(stripReason)) === JSON.stringify(sourceInputs));

const sourceAuditCode = read(path.join(sourceReview, 'audit.mjs')).toString();
const sourceReviewText = read(path.join(sourceReview, 'REVIEW.md')).toString();
const semantics = sourceAudit.summary?.canonical;
const checksByName = new Map(sourceAudit.checks.map(entry => [entry.name, entry.pass]));
check('AC3 evidence is actual library Chat, exact persisted canonical completion, and no reselection',
  semantics?.exactFinalBEEP === true
  && checksByName.get('New character entered through actual library Chat button') === true
  && checksByName.get('Completion persists new exact final answer200') === true
  && checksByName.get('Retained new helper actions contain no model-picker/reselection or storage mutation') === true);
check('Fresh-profile scope and UAT261 limitation remain explicit',
  sourceReviewText.includes('fresh application profile')
  && sourceReviewText.includes('not a clean OS installation')
  && sourceReviewText.includes('does not retroactively erase')
  && sourceAudit.limits.some(limit => limit.includes('does not revise earlier261')));

const outputTexts = manifest.files.map(entry => read(path.join(packet, entry.path)).toString());
const privateHelper = read('.tmp/uat-repairs-231-246/native-preparation/model236-fresh-save-key.private.js').toString();
const privateFillValues = [...privateHelper.matchAll(/\.fill\(\s*(['"`])([^'"`]+)\1\s*\)/g)]
  .map(match => match[2])
  .filter(value => value.length >= 8);
const genericSecretPattern = /(?:\bsk-[A-Za-z0-9_-]{8,}|\bBearer\s+[A-Za-z0-9._-]{8,}|(?:api[_-]?key|authorization)\s*[:=]\s*['"]?[A-Za-z0-9._-]{8,})/i;
const packetText = outputTexts.join('\n');
const privateCredentialLeaked = privateFillValues.some(value => packetText.includes(value));
check('Safe packet does not contain known private credential values or generic token forms',
  !privateCredentialLeaked && !genericSecretPattern.test(packetText));

const retainer = read('.tmp/uat-repairs-231-246/retain-native-model236-fresh.mjs');
const result = {
  task: 'TASK13260.178 UAT236 fresh-profile retention audit',
  verdict: checks.every(entry => entry.pass) ? 'CLEAR bounded fresh-profile AC3 packet retained exactly' : 'GAPS',
  manifestSha256: sha(manifestBytes),
  retainerSha256: sha(retainer),
  checks,
  counts: {
    payloads: files.length,
    checkpoints: checkpointEntries.length,
    provenanceInputs: inputChecks.length,
    privateHashOnlyInputs: inputChecks.filter(input => input.privateHashOnly).length,
    privateFillCandidatesScanned: privateFillValues.length,
  },
  payloadHashes: files.map(file => file.packagedSha256),
  inputHashes: inputChecks.map(input => input.sha256).sort(),
};
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, 'audit.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ verdict: result.verdict, checks: checks.length, failed: checks.filter(entry => !entry.pass).map(entry => entry.name), inputs: inputChecks.length }));
