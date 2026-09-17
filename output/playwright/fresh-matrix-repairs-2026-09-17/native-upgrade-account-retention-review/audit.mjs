import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
const packet = 'output/playwright/fresh-matrix-repairs-2026-09-17/native-upgrade-account-accepted';
const out = '.tmp/uat-repairs-231-246/native-upgrade-account-retention-review';
const root = '.tmp/uat-repairs-231-246';
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const bytes = name => fs.readFileSync(name);
const json = name => JSON.parse(bytes(name));
const problems = [];
const check = (value, problem) => { if (!value) problems.push(problem); };
const safe = name => typeof name === 'string' && !path.isAbsolute(name) && name !== '.' && path.normalize(name) === name && !name.split(path.sep).includes('..');
const walk = (dir, prefix = '') => fs.readdirSync(path.join(dir, prefix), { withFileTypes: true }).flatMap(entry => {
  const name = path.join(prefix, entry.name);
  check(!entry.isSymbolicLink(), `Symlink in retained inventory: ${name}`);
  return entry.isDirectory() ? walk(dir, name) : [name];
});
const manifest = json(path.join(packet, 'manifest.json'));
const manifestHash = sha(bytes(path.join(packet, 'manifest.json')));
const checksumsHash = sha(bytes(path.join(packet, 'CHECKPOINT_SHA256SUMS')));
check(manifestHash === '6cb62377b2f6c39218c4c367164540648053dd0722203ddee778e66d07d1ff3f', 'Top manifest hash differs');
check(checksumsHash === '8ccdc5991de8c7c28331b30f35c864471c52c3b61e1741ebb4e820abe089bcdc', 'Top checksums hash differs');
check(JSON.stringify(manifest.acceptedFindings) === '[243,248,254]' && manifest.fullMatrixAccepted === false && manifest.runtimeSource === 'a7d3155a567afb25982eb360ea24b973cc3249c9', 'Acceptance scope metadata differs');
const files = walk(packet).sort();
const expectedFiles = manifest.files.map(file => file.path).concat(['README.md', 'manifest.json', 'CHECKPOINT_SHA256SUMS']).sort();
check(manifest.files.length === 74 && files.length === 77, 'Expected 74 payloads / 77 total files');
check(new Set(expectedFiles).size === expectedFiles.length && new Set(manifest.files.map(file => file.source)).size === manifest.files.length, 'Duplicate retained/source paths');
check(JSON.stringify(files) === JSON.stringify(expectedFiles), 'Manifest inventory differs from disk');
const expectedSources = new Map(), excludedPrivate = [];
function expect(source, hash) {
  check(!expectedSources.has(source) || expectedSources.get(source) === hash, `Independent reviews disagree on source hash: ${source}`);
  expectedSources.set(source, hash);
}
for (const [dir, key] of [['upgrade254-native-review', 'evidence'], ['native-account-review', 'reviewedFiles']]) {
  const audit = json(path.join(root, dir, 'audit.json'));
  for (const entry of audit[key]) {
    if (/private|credentials/i.test(path.basename(entry.path))) excludedPrivate.push(entry.path);
    else expect(entry.path, entry.sha256);
  }
  for (const name of ['REVIEW.md', 'audit.mjs', 'audit.json']) {
    const source = path.join(root, dir, name); expect(source, sha(bytes(source)));
  }
}
for (const name of ['REVIEW.md', 'audit.mjs', 'audit.json']) {
  const source = path.join(root, 'upgrade254-retention-review', name); expect(source, sha(bytes(source)));
}
for (const name of ['copy-result.txt', 'runtime-health.json']) {
  const source = path.join(root, 'upgrade254', name); expect(source, sha(bytes(source)));
}
check(expectedSources.size === 74 && JSON.stringify([...expectedSources.keys()].sort()) === JSON.stringify(manifest.files.map(file => file.source).sort()), 'Inventory differs from the independent-review inputs and specified metadata');
const payloads = [], payloadResults = [], gzipResults = [];
for (const file of manifest.files) {
  check(safe(file.source) && safe(file.path), `Unsafe path: ${file.path}`);
  check(!/private|credentials/i.test(path.basename(file.source)), `Private record copied: ${file.source}`);
  check(fs.lstatSync(file.source).isFile(), `Original source not regular: ${file.source}`);
  const stored = bytes(path.join(packet, file.path)), original = bytes(file.source);
  const decoded = file.encoding === 'gzip' ? zlib.gunzipSync(stored) : stored;
  const identity = file.encoding === 'identity', compressed = file.encoding === 'gzip';
  check(identity || compressed, `Unknown payload encoding: ${file.path}`);
  check(file.path === `evidence/${file.source}${compressed ? '.gz' : ''}`, `Destination mapping differs: ${file.path}`);
  check(stored.length === file.bytes && sha(stored) === file.sha256, `Stored byte/hash mismatch: ${file.path}`);
  check(decoded.length === file.sourceBytes && sha(decoded) === file.sourceSha256, `Decoded byte/hash mismatch: ${file.path}`);
  check(original.equals(decoded), `Original and decoded bytes differ: ${file.source}`);
  check(expectedSources.get(file.source) === file.sourceSha256, `Reviewed input hash mismatch: ${file.source}`);
  if (compressed) gzipResults.push({ source: file.source, storedBytes: stored.length, sourceBytes: decoded.length, storedSha256: sha(stored), sourceSha256: sha(decoded), exactRoundtrip: decoded.equals(original) });
  payloads.push({ name: file.source, text: decoded.toString('utf8') });
  payloadResults.push({ source: file.source, sha256: sha(original), sourceMatches: original.equals(decoded), reviewedHashMatches: expectedSources.get(file.source) === file.sourceSha256 });
}
check(gzipResults.length === 2, 'Expected exactly two gzip payloads');
const checksumNames = [];
for (const line of bytes(path.join(packet, 'CHECKPOINT_SHA256SUMS')).toString().trim().split('\n')) {
  const match = /^([a-f0-9]{64})  (.+)$/.exec(line);
  if (!match) { check(false, 'Malformed checksum entry'); continue; }
  const [, hash, name] = match; checksumNames.push(name);
  check(safe(name) && sha(bytes(path.join(packet, name))) === hash, `Checkpoint checksum mismatch: ${name}`);
}
check(new Set(checksumNames).size === checksumNames.length && JSON.stringify(checksumNames.sort()) === JSON.stringify(files.filter(file => file !== 'CHECKPOINT_SHA256SUMS')), 'Checkpoint inventory differs');
const known = [];
const matrix = '.tmp/uat-next-matrix-20260916';
const profileNames = fs.readdirSync(matrix).filter(name => name.endsWith('.profile.private.json')).sort();
for (const name of profileNames) {
  const profile = json(path.join(matrix, name)), credentials = json(profile.credentialsPath);
  known.push(credentials.apiKey, credentials.jwtSecret, credentials.apiHashSecret, ...Object.values(credentials.accounts ?? {}).map(account => account.password), ...(credentials.providerSecrets ?? []));
  if (profile.pgConfigPath) known.push(json(profile.pgConfigPath).password);
}
known.push(json('.tmp/fresh-uat-recovery-20260916/postgres-private.json').password);
const rawValues = [...new Set(known.filter(value => typeof value === 'string' && value.length > 0))];
const variants = [...new Set(rawValues.flatMap(value => [value, encodeURIComponent(value), JSON.stringify(value).slice(1, -1)]))];
const encodedVariants = [...new Set(rawValues.flatMap(value => [Buffer.from(value).toString('base64'), Buffer.from(value).toString('base64url')]))];
const metadataNames = ['README.md', 'manifest.json', 'CHECKPOINT_SHA256SUMS'];
const scanInputs = payloads.concat(metadataNames.map(name => ({ name, text: bytes(path.join(packet, name)).toString('utf8') })));
const knownMatches = [], encodedMatches = [], jwtMatches = [];
for (const file of scanInputs) {
  if (variants.some(value => file.text.includes(value))) knownMatches.push(file.name);
  if (encodedVariants.some(value => file.text.includes(value))) encodedMatches.push(file.name);
  if (/eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/.test(file.text)) jwtMatches.push(file.name);
}
check(profileNames.includes('mcp251-fresh-targeted-20260917-pg-single.profile.private.json'), 'Current MCP251 credential profile not included');
check(variants.length >= manifest.secretScan.knownVariants && manifest.secretScan.matches === 0 && manifest.secretScan.jwtMatches === 0, 'Credential scan metadata mismatch');
check(knownMatches.length === 0 && encodedMatches.length === 0 && jwtMatches.length === 0, 'Credential or JWT candidate found');
const textFor = suffix => payloads.find(file => file.name.endsWith(suffix))?.text;
const nativeAudit = JSON.parse(textFor('upgrade254-native-review/audit.json'));
const accountAudit = JSON.parse(textFor('native-account-review/audit.json'));
const nativeReview = textFor('upgrade254-native-review/REVIEW.md');
const accountReview = textFor('native-account-review/REVIEW.md');
const readme = bytes(path.join(packet, 'README.md')).toString();
check(nativeAudit.verdict === 'CLEAR' && nativeAudit.problems.length === 0, 'Native upgrade review is not clear');
check(accountAudit.checks.length === 26 && accountAudit.checks.every(item => item.passed === true), 'Native account review does not preserve all 26 successful checks');
check(accountReview.includes('CLEAR for both bounded PostgreSQL multi-user acceptance cases'), 'Account review acceptance missing');
check(readme.includes('UAT254') && readme.includes('UAT243') && readme.includes('UAT248') && readme.includes('full48-outcome fresh rerun remain open'), 'README bounded scope mismatch');
check(readme.includes('earlier reload\'s stale canonical events are excluded') && accountReview.includes('It is therefore excluded from settled/canonical acceptance'), 'README early reload qualification mismatch');
check(readme.includes('not provider-side cancellation') && accountReview.includes('does not prove provider-side cancellation'), 'README cancellation qualification mismatch');
check(readme.includes('does not establish continuous-tab survival') && nativeReview.includes('not continuous survival of the same tab'), 'README browser continuity qualification mismatch');
const failurePaths = ['account-boundary-prepared.txt', 'account-boundary-logout.txt', 'account-boundary-bob-login.txt', 'bob-upgrade-readback.txt', 'bob-ordinary-reloaded.txt', 'media-upgraded-read.txt'];
check(failurePaths.every(name => payloads.some(file => file.name.endsWith('/' + name))), 'Historical failure/contradiction evidence omitted');
const previous = 'output/playwright/fresh-matrix-repairs-2026-09-17/targeted-upgrade-harness-reviewed';
const previousManifestHash = sha(bytes(path.join(previous, 'manifest.json')));
const previousChecksumsHash = sha(bytes(path.join(previous, 'CHECKPOINT_SHA256SUMS')));
check(previousManifestHash === '592c3292c98a5a4abfb5c72f0b9ff1e933751d2cb0c6631207c57e35d0e50f70' && previousChecksumsHash === 'be750f6be7a0eadff9f8580c78f062d9a152e89b94d8220b2ab0f9a5b6ab0b04', 'Earlier harness packet metadata changed');
let previousChecks = 0;
for (const line of bytes(path.join(previous, 'CHECKPOINT_SHA256SUMS')).toString().trim().split('\n')) {
  const [, hash, name] = /^([a-f0-9]{64})  (.+)$/.exec(line);
  check(sha(bytes(path.join(previous, name))) === hash, `Earlier harness packet file changed: ${name}`); previousChecks++;
}
check(bytes(path.join(previous, 'README.md')).toString().includes('Actual corrected startup and original-data readback remain pending'), 'Earlier checkpoint historical pending status changed');
const previousFiles = walk(previous).sort();
const previousExpectedFiles = bytes(path.join(previous, 'CHECKPOINT_SHA256SUMS')).toString().trim().split('\n').map(line => line.slice(66)).concat('CHECKPOINT_SHA256SUMS').sort();
check(previousFiles.length === 66 && JSON.stringify(previousFiles) === JSON.stringify(previousExpectedFiles), 'Earlier packet inventory changed');
const result = { verdict: problems.length ? 'BLOCK' : 'CLEAR', at: new Date().toISOString(), packet, acceptedFindings: manifest.acceptedFindings, fullMatrixAccepted: false, payloadCount: payloads.length, totalFiles: files.length, checkpointEntries: checksumNames.length, independentSourceInventoryCount: expectedSources.size, privateInputsExcluded: new Set(excludedPrivate).size, manifestSha256: manifestHash, checksumsSha256: checksumsHash, retainerSha256: sha(bytes(path.join(root, 'retain-upgrade-account-native.mjs'))), gzipResults, payloadResults, credentialScan: { scannedOriginalPayloads: payloads.length, scannedMetadataFiles: metadataNames.length, profiles: profileNames, uniqueRawSecrets: rawValues.length, knownVariants: variants.length, additionalEncodedVariants: encodedVariants.length, knownMatches, encodedMatches, jwtMatches }, priorPacket: { manifestSha256: previousManifestHash, checksumsSha256: previousChecksumsHash, verifiedFiles: previousChecks, historicalPendingStatusPreserved: true }, problems };
fs.writeFileSync(path.join(out, 'audit.json'), JSON.stringify(result, null, 2) + '\n', { mode: 0o600 });
console.log(JSON.stringify({ ...result, payloadResults: undefined, credentialScan: { ...result.credentialScan, profiles: undefined } }, null, 2));
if (problems.length) process.exitCode = 1;
