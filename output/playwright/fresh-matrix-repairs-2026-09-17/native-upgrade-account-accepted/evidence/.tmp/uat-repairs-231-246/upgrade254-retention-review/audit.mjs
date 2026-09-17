import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const packet = 'output/playwright/fresh-matrix-repairs-2026-09-17/targeted-upgrade-harness-reviewed';
const out = '.tmp/uat-repairs-231-246/upgrade254-retention-review';
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
const read = file => fs.readFileSync(file);
const json = file => JSON.parse(read(file));
const problems = [];
const check = (condition, message) => { if (!condition) problems.push(message); };
const walk = (root, prefix = '') => fs.readdirSync(path.join(root, prefix), { withFileTypes: true }).flatMap(entry => {
  const relative = path.join(prefix, entry.name);
  check(!entry.isSymbolicLink(), `Symbolic link in retained packet: ${relative}`);
  return entry.isDirectory() ? walk(root, relative) : [relative];
});
const safeRelative = name => typeof name === 'string' && !path.isAbsolute(name) && name !== '.' && path.normalize(name) === name && !name.split(path.sep).includes('..');
const files = walk(packet).sort();
const manifest = json(path.join(packet, 'manifest.json'));
const actualManifestHash = sha(read(path.join(packet, 'manifest.json')));
const actualChecksumsHash = sha(read(path.join(packet, 'CHECKPOINT_SHA256SUMS')));
check(actualManifestHash === '592c3292c98a5a4abfb5c72f0b9ff1e933751d2cb0c6631207c57e35d0e50f70', 'Top manifest hash mismatch');
check(actualChecksumsHash === 'be750f6be7a0eadff9f8580c78f062d9a152e89b94d8220b2ab0f9a5b6ab0b04', 'Checksums hash mismatch');
check(manifest.files.length === 63, 'Expected 63 payloads');
check(files.length === 66, 'Expected 66 retained files');
check(manifest.task === 'TASK13260.196' && manifest.finding === 254, 'Task/finding mismatch');
const expectedFiles = manifest.files.map(entry => entry.path).concat(['README.md', 'manifest.json', 'CHECKPOINT_SHA256SUMS']).sort();
check(new Set(expectedFiles).size === expectedFiles.length, 'Duplicate inventory paths');
check(JSON.stringify(expectedFiles) === JSON.stringify(files), 'Manifest file inventory differs from disk');
for (const entry of manifest.files) {
  check(safeRelative(entry.path) && safeRelative(entry.source), `Unsafe manifest path: ${entry.path}`);
  check(!/private|credential/i.test(path.basename(entry.path)), `Private filename retained: ${entry.path}`);
  const retained = read(path.join(packet, entry.path)), source = read(entry.source);
  check(fs.lstatSync(entry.source).isFile(), `Source not regular: ${entry.source}`);
  check(retained.length === entry.bytes && sha(retained) === entry.sha256, `Payload bytes/hash mismatch: ${entry.path}`);
  check(Buffer.compare(source, retained) === 0, `Source differs from retained payload: ${entry.path}`);
}
const checksumLines = read(path.join(packet, 'CHECKPOINT_SHA256SUMS')).toString().trim().split('\n');
const checksumPaths = [];
for (const line of checksumLines) {
  const match = /^([a-f0-9]{64})  (.+)$/.exec(line);
  if (!match) { check(false, 'Malformed checksum line'); continue; }
  const [, hash, file] = match;
  checksumPaths.push(file);
  check(safeRelative(file), `Unsafe checksum path: ${file}`);
  check(sha(read(path.join(packet, file))) === hash, `Checkpoint checksum mismatch: ${file}`);
}
check(JSON.stringify(checksumPaths.sort()) === JSON.stringify(files.filter(file => file !== 'CHECKPOINT_SHA256SUMS')), 'Checkpoint inventory differs from disk');
const initial = json(path.join(packet, 'initial-reviewed/evidence-manifest.json'));
for (const entry of initial.files) check(sha(read(path.join(packet, 'initial-reviewed', entry.path))) === entry.sha256, `Initial evidence hash mismatch: ${entry.path}`);
check(sha(read(path.join(packet, 'initial-reviewed/owned-manifest.json'))) === initial.ownedManifestSha256, 'Initial owned manifest hash mismatch');
const owned = json(path.join(packet, 'initial-reviewed/owned-manifest.json'));
for (const entry of owned.files) check(sha(read(path.join(packet, 'initial-reviewed/source-snapshot', entry.path))) === entry.sha256, `Initial snapshot hash mismatch: ${entry.path}`);
const corrected = json(path.join(packet, 'upgrade254/source-freeze.json'));
for (const entry of corrected.files) check(sha(read(path.join(packet, 'corrected-source', entry.path))) === entry.sha256, `Corrected source hash mismatch: ${entry.path}`);
check(owned.productCandidate === corrected.productCandidate && corrected.productCandidate === 'a7d3155a567afb25982eb360ea24b973cc3249c9', 'Product candidate mismatch');
const oldTest = 'initial-reviewed/matrix-upgrade.test.mjs';
check(Buffer.compare(read(path.join(packet, oldTest)), read(path.join(packet, 'upgrade254/baseline-matrix-upgrade.test.mjs'))) === 0, 'Initial test baseline preservation mismatch');
check(sha(read(path.join(packet, oldTest))) !== sha(read(path.join(packet, 'corrected-source/.tmp/uat-repairs-231-246/native-upgrade-preparation/matrix-upgrade.test.mjs'))), 'Initial and corrected tests incorrectly conflated');
const texts = new Map(files.map(file => [file, read(path.join(packet, file)).toString('utf8')]));
const secrets = [];
const addSecret = value => { if (typeof value === 'string' && value.length) secrets.push(value); };
const matrix = '.tmp/uat-next-matrix-20260916';
for (const [run, cells] of [['fresh-final-20260917', ['sqlite-single', 'sqlite-multi', 'pg-single', 'pg-multi']], ['repairs231-250-targeted-20260917', ['pg-single', 'pg-multi']]]) {
  for (const cell of cells) {
    const profile = json(path.join(matrix, `${run}-${cell}.profile.private.json`));
    const credentials = json(profile.credentialsPath);
    for (const key of ['apiKey', 'jwtSecret', 'apiHashSecret']) addSecret(credentials[key]);
    for (const account of Object.values(credentials.accounts ?? {})) addSecret(account.password);
    for (const value of credentials.providerSecrets ?? []) addSecret(value);
    if (profile.pgConfigPath) addSecret(json(profile.pgConfigPath).password);
  }
}
addSecret(json('.tmp/fresh-uat-recovery-20260916/postgres-private.json').password);
for (const cell of ['pg-single', 'pg-multi']) addSecret(json(path.join(matrix, `holders/fresh-final-20260917-${cell}/runtime.pg-config.private.json`)).password);
const rawSecrets = [...new Set(secrets)];
const variants = [...new Set(rawSecrets.flatMap(value => [value, encodeURIComponent(value), JSON.stringify(value).slice(1, -1)]))];
const encodedVariants = [...new Set(rawSecrets.flatMap(value => [Buffer.from(value).toString('base64'), Buffer.from(value).toString('base64url')]))];
const knownMatches = [], encodedMatches = [], jwtMatches = [];
for (const [file, text] of texts) {
  if (variants.some(value => text.includes(value))) knownMatches.push(file);
  if (encodedVariants.some(value => text.includes(value))) encodedMatches.push(file);
  if (/eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/.test(text)) jwtMatches.push(file);
}
check(variants.length === manifest.secretScan.knownValues, 'Known credential scan inventory differs');
check(knownMatches.length === 0 && encodedMatches.length === 0 && jwtMatches.length === 0, 'Credential or JWT candidate found');
const hasSummary = (file, total) => {
  const text = texts.get(file);
  return new RegExp(`tests ${total}(?:\\s|$)`).test(text) && new RegExp(`pass ${total}(?:\\s|$)`).test(text) && /fail 0(?:\s|$)/.test(text) && /skipped 0(?:\s|$)/.test(text);
};
check(hasSummary('native-upgrade-review/combined.log', 124), 'Initial independent 124-pass evidence missing');
check(hasSummary('upgrade254-review/combined-independent.log', 125), 'Corrected independent 125-pass evidence missing');
check(texts.get('initial-native/frontend-first-start.redacted.log').includes('TLDW_NEXT_DIST_DIR must be a direct .next-live-tier-* child directory'), 'Native failure evidence missing');
check(texts.get('upgrade254-review/red-independent.log').includes('TLDW_NEXT_DIST_DIR must be a direct .next-live-tier-* child directory'), 'Causal RED evidence missing');
const staticReport = json(path.join(packet, 'upgrade254-review/static-independent.json'));
check(staticReport.helperExactlyPrefixReplacement && staticReport.nextConfigMatchesCandidate && staticReport.syntax.every(result => result.status === 0) && staticReport.lint.every(result => result.errors === 0 && result.warnings === 0), 'Static README claim not supported');
const readme = texts.get('README.md');
check(readme.includes('Actual corrected startup and original-data readback remain pending') && readme.includes('No fresh-install/full-matrix acceptance is claimed'), 'README acceptance boundary missing');
const result = { verdict: problems.length ? 'BLOCK' : 'CLEAR', packet, payloadCount: manifest.files.length, totalFiles: files.length, checkpointEntries: checksumPaths.length, sourcePayloadMatches: manifest.files.length - problems.filter(problem => problem.startsWith('Source differs')).length, initialEvidenceEntries: initial.files.length, initialSourceSnapshots: owned.files.length, correctedSourceSnapshots: corrected.files.length, manifestSha256: actualManifestHash, checksumsSha256: actualChecksumsHash, retainerSha256: sha(read('.tmp/uat-repairs-231-246/retain-upgrade254.mjs')), credentialScan: { scannedFiles: files.length, uniqueRawSecrets: rawSecrets.length, knownVariants: variants.length, additionalEncodedVariants: encodedVariants.length, knownMatches, encodedMatches, jwtMatches }, problems };
fs.writeFileSync(path.join(out, 'audit.json'), JSON.stringify(result, null, 2) + '\n', { mode: 0o600 });
console.log(JSON.stringify(result, null, 2));
if (problems.length) process.exitCode = 1;
