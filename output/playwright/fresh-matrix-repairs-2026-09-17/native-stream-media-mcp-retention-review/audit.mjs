import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
const packet = 'output/playwright/fresh-matrix-repairs-2026-09-17/native-stream-media-mcp-accepted';
const root = '.tmp/uat-repairs-231-246';
const out = path.join(root, 'native-stream-media-mcp-retention-review');
const cwd = fs.realpathSync(process.cwd());
const read = file => fs.readFileSync(file);
const json = file => JSON.parse(read(file));
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const problems = [];
const check = (condition, problem) => { if (!condition) problems.push(problem); };
const safeRelative = name => typeof name === 'string' && !path.isAbsolute(name) && name !== '.' && path.normalize(name) === name && !name.split(path.sep).includes('..');
const within = name => name.startsWith(cwd + path.sep);
const walk = (dir, prefix = '') => fs.readdirSync(path.join(dir, prefix), { withFileTypes: true }).flatMap(entry => {
  const relative = path.join(prefix, entry.name);
  check(!entry.isSymbolicLink(), `Retained symlink: ${relative}`);
  return entry.isDirectory() ? walk(dir, relative) : [relative];
});
const manifest = json(path.join(packet, 'manifest.json'));
const manifestSha256 = sha(read(path.join(packet, 'manifest.json')));
const checksumsSha256 = sha(read(path.join(packet, 'CHECKPOINT_SHA256SUMS')));
check(manifestSha256 === 'f8d1c2c9287e456fa71b5eb0ff731301bc42e3f8fa755caabec99cb1f0f04902', 'Manifest SHA256 mismatch');
check(checksumsSha256 === '84a2297b176074dcab1b1675e009893a16315e13319ee620972dbc819994b984', 'Checksums SHA256 mismatch');
check(JSON.stringify(manifest.acceptedFindings) === '[241,246,251,253]' && manifest.fullMatrixAccepted === false && manifest.runtimeSource === 'a7d3155a567afb25982eb360ea24b973cc3249c9', 'Acceptance metadata scope mismatch');
const inventory = walk(packet).sort();
const expectedInventory = manifest.files.map(entry => entry.path).concat(['README.md', 'manifest.json', 'CHECKPOINT_SHA256SUMS']).sort();
check(manifest.files.length === 186 && inventory.length === 189, 'Expected 186 payloads and 189 total files');
check(new Set(expectedInventory).size === expectedInventory.length && new Set(manifest.files.map(entry => entry.source)).size === manifest.files.length, 'Duplicate destination/source');
check(JSON.stringify(inventory) === JSON.stringify(expectedInventory), 'Disk inventory differs from manifest');
const expectedSources = new Map(), omissions = [], reviewSummaries = [];
const add = (source, expectedHash) => {
  const normalized = path.relative(cwd, path.resolve(source));
  check(safeRelative(normalized), `Source outside permitted repository inventory: ${normalized}`);
  check(!expectedSources.has(normalized) || expectedSources.get(normalized) === expectedHash, `Independent reviews disagree on bytes: ${normalized}`);
  expectedSources.set(normalized, expectedHash);
};
for (const [name, expectedChecks, expectedFiles] of [['native-stream246-review', 42, 53], ['native-media-review', 27, 78], ['native-mcp251-review', 38, 73]]) {
  const audit = json(path.join(root, name, 'audit.json'));
  check(audit.checks.length === expectedChecks && audit.checks.every(entry => entry.passed === true) && audit.reviewedFiles.length === expectedFiles, `${name}: expected independent verification counts not retained`);
  const verdictsClear = audit.verdict === 'CLEAR' || (audit.verdicts && Object.values(audit.verdicts).every(verdict => verdict.startsWith('CLEAR')));
  check(verdictsClear, `${name}: verdict not CLEAR`);
  reviewSummaries.push({ name, checks: audit.checks.length, passed: audit.checks.filter(entry => entry.passed).length, reviewedFiles: audit.reviewedFiles.length, verdict: audit.verdict ?? audit.verdicts });
  for (const entry of audit.reviewedFiles) {
    const reason = /private|credentials/i.test(path.basename(entry.path)) ? 'private record or runtime log; represented by safe audit hash' : entry.path.endsWith('.tar') ? 'repository archive; represented by safe audit hash' : undefined;
    if (reason) { omissions.push({ path: path.relative(cwd, path.resolve(entry.path)), sha256: entry.sha256, reason }); continue; }
    add(entry.path, entry.sha256);
  }
  for (const nameOfFile of ['REVIEW.md', 'audit.mjs', 'audit.json']) {
    const source = path.join(root, name, nameOfFile); add(source, sha(read(source)));
  }
}
check(expectedSources.size === 186 && JSON.stringify([...expectedSources.keys()].sort()) === JSON.stringify(manifest.files.map(entry => entry.source).sort()), 'Payload inventory differs from all nonprivate reviewed inputs and three review packets');
const payloadResults = [], gzipResults = [], scanInputs = [], sourceAliases = [];
let sourceBytes = 0, storedBytes = 0;
for (const entry of manifest.files) {
  check(safeRelative(entry.source) && safeRelative(entry.path), `Unsafe retained path: ${entry.path}`);
  check(!/private|credentials/i.test(path.basename(entry.source)) && !entry.source.endsWith('.tar'), `Excluded private/archive source copied: ${entry.source}`);
  const absoluteSource = path.resolve(entry.source), realSource = fs.realpathSync(absoluteSource);
  const regularLeaf = fs.lstatSync(absoluteSource).isFile() && !fs.lstatSync(absoluteSource).isSymbolicLink();
  const reviewedBunAlias = /\/apps\/tldw-frontend\/node_modules\/(?:next|react|typescript)\/package\.json$/.test(absoluteSource) && realSource.startsWith(absoluteSource.split('/apps/tldw-frontend/node_modules/')[0] + '/apps/node_modules/.bun/');
  if (realSource !== absoluteSource) sourceAliases.push({ source: entry.source, resolved: path.relative(cwd, realSource), reviewedBunAlias });
  check(within(realSource) && regularLeaf && (realSource === absoluteSource || reviewedBunAlias), `Source target is not a confined regular file or reviewed dependency alias: ${entry.source}`);
  const original = read(entry.source), stored = read(path.join(packet, entry.path));
  const compressed = entry.encoding === 'gzip';
  check(compressed || entry.encoding === 'identity', `Unknown encoding: ${entry.path}`);
  const restored = compressed ? zlib.gunzipSync(stored) : stored;
  check(entry.path === `evidence/${entry.source}${compressed ? '.gz' : ''}`, `Storage path mapping mismatch: ${entry.path}`);
  check(compressed === (original.length > 256000), `Compression threshold mismatch: ${entry.path}`);
  check(stored.length === entry.bytes && sha(stored) === entry.sha256, `Stored byte/hash mismatch: ${entry.path}`);
  check(restored.length === entry.sourceBytes && sha(restored) === entry.sourceSha256, `Original byte/hash mismatch: ${entry.path}`);
  check(original.equals(restored), `Original source differs from restored payload: ${entry.source}`);
  check(expectedSources.get(entry.source) === entry.sourceSha256, `Independent audit source hash mismatch: ${entry.source}`);
  const result = { source: entry.source, originalBytes: original.length, originalSha256: sha(original), storedBytes: stored.length, storedSha256: sha(stored), encoding: entry.encoding, exactOriginalMatch: original.equals(restored) };
  payloadResults.push(result); if (compressed) gzipResults.push(result);
  scanInputs.push({ name: entry.source, text: restored.toString('utf8') });
  sourceBytes += original.length; storedBytes += stored.length;
}
check(gzipResults.length === 12, 'Expected exactly twelve gzip payloads');
const checksumPaths = [];
for (const line of read(path.join(packet, 'CHECKPOINT_SHA256SUMS')).toString().trim().split('\n')) {
  const match = /^([a-f0-9]{64})  (.+)$/.exec(line);
  if (!match) { check(false, 'Malformed checksum line'); continue; }
  const [, hash, name] = match; checksumPaths.push(name);
  check(safeRelative(name) && sha(read(path.join(packet, name))) === hash, `Checkpoint hash mismatch: ${name}`);
}
check(new Set(checksumPaths).size === checksumPaths.length && JSON.stringify(checksumPaths.sort()) === JSON.stringify(inventory.filter(name => name !== 'CHECKPOINT_SHA256SUMS')), 'Checkpoint inventory mismatch');
const metadata = ['README.md', 'manifest.json', 'CHECKPOINT_SHA256SUMS'];
for (const name of metadata) scanInputs.push({ name, text: read(path.join(packet, name)).toString('utf8') });
const matrix = '.tmp/uat-next-matrix-20260916';
const profiles = fs.readdirSync(matrix).filter(name => name.endsWith('.profile.private.json')).sort();
const values = [];
for (const name of profiles) {
  const profile = json(path.join(matrix, name)), credentials = json(profile.credentialsPath);
  values.push(credentials.apiKey, credentials.jwtSecret, credentials.apiHashSecret, ...Object.values(credentials.accounts ?? {}).map(account => account.password), ...(credentials.providerSecrets ?? []));
  if (profile.pgConfigPath) values.push(json(profile.pgConfigPath).password);
}
values.push(json('.tmp/fresh-uat-recovery-20260916/postgres-private.json').password);
const raw = [...new Set(values.filter(value => typeof value === 'string' && value.length > 0))];
const variants = [...new Set(raw.flatMap(value => [value, encodeURIComponent(value), JSON.stringify(value).slice(1, -1), Buffer.from(value).toString('base64'), Buffer.from(value).toString('base64url')]))];
const secretMatches = [], jwtMatches = [];
for (const entry of scanInputs) {
  if (variants.some(value => entry.text.includes(value))) secretMatches.push(entry.name);
  if (/eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/.test(entry.text)) jwtMatches.push(entry.name);
}
check(variants.length >= manifest.secretScan.knownVariants && manifest.secretScan.matches === 0 && manifest.secretScan.jwtMatches === 0, 'Secret scan metadata mismatch');
check(secretMatches.length === 0 && jwtMatches.length === 0, 'Credential/JWT candidate found');
check(profiles.includes('mcp251-fresh-targeted-20260917-pg-single.profile.private.json'), 'Current fresh MCP credentials omitted from scan');
const retained = suffix => scanInputs.find(entry => entry.name.endsWith(suffix))?.text;
const readme = read(path.join(packet, 'README.md')).toString('utf8');
const streamReview = retained('native-stream246-review/REVIEW.md');
const mediaReview = retained('native-media-review/REVIEW.md');
const mcpReview = retained('native-mcp251-review/REVIEW.md');
check(readme.includes('UAT246') && readme.includes('UAT253') && readme.includes('UAT241') && readme.includes('UAT251') && readme.includes('does not release or accept the full fresh matrix'), 'README acceptance scope missing');
check(streamReview.includes('exact fresh TestBot first-turn scenario in both PostgreSQL modes') && streamReview.includes('does not retrospectively establish why the original 45-second native request failed') && readme.includes('existing-history runs ended with reasoning but no final answer') && readme.includes('initial claim of model overlap was corrected'), 'README stream qualification mismatch');
check(mediaReview.includes('single final newline') && mediaReview.includes('MutationObserver') && mediaReview.includes('original SQLite multi-user itself'), 'Media qualifications not retained');
check(mcpReview.includes('first-chat step remains uncompleted') && mcpReview.includes('overall setup is still') && readme.includes('first chat and optional audio remain outside its acceptance'), 'MCP boundary mismatch');
check(readme.includes('Private records, raw runtime logs and large repository archives') && readme.includes('not copied') && readme.includes('UAT232 remains open') && readme.includes('additional upload-isolation findings'), 'README omitted inputs or unaccepted issues are not qualified');
const retainerFile = path.join(root, 'retain-stream-media-mcp-native.mjs');
const retainer = read(retainerFile).toString();
const pathValidationBeforeWrites = retainer.indexOf("throw Error('Unsafe source path')") < retainer.indexOf('fs.mkdirSync(destination');
check(retainer.includes('path.relative(process.cwd(),path.resolve(entry.path))') && retainer.includes("source.split('/').includes('..')") && retainer.includes('fs.lstatSync(source).isSymbolicLink()') && pathValidationBeforeWrites, 'Retainer normalized-path validation/order differs');
const result = { verdict: problems.length ? 'BLOCK' : 'CLEAR', at: new Date().toISOString(), packet, acceptedFindings: manifest.acceptedFindings, fullMatrixAccepted: false, payloads: manifest.files.length, totalFiles: inventory.length, checksumEntries: checksumPaths.length, gzipCount: gzipResults.length, sourceBytes, storedBytes, manifestSha256, checksumsSha256, retainerSha256: sha(retainer), reviewSummaries, omittedInputs: omissions, payloadResults, gzipResults, credentialScan: { profiles, uniqueRawSecrets: raw.length, knownVariants: variants.length, originalPayloadsScanned: manifest.files.length, metadataScanned: metadata.length, secretMatches, jwtMatches }, pathSafety: { allActualTargetsConfinedAndRegular: !problems.some(problem => problem.startsWith('Source target')), reviewedInternalDependencyAliases: sourceAliases, relativeNormalizationPresent: true, lexicalOutsideAndLeafSymlinkGuardBeforeWrites: pathValidationBeforeWrites, retainerExecuted: false, initialFailedAttempt: 'Controller-reported history; current source ordering and completed packet were independently checked, not the failed invocation replayed.' }, auditCorrections: ['Top-level README selected explicitly instead of first README payload.', 'Three reviewed internal Bun dependency-directory aliases are accepted; regular-leaf and resolved-target confinement checks remain enforced.'], problems };
fs.writeFileSync(path.join(out, 'audit.json'), JSON.stringify(result, null, 2) + '\n', { mode: 0o600 });
console.log(JSON.stringify({ ...result, payloadResults: undefined, gzipResults: undefined, omittedInputs: { total: omissions.length, private: omissions.filter(entry => !entry.path.endsWith('.tar')).length, archive: omissions.filter(entry => entry.path.endsWith('.tar')).length }, credentialScan: { ...result.credentialScan, profiles: undefined } }, null, 2));
if (problems.length) process.exitCode = 1;
