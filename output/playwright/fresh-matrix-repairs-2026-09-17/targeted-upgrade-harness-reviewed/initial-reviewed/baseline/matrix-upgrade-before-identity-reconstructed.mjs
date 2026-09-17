// TASK13260.196: existing-profile targeted acceptance; never prepare or provision.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import net from 'node:net';
import { fileURLToPath } from 'node:url';
import { spawn, spawnSync } from 'node:child_process';
import { privateRoot, privateJson, writePrivate, specFor, inspectInputs, requireInitialized, runtimeEnv, frontendEnv, fail, reportFailure } from './matrix-launcher.mjs';

const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const fileHash = file => hash(fs.readFileSync(file));
const requireThat = (condition, message) => { if (!condition) fail(message); };
const within = (file, root) => file.startsWith(root + path.sep);
const stat = file => { try { return fs.lstatSync(file); } catch (error) { if (error.code === 'ENOENT') return undefined; throw error; } };

function confined(file, root) {
  requireThat(within(file, root) && fs.realpathSync(file) === file, 'Upgrade path escaped its owned root');
  return file;
}

function directoryPath(dir) {
  requireThat(within(dir, privateRoot), 'Upgrade directory escaped its packet');
  let current = privateRoot;
  for (const part of path.relative(privateRoot, dir).split(path.sep)) {
    current = path.join(current, part);
    const existing = stat(current);
    requireThat(!existing || (existing.isDirectory() && !existing.isSymbolicLink()), 'Upgrade directory is not an owned directory');
  }
}

function checkArchiveStorage(root) {
  for (const relative of ['Databases', 'Databases/downloads']) {
    const parent = stat(path.join(root, relative));
    requireThat(!parent || (parent.isDirectory() && !parent.isSymbolicLink()), 'Archive-local storage parent is not a regular directory');
  }
  const absent = ['Databases/system_ops.json', 'Databases/document_upload_drafts.db', 'Databases/document_upload_drafts.db-wal', 'Databases/document_upload_drafts.db-shm', 'Databases/webscraper', 'Databases/downloads/audio'];
  for (const relative of absent) requireThat(!stat(path.join(root, relative)), 'Archive-local storage requires explicit preservation before upgrade');
  const lock = stat(path.join(root, 'Databases/system_ops.json.lock'));
  requireThat(!lock || (lock.isFile() && lock.size === 0), 'Archive-local lock is not an empty regular file');
}

function releasedSource(cell, originalRun, upgradeRun, original) {
  const base = path.join(privateRoot, 'repair-sources', upgradeRun), root = confined(path.join(base, cell), privateRoot);
  const preparation = confined(path.join(base, 'preparation'), privateRoot);
  const gateFile = confined(path.join(preparation, 'gate.json'), preparation), gate = privateJson(gateFile);
  const completeFile = confined(path.join(preparation, 'complete.json'), preparation), complete = privateJson(completeFile);
  requireThat(gate.purpose === 'targeted-acceptance' && gate.status === 'RELEASED' && gate.dataPolicy === 'existing-profile-upgrade' && gate.originalRunId === originalRun && gate.runId === upgradeRun && /^[a-f0-9]{40}$/.test(gate.revision) && JSON.stringify(gate.cells) === JSON.stringify(['pg-single', 'pg-multi']), 'Matching released targeted-upgrade gate required');
  requireThat(complete.purpose === gate.purpose && complete.revision === gate.revision && complete.runId === upgradeRun && complete.noProfileFixtureRuntimeOrBrowserStarted === true && JSON.stringify(complete.cells) === JSON.stringify(gate.cells), 'Copy completion does not match the released upgrade');
  const archive = confined(path.join(preparation, `${gate.revision}.tar`), preparation);
  requireThat(complete.archive === archive && fileHash(archive) === complete.archiveSha256, 'Released archive bytes changed');
  const manifestFile = confined(path.join(preparation, `${cell}-source-manifest.json`), preparation), manifest = privateJson(manifestFile);
  requireThat(manifest.root === root && manifest.revision === gate.revision && Array.isArray(manifest.files) && manifest.files.length > 0 && manifest.fileCount === manifest.files.length, 'Source manifest does not match the upgrade cell');
  const seen = new Set();
  for (const entry of manifest.files) {
    requireThat(typeof entry.path === 'string' && entry.path !== '.' && !path.isAbsolute(entry.path) && path.normalize(entry.path) === entry.path && !entry.path.split(path.sep).includes('..') && !seen.has(entry.path), 'Invalid tracked source path');
    seen.add(entry.path);
    const file = path.join(root, entry.path), info = stat(file);
    requireThat(info && within(fs.realpathSync(file), root), 'Tracked source escaped its archive');
    if (entry.kind === 'symlink') requireThat(info.isSymbolicLink() && fs.readlinkSync(file) === entry.target, 'Tracked source link changed');
    else requireThat(entry.kind === 'file' && info.isFile() && !info.isSymbolicLink() && info.size === entry.bytes && fileHash(file) === entry.sha256, 'Tracked source bytes changed');
  }
  const reuseFile = confined(path.join(preparation, 'python-reuse.json'), preparation), reuse = privateJson(reuseFile);
  requireThat(reuse.pythonVenv === original.pythonVenv && reuse.reused === true && reuse.copiedOrModified === false, 'Upgrade must reuse the inspected private Python installation');
  for (const name of fs.readdirSync(privateRoot).filter(name => name.endsWith('.profile.private.json'))) {
    const profile = privateJson(path.join(privateRoot, name));
    requireThat(profile.sourceRoot && fs.realpathSync(profile.sourceRoot) !== root, 'Upgrade source is already bound to a profile');
  }
  const checked = inspectInputs({ 'source-root': root, 'python-venv': original.pythonVenv, 'source-commit': gate.revision });
  return { checked, proof: { gate: fileHash(gateFile), completion: fileHash(completeFile), sourceManifest: fileHash(manifestFile), pythonReuse: fileHash(reuseFile), archive: complete.archiveSha256 } };
}

function portFree(port) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', () => reject(Error('Original application port is occupied')));
    server.listen(port, '127.0.0.1', () => server.close(resolve));
  });
}

export async function launchUpgrade(action, cell, originalRun, upgradeRun) {
  requireThat(['backend', 'frontend'].includes(action) && ['pg-single', 'pg-multi'].includes(cell) && [originalRun, upgradeRun].every(value => /^[A-Za-z0-9][A-Za-z0-9._-]{0,70}$/.test(value || '')) && originalRun !== upgradeRun, 'Require backend|frontend, PG cell, original run and distinct upgrade run');
  const record = confined(path.join(privateRoot, `${originalRun}-${cell}.profile.private.json`), privateRoot), p = privateJson(record), spec = specFor(cell);
  requireThat(p.runId === originalRun && p.name === cell && p.spec?.mode === spec.mode && p.spec.engine === spec.engine && p.preparationStatus === 'complete' && p.preparationId && p.root === path.join(privateRoot, 'profiles', `tldw-onboarding-uat-${originalRun}-${cell}`), 'Original prepared profile identity mismatch');
  confined(p.root, privateRoot); confined(p.sourceRoot, privateRoot);
  requireThat([p.spec.api, p.spec.web].every(port => Number.isInteger(port) && port > 1024 && port <= 65535) && p.spec.api !== p.spec.web, 'Invalid original application ports');
  requireInitialized(p);
  const original = inspectInputs({ 'source-root': p.sourceRoot, 'python-venv': p.pythonVenv, 'source-commit': p.sourceCommit });
  requireThat(JSON.stringify(original.sourceHashes) === JSON.stringify(p.sourceHashes), 'Original source fingerprints changed');
  checkArchiveStorage(p.sourceRoot);
  const env = await runtimeEnv(p); // Validates the original holder, without relabelling its source.
  const { checked, proof } = releasedSource(cell, originalRun, upgradeRun, p);
  const role = spawnSync(p.python, [path.join(privateRoot, 'pg_role_adapter.py'), p.pgConfigPath, p.pgReceiptPath], { cwd: p.root, env, encoding: 'utf8', timeout: 30000, maxBuffer: 1024 * 1024 });
  requireThat(role.status === 0 && !role.error && !role.signal, 'PostgreSQL runtime role live verification failed; no process launched');
  const root = path.join(privateRoot, 'targeted-upgrades', upgradeRun, cell), bindingFile = path.join(root, 'binding.private.json');
  directoryPath(root);
  const binding = { purpose: 'upgrade-targeted-acceptance', cell, originalRun, upgradeRun, originalProfile: record, originalProfileHash: fileHash(record), originalInitializationHash: fileHash(path.join(p.root, 'initialized.private.json')), originalHolderHash: fileHash(p.pgReceiptPath), originalSourceCommit: p.sourceCommit, sourceRoot: checked.sourceRoot, sourceCommit: checked.sourceCommit, pythonVenv: checked.pythonVenv, proof };
  if (stat(bindingFile)) requireThat(JSON.stringify(privateJson(confined(bindingFile, root))) === JSON.stringify(binding), 'Existing upgrade binding changed; preserve its evidence');
  else checkArchiveStorage(checked.sourceRoot);
  const nextDistDir = `.next-upgrade-${upgradeRun}-${cell}`, dist = path.join(checked.frontend, nextDistDir), buildReceipt = path.join(root, 'frontend-build-root.private.json');
  if (action === 'frontend' && stat(dist)) requireThat(stat(dist).isDirectory() && !stat(dist).isSymbolicLink() && stat(buildReceipt) && privateJson(confined(buildReceipt, root)).path === dist, 'Refusing an unowned Next build directory');
  await portFree(action === 'backend' ? p.spec.api : p.spec.web);
  // All refusal checks precede writes; never touch origin profile/init/holder records.
  fs.mkdirSync(root, { recursive: true, mode: 0o700 });
  if (!stat(bindingFile)) fs.writeFileSync(bindingFile, JSON.stringify(binding, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
  if (action === 'frontend') { fs.mkdirSync(dist, { recursive: true, mode: 0o700 }); writePrivate(buildReceipt, { path: dist, upgradeRun, cell }); }
  env.PYTHONPATH = checked.sourcePaths.join(path.delimiter);
  const command = action === 'backend' ? checked.python : process.execPath;
  const args = action === 'backend' ? ['-m', 'uvicorn', 'tldw_Server_API.app.main:app', '--host', '127.0.0.1', '--port', String(p.spec.api)] : [checked.nextCli, 'dev', '--hostname', '127.0.0.1', '--port', String(p.spec.web)];
  const cwd = action === 'backend' ? p.root : checked.frontend;
  const attempt = `${action}-${Date.now()}-${crypto.randomUUID()}`, receipt = path.join(root, `${attempt}.process.private.json`), logPath = path.join(root, `${attempt}.private.log`);
  let state = { purpose: binding.purpose, action, cell, originalRun, upgradeRun, sourceRoot: checked.sourceRoot, sourceCommit: checked.sourceCommit, bindingHash: fileHash(bindingFile), command, args, cwd, logPath, startedAt: new Date().toISOString(), status: 'starting' };
  writePrivate(receipt, state);
  const fd = fs.openSync(logPath, 'wx', 0o600);
  let child;
  try { child = spawn(command, args, { cwd, env: action === 'backend' ? env : frontendEnv({ ...p, frontend: checked.frontend, nextDistDir }), stdio: ['ignore', fd, fd] }); }
  catch { fs.closeSync(fd); writePrivate(receipt, { ...state, status: 'spawn-failed', resultCode: 1, endedAt: new Date().toISOString() }); fail('Upgrade process failed to spawn; preserved private attempt'); }
  state = { ...state, status: 'started', pid: child.pid }; writePrivate(receipt, state);
  let requestedSignal;
  const forward = signal => { requestedSignal = signal; child.kill(signal); };
  const sigint = () => forward('SIGINT'), sigterm = () => forward('SIGTERM');
  process.on('SIGINT', sigint); process.on('SIGTERM', sigterm);
  let closed = false;
  const finish = (code, signal, failed = false) => {
    if (closed) return; closed = true; fs.closeSync(fd);
    process.removeListener('SIGINT', sigint); process.removeListener('SIGTERM', sigterm);
    const resultCode = failed || requestedSignal ? 1 : code ?? 1;
    writePrivate(receipt, { ...state, status: failed ? 'spawn-failed' : 'exited', code, signal, requestedSignal, resultCode, endedAt: new Date().toISOString() });
    process.exitCode = resultCode;
  };
  child.once('error', () => finish(null, null, true)); child.once('exit', (code, signal) => finish(code, signal));
  console.log(JSON.stringify({ purpose: binding.purpose, action, cell, upgradeRun, sourceCommit: checked.sourceCommit, pid: child.pid, receipt }));
  return { receipt, child };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { requireThat(process.argv.length === 6, 'Use backend|frontend CELL ORIGINAL_RUN UPGRADE_RUN'); await launchUpgrade(...process.argv.slice(2)); }
  catch (error) { reportFailure(error); process.exitCode = 1; }
}
