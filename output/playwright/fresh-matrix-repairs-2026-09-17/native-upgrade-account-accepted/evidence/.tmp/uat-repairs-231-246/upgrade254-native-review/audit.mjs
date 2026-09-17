import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import http from 'node:http';
import { spawnSync } from 'node:child_process';
const out = '.tmp/uat-repairs-231-246/upgrade254-native-review';
const matrix = path.resolve('.tmp/uat-next-matrix-20260916');
const native = '.tmp/uat-repairs-231-246/native-targeted';
const originalRun = 'repairs231-250-targeted-20260917';
const upgradeRun = 'repairs251-254-upgrade2-20260917';
const candidate = 'a7d3155a567afb25982eb360ea24b973cc3249c9';
const started = new Date().toISOString();
const problems = [], evidence = [];
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const read = file => fs.readFileSync(file);
const capture = file => { const data = read(file); evidence.push({ path: path.relative(process.cwd(), file), bytes: data.length, sha256: sha(data) }); return data; };
const load = file => JSON.parse(capture(file));
const check = (condition, label) => { if (!condition) problems.push(label); return Boolean(condition); };
const command = (name, args) => { const r = spawnSync(name, args, { encoding: 'utf8', maxBuffer: 2 * 1024 * 1024 }); if (r.error || r.status !== 0) throw Error(`${name} inspection failed: ${r.error?.code ?? r.status}`); return r.stdout; };
const processInfo = pid => {
  const raw = command('/bin/ps', ['-ww', '-p', String(pid), '-o', 'pid=,ppid=,command=']).trim();
  const match = /^(\d+)\s+(\d+)\s+([\s\S]+)$/.exec(raw);
  const cwd = command('/usr/sbin/lsof', ['-a', '-p', String(pid), '-d', 'cwd', '-Fn']).split('\n').find(line => line.startsWith('n'))?.slice(1);
  return { pid: Number(match[1]), ppid: Number(match[2]), cwd, commandText: match[3], commandSha256: sha(match[3]) };
};
const envField = (raw, key) => new RegExp(`(?:^|\\s)${key}=([^\\s]*)`).exec(raw)?.[1];
const listeners = port => [...new Set(command('/usr/sbin/lsof', ['-nP', `-iTCP:${port}`, '-sTCP:LISTEN', '-Fp']).split('\n').filter(line => /^p\d+$/.test(line)).map(line => Number(line.slice(1))))];
const requestStatus = port => new Promise(resolve => {
  const req = http.get({ hostname: '127.0.0.1', port, path: '/', timeout: 10000 }, res => { res.resume(); resolve({ port, path: '/', status: res.statusCode }); });
  req.on('timeout', () => req.destroy()); req.on('error', error => resolve({ port, path: '/', error: error.code ?? 'request-error' }));
});
const helperHash = sha(capture(path.join(matrix, 'matrix-upgrade.mjs')));
const launcherHash = sha(capture(path.join(matrix, 'matrix-launcher.mjs')));
const preparation = path.join(matrix, 'repair-sources', upgradeRun, 'preparation');
const gate = load(path.join(preparation, 'gate.json'));
const complete = load(path.join(preparation, 'complete.json'));
const reuse = load(path.join(preparation, 'python-reuse.json'));
const proofHash = name => sha(read(path.join(preparation, name)));
check(gate.status === 'RELEASED' && gate.revision === candidate && gate.originalRunId === originalRun && gate.runId === upgradeRun && gate.dataPolicy === 'existing-profile-upgrade', 'Released gate provenance mismatch');
check(complete.revision === candidate && complete.runId === upgradeRun && complete.noProfileFixtureRuntimeOrBrowserStarted === true, 'Copy completion provenance mismatch');
const results = [];
for (const [cell, backendPid, frontendPid, holderPid] of [['pg-single', 64579, 64737, 18859], ['pg-multi', 64596, 64765, 18878]]) {
  const profilePath = path.join(matrix, `${originalRun}-${cell}.profile.private.json`);
  const p = load(profilePath), initializedPath = path.join(p.root, 'initialized.private.json');
  const initialized = load(initializedPath), holder = load(p.pgReceiptPath);
  const root = path.join(matrix, 'targeted-upgrades', upgradeRun, cell);
  const bindingPath = path.join(root, 'binding.private.json'), binding = load(bindingPath), bindingHash = sha(read(bindingPath));
  const sourceManifestFile = path.join(preparation, `${cell}-source-manifest.json`), sourceManifest = load(sourceManifestFile);
  const sourceRoot = path.join(matrix, 'repair-sources', upgradeRun, cell);
  const build = load(path.join(root, 'frontend-build-root.private.json'));
  const expectedDistDir = `.next-live-tier-upgrade-${upgradeRun}-${cell}`;
  const unchanged = {
    profile: sha(read(profilePath)) === binding.originalProfileHash,
    initialization: sha(read(initializedPath)) === binding.originalInitializationHash,
    holder: sha(read(p.pgReceiptPath)) === binding.originalHolderHash,
  };
  check(Object.values(unchanged).every(Boolean), `${cell}: original records differ from immutable binding`);
  const oldBindingPath = path.join(matrix, 'targeted-upgrades', 'repairs251-253-upgrade-20260917', cell, 'binding.private.json');
  const oldBinding = load(oldBindingPath);
  const unchangedSinceFailedRun = { profile: oldBinding.originalProfileHash === binding.originalProfileHash, initialization: oldBinding.originalInitializationHash === binding.originalInitializationHash, holder: oldBinding.originalHolderHash === binding.originalHolderHash };
  check(Object.values(unchangedSinceFailedRun).every(Boolean) && oldBinding.upgradeHelperHash === '1d3254bc0c7403770348132974b42ff28568051fc5e3a06a829f920b410c01e6', `${cell}: originals changed since failed upgrade binding`);

  check(initialized.status === 'completed' && initialized.code === 0 && initialized.preparationHash === sha(JSON.stringify(p)), `${cell}: original initialization identity mismatch`);
  check(binding.originalRun === originalRun && binding.upgradeRun === upgradeRun && binding.originalProfile === profilePath && binding.sourceRoot === sourceRoot && binding.sourceCommit === candidate && binding.originalSourceCommit === p.sourceCommit, `${cell}: binding identity mismatch`);
  check(binding.launcherHash === launcherHash && binding.upgradeHelperHash === helperHash, `${cell}: runtime helper byte mismatch`);
  check(binding.proof.gate === proofHash('gate.json') && binding.proof.completion === proofHash('complete.json') && binding.proof.sourceManifest === sha(read(sourceManifestFile)) && binding.proof.pythonReuse === proofHash('python-reuse.json') && binding.proof.archive === complete.archiveSha256, `${cell}: copied-source proof mismatch`);
  check(sourceManifest.root === sourceRoot && sourceManifest.revision === candidate && reuse.pythonVenv === p.pythonVenv && reuse.reused === true && reuse.copiedOrModified === false, `${cell}: source/Python origin mismatch`);
  const sourceChecks = [];
  for (const relative of ['apps/tldw-frontend/next.config.mjs', 'tldw_Server_API/app/main.py', 'apps/tldw-frontend/scripts/validate-networking-config.mjs']) {
    const entry = sourceManifest.files.find(item => item.path === relative), actual = sha(capture(path.join(sourceRoot, relative)));
    sourceChecks.push({ path: relative, sha256: actual, matchesManifest: entry?.sha256 === actual });
  }
  check(sourceChecks.every(item => item.matchesManifest), `${cell}: entry/config source bytes changed`);
  check(sourceChecks[0].sha256 === '55f31677133e8fc03759d961e6bca7534c1a6d007fd0554723ee54d5bd5e3188', `${cell}: actual Next guard differs from reviewed config`);
  const holderLive = processInfo(holderPid);
  check(holder.pid === holderPid && holder.status === 'held' && holder.run_id === originalRun && holder.source_root === p.sourceRoot && holder.source_commit === p.sourceCommit, `${cell}: original fixture holder identity mismatch`);
  const processes = [];
  for (const [action, pid] of [['backend', backendPid], ['frontend', frontendPid]]) {
    const name = fs.readdirSync(root).find(file => file.startsWith(action + '-') && file.endsWith('.process.private.json'));
    const receiptPath = path.join(root, name), receipt = load(receiptPath), info = processInfo(pid);
    const actualEnv = command('/bin/ps', ['eww', '-p', String(pid), '-o', 'command=']);
    const safeEnv = Object.fromEntries(['PYTHONPATH', 'TLDW_CONFIG_FILE', 'TLDW_ENV_FILE', 'USER_DB_BASE_DIR', 'TLDW_NEXT_DIST_DIR', 'TLDW_INTERNAL_API_ORIGIN', 'AUTH_MODE'].map(key => [key, envField(actualEnv, key)]));
    const expectedCwd = action === 'backend' ? p.root : path.join(sourceRoot, 'apps/tldw-frontend');
    const exactReceiptCommand = info.commandText === [receipt.command, ...receipt.args].join(' ');
    const argsMatch = info.commandText.endsWith(receipt.args.join(' '));
    check(receipt.pid === pid && receipt.status === 'started' && receipt.bindingHash === bindingHash && receipt.sourceCommit === candidate && receipt.sourceRoot === sourceRoot && receipt.cwd === expectedCwd, `${cell}/${action}: receipt mismatch`);
    check(info.cwd === expectedCwd && argsMatch, `${cell}/${action}: live command/cwd mismatch`);
    if (action === 'backend') {
      check(safeEnv.PYTHONPATH?.split(path.delimiter)[0] === sourceRoot && safeEnv.TLDW_CONFIG_FILE === p.configPath && safeEnv.TLDW_ENV_FILE === p.envPath && safeEnv.USER_DB_BASE_DIR === p.userDatabasesDir, `${cell}: actual backend source/data environment mismatch`);
      check(listeners(p.spec.api).includes(pid), `${cell}: API listener ownership mismatch`);
    } else {
      check(safeEnv.TLDW_NEXT_DIST_DIR === expectedDistDir && safeEnv.TLDW_INTERNAL_API_ORIGIN === `http://127.0.0.1:${p.spec.api}`, `${cell}: actual frontend environment mismatch`);
      check(build.path === path.join(expectedCwd, expectedDistDir) && build.upgradeRun === upgradeRun && build.cell === cell, `${cell}: Next build ownership mismatch`);
    }
    const logBytes = capture(receipt.logPath), log = logBytes.toString();
    const startupObserved = action === 'backend' ? /Application startup complete/.test(log) : /Ready in/.test(log);
    check(startupObserved, `${cell}/${action}: successful startup line not found`);
    const listenerPids = action === 'frontend' ? listeners(p.spec.web) : [pid];
    const listenerOwnership = listenerPids.map(listenerPid => { const item = processInfo(listenerPid); return { pid: item.pid, ppid: item.ppid, cwd: item.cwd, commandSha256: item.commandSha256, owned: listenerPid === pid || item.ppid === pid }; });
    check(listenerOwnership.length > 0 && listenerOwnership.every(item => item.owned), `${cell}/${action}: listener parent mismatch`);
    processes.push({ action, pid, ppid: info.ppid, receipt: path.relative(process.cwd(), receiptPath), receiptHash: sha(read(receiptPath)), commandSha256: info.commandSha256, receiptExecutable: receipt.command, receiptArgs: receipt.args, exactReceiptCommand, argsMatch, cwd: info.cwd, safeEnv, startedAt: receipt.startedAt, startupObserved, listenerOwnership });
  }
  results.push({ cell, originalRun, upgradeRun, sourceCommit: candidate, sourceRoot, profileRoot: p.root, originalSourceRoot: p.sourceRoot, originalSourceCommit: p.sourceCommit, bindingHash, originalRecordHashes: { profile: binding.originalProfileHash, initialization: binding.originalInitializationHash, holder: binding.originalHolderHash }, unchanged, unchangedSinceFailedRun, oldBindingHash: sha(read(oldBindingPath)), initializationAt: initialized.at, originalInitializationCompleted: initialized.status === 'completed', sourceManifestHash: sha(read(sourceManifestFile)), sourceManifestFileCount: sourceManifest.fileCount, sourceChecks, holder: { pid: holderPid, ppid: holderLive.ppid, status: holder.status, commandSha256: holderLive.commandSha256 }, ports: { api: p.spec.api, web: p.spec.web }, buildDirectory: build.path, processes });
}
const parseNative = name => {
  const text = capture(path.join(native, name)).toString();
  const match = /### Result\s*\n([\s\S]*?)(?=\n### |$)/.exec(text);
  let result; try { result = match ? JSON.parse(match[1].trim()) : undefined; } catch {}
  return { text, result };
};
const media = parseNative('pg-single/media-upgraded-read-v2.txt').result;
const mediaObserverError = parseNative('pg-single/media-upgraded-read.txt').text;
const mediaBody = media.response.body;
check(media.response.status === 200 && mediaBody.media_id === 1 && mediaBody.content.text.length === 1914 && mediaBody.versions[0].created_at === '2026-09-17T20:32:22.772000Z', 'Media 1 retained readback mismatch');
const oldBob = parseNative('pg-multi/bob-ordinary-reloaded.txt').result;
const oldBobTerminal = parseNative('pg-multi/bob-ordinary-terminal.txt').result;
const newBob = parseNative('pg-multi/bob-upgrade-captured.txt').result;
const navigation = parseNative('pg-multi/bob-upgrade-navigate.txt').text;
const settled = parseNative('pg-multi/bob-upgrade-settled.txt').text;
const blankCurrent = parseNative('pg-multi/upgrade-current.txt').text;
const blankTabs = parseNative('pg-multi/upgrade-tabs.txt').text;
const logout = parseNative('pg-multi/bob-upgrade-logout.txt').text;
const aliceLogin = parseNative('pg-multi/alice-upgrade-login.txt').text;
const chatId = 'e3755001-1cd4-40a4-aaaf-edd4b67b3418';
const messageMap = events => new Map(events.filter(event => event.status === 200 && Array.isArray(event.body?.messages)).flatMap(event => event.body.messages).filter(message => message.conversation_id === chatId).map(message => [message.id, message]));
const before = messageMap(oldBob.canonical), after = messageMap(newBob.events);
const compared = [...before].map(([id, old]) => ({ id, sender: old.sender, contentSha256: sha(old.content), timestamp: old.timestamp, exactRecordMatch: JSON.stringify(old) === JSON.stringify(after.get(id)) }));
check(compared.length === 2 && compared.every(item => item.exactRecordMatch), 'Pre-upgrade canonical Bob records differ');
const assistant = [...after.values()].find(message => message.sender === 'assistant');
const oldAssistantDisplayed = oldBobTerminal.body.includes(assistant.content);
const identities = newBob.events.filter(event => event.event === 'identity').map(event => ({ at: event.at, status: event.status, id: event.identity?.id, username: event.identity?.username }));
const currentChatResponses = newBob.events.filter(event => event.status === 200 && Array.isArray(event.body?.messages) && event.body.messages.some(message => message.conversation_id === chatId));
check(after.size === 3 && oldAssistantDisplayed && Date.parse(assistant.timestamp) < Date.parse(results[1].processes[0].startedAt), 'Original assistant display/readback evidence missing');
check(identities.some(item => item.status === 200 && item.id === 3 && item.username === 'bob'), 'Authenticated Bob identity evidence missing');
check(newBob.url === `http://127.0.0.1:18783/chat?chatId=${chatId}` && currentChatResponses.length > 0, 'Bob canonical upgraded readback missing');
const loginRequests = newBob.events.filter(event => event.event === 'request' && /\/auth\/(?:login|token)(?:\?|$)/.test(event.url ?? ''));
const httpReadback = await Promise.all(results.map(result => requestStatus(result.ports.web)));
check(httpReadback.every(item => item.status === 200), 'Current frontend root did not return HTTP 200');
const audit = { verdict: problems.length ? 'ACTIONABLE_GAPS' : 'CLEAR', startedAt: started, completedAt: new Date().toISOString(), task: 'TASK13260.196', finding: 254, scope: 'Corrected actual startup plus bounded original-data readback', candidate, helperHash, launcherHash, results, native: { media: { at: media.at, status: media.response.status, id: mediaBody.media_id, contentLength: mediaBody.content.text.length, contentSha256: sha(mediaBody.content.text), versionCreatedAt: mediaBody.versions[0].created_at, oldObserverErrorPreserved: mediaObserverError.includes('### Error') }, bob: { at: newBob.at, chatId, identities, canonical200Responses: currentChatResponses.length, messagesBefore: before.size, messagesAfter: after.size, exactOldCanonicalRecords: compared, assistant: { id: assistant.id, timestamp: assistant.timestamp, contentSha256: sha(assistant.content), originalTerminalDisplayedSameContent: oldAssistantDisplayed }, automationHadBlankTab: blankCurrent.includes('about:blank') && blankTabs.includes('about:blank'), navigationTargetsOriginalChat: navigation.includes(chatId), settledShowsOriginalChat: settled.includes(chatId), loginRequestsInCapturedSequence: loginRequests.length, laterExplicitLogoutEvidencePresent: logout.length > 0, laterAliceLoginEvidencePresent: aliceLogin.length > 0 } }, httpReadback, evidence, limits: ['No continuous-same-tab claim: automation blank tab was recovered by normal navigation in the existing authenticated browser context.', 'Only original profile/init/holder byte identities and selected native records were verified; no full database equivalence or full/fresh UAT claim.', 'Source proof manifests and entry/config bytes were verified; all 35,159 tracked source files per cell were not independently rehashed in this audit.', 'No credential values or full private logs retained; process environment inspection emitted only the named safe fields.', 'HTTP 401 health observations alone establish authentication boundary, not readiness.'], problems };
fs.writeFileSync(path.join(out, 'audit.json'), JSON.stringify(audit, null, 2) + '\n', { mode: 0o600 });
console.log(JSON.stringify({ verdict: audit.verdict, problems, currentFrontendHttp: httpReadback, unchanged: results.map(result => ({ cell: result.cell, unchanged: result.unchanged, pid: result.processes.map(item => item.pid), holder: result.holder.pid })), media: audit.native.media, bob: { canonicalRecordsExactlyPreserved: compared.length, messagesAfter: after.size, assistantPreviouslyDisplayed: oldAssistantDisplayed, blankTabRecovered: audit.native.bob.automationHadBlankTab }, evidenceFiles: evidence.length }, null, 2));
if (problems.length) process.exitCode = 1;
