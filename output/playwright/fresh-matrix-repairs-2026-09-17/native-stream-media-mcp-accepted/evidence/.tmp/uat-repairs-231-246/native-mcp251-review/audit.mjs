import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// Read-only evidence/process audit. Never emit private JSON, environment or logs.
// The sole write is audit.json beside this script.
const out = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(out, '../../..');
const base = '.tmp/uat-repairs-231-246';
const native = `${base}/native-preparation`;
const matrix = '.tmp/uat-next-matrix-20260916';
const run = 'mcp251-fresh-targeted-20260917';
const revision = 'a7d3155a567afb25982eb360ea24b973cc3249c9';
const prep = `${matrix}/repair-sources/${run}/preparation`;
const profilePath = `${matrix}/${run}-pg-single.profile.private.json`;
const packet = 'output/playwright/fresh-matrix-repairs-2026-09-17/mcp-setup-fixture-reviewed';
const production = 'tldw_Server_API/app/core/AuthNZ/repos/mcp_hub_repo.py';
const expectedProduction = 'a63c6cf5c51ffa1a702894740240a4767e27f677ddf8512ed1131ad1edd02fda';
const files = new Map();
const hash = v => crypto.createHash('sha256').update(v).digest('hex');
const relative = p => path.relative(root, path.resolve(root, p));
function read(p) {
  const bytes = fs.readFileSync(path.resolve(root, p));
  files.set(relative(p), { path: relative(p), bytes: bytes.length, sha256: hash(bytes) });
  return bytes.toString('utf8');
}
const json = p => JSON.parse(read(p));
const checks = [];
const check = (name, value) => checks.push({ name, passed: Boolean(value) });
function nativeResult(name) {
  const text = read(`${native}/${name}`);
  const line = text.split('\n').find(l => l.startsWith('{'));
  return { text, data: line ? JSON.parse(line) : null };
}
function command(binary, args) {
  const r = spawnSync(binary, args, { encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 });
  // Do not surface stdout/stderr on failure: environment output may be private.
  return { status: r.status, errorCode: r.error?.code, text: r.status === 0 ? r.stdout.trim() : '' };
}
function processInfo(pid) {
  const r = command('/bin/ps', ['-ww', '-p', String(pid), '-o', 'pid=,ppid=,command=']);
  const m = r.text.match(/^(\d+)\s+(\d+)\s+([\s\S]*)$/);
  const cwd = command('/usr/sbin/lsof', ['-a', '-p', String(pid), '-d', 'cwd', '-Fn']);
  return { available: !!m, errorCode: r.errorCode, pid: m ? Number(m[1]) : pid,
    ppid: m ? Number(m[2]) : null, command: m?.[3] ?? '', cwd: cwd.text.split('\n').find(l => l.startsWith('n'))?.slice(1) };
}
function environment(pid) {
  const r = command('/bin/ps', ['eww', '-p', String(pid), '-o', 'command=']);
  return { available: r.status === 0, values: Object.fromEntries([...r.text.matchAll(/(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=([^\s]*)/g)].map(m => [m[1], m[2]])) };
}
function dbMatches(value, db) {
  try { const u = new URL(value); return u.hostname === db.host && Number(u.port || 5432) === Number(db.port) &&
    decodeURIComponent(u.username) === db.user && decodeURIComponent(u.password) === db.password && decodeURIComponent(u.pathname.slice(1)) === db.database; }
  catch { return false; }
}

const save = nativeResult('mcp251-save-packs.txt').data;
const sample = nativeResult('mcp251-sample-tool.txt').data;
const reload = nativeResult('mcp251-reloaded.txt').data;
const saveCode = read(`${native}/mcp251-save-packs.js`);
const sampleCode = read(`${native}/mcp251-validate.js`);
const reloadCode = read(`${native}/mcp251-reload.js`);
const packs = ['research', 'learning', 'writing', 'media_library', 'personal_knowledge'];
const applied = save.response.body, validated = sample.response.body, state = reload.response.state;
const same = (a,b) => JSON.stringify(a) === JSON.stringify(b);
check('normal Save packs click correlates actual POST apply response', saveCode.includes("method()==='POST'") && saveCode.includes('/api/v1/setup/first-run/mcp-tools/apply') && saveCode.includes("name:'Save packs'") && save.response.status === 200 && applied.status === 'applied');
check('default five read-only packs applied without addons or conflict', same(save.response.request.selected_pack_ids,packs) && same(applied.selected_pack_ids,packs) && applied.selected_addon_ids.length === 0 && applied.disabled_addons.length === 0 && applied.conflict === null);
check('apply creates profile one assignment one with 23 unique tools', applied.profile_id === 1 && applied.assignment_id === 1 && applied.effective_tool_count === 23 && new Set(applied.effective_tools).size === 23);
check('normal Run sample tool click correlates actual POST validation', sampleCode.includes("name:'Run sample tool'") && sampleCode.includes('/api/v1/setup/first-run/mcp-tools/validate') && sample.response.status === 200 && validated.status === 'validated');
check('sample uses built-in tools list and external tools remain disabled', validated.validation_state === 'built_in_passed' && validated.sample_tool_name === 'mcp.tools.list' && validated.external_status === 'not_enabled');
check('sample preserves IDs packs and tool count', validated.profile_id === applied.profile_id && validated.assignment_id === applied.assignment_id && same(validated.selected_pack_ids,packs) && validated.effective_tool_count === 23);
check('normal reload reads actual state GET', reloadCode.includes('await page.reload()') && reloadCode.includes("method()==='GET'") && reloadCode.includes('/api/v1/setup/first-run/state') && reload.response.status === 200 && reload.start > sample.at);
check('reloaded setup acknowledges completed MCP step but remains in progress', state.status === 'in_progress' && state.completed_steps.includes('mcp_tools') && state.mcp_tools.acknowledged === true);
check('reload retains complete MCP identity and validation receipt', ['profile_id','assignment_id','catalog_version','validation_state','effective_tool_count','validated_at','validation_message','last_validation_run_id','sample_tool_name','external_status'].every(k => state.mcp_tools[k] === validated[k]) && same(state.mcp_tools.selected_pack_ids,packs) && state.mcp_tools.selected_addon_ids.length === 0 && state.mcp_tools.confirmed_addon_ids.length === 0);
check('post-reload UI advances to First chat without claiming its completion', reload.ui.includes('heading "First chat"') && reload.ui.includes('button "Send test chat"') && !state.completed_steps.includes('first_chat'));
check('all native URLs belong to fresh single-user web port', [save,sample,reload].every(p => new URL(p.url).origin === 'http://127.0.0.1:18784'));

const supportingNames = ['mcp251-browser-open.txt','mcp251-browser-settled.txt','mcp251-local-path.txt',
  'mcp251-privacy-checked.txt','mcp251-provider-selected.txt','mcp251-providers-open.txt','mcp251-provider-form.txt',
  'mcp251-endpoint.txt','mcp251-endpoint-corrected.txt','mcp251-discovery.txt','mcp251-discovered.txt',
  'mcp251-provider-validated.txt','mcp251-providers-saved.txt','mcp251-provider-continue.txt',
  'mcp251-ingest-defaults.txt','mcp251-audio-step.txt','mcp251-audio-continue.txt',
  'mcp251-advanced-continue.txt','mcp251-tools-step.txt','mcp251-observer-installed.txt','mcp251-console.txt','mcp251-copy-result.txt'];
const supporting = Object.fromEntries(supportingNames.map(n => [n,nativeResult(n)]));
check('wizard starts at setup path in new profile', supporting['mcp251-browser-settled.txt'].text.includes('Choose your setup path'));
check('actual local provider endpoint validates and saves through wizard', supporting['mcp251-providers-saved.txt'].data.body.includes('http://127.0.0.1:9099/v1') && supporting['mcp251-providers-saved.txt'].data.body.includes('Provider validation is ready.') && supporting['mcp251-providers-saved.txt'].data.body.includes('paragraph: Saved'));
check('balanced ingest and deferred optional audio/advanced defaults retained', supporting['mcp251-ingest-defaults.txt'].text.includes('radio "balanced" [checked]') && supporting['mcp251-audio-continue.txt'].data.ui.includes('radio "Skip for now" [checked]') && supporting['mcp251-advanced-continue.txt'].data.ui.includes('radio "Defer" [checked]'));
check('MCP starts with five checked packs and disabled Continue', (supporting['mcp251-tools-step.txt'].text.match(/checkbox .*\[checked\]/g) ?? []).length === 5 && supporting['mcp251-tools-step.txt'].text.includes('button "Continue" [disabled]'));

const gate = json(`${native}/mcp251-targeted-gate.json`), copiedGate = json(`${prep}/gate.json`), completion = json(`${prep}/complete.json`);
const profile = json(profilePath), safeProfile = json(path.join(profile.root,'safe-profile-manifest.json'));
const initialized = json(path.join(profile.root,'initialized.private.json'));
const initProcess = json(path.join(profile.root,'initialize-process.private.json'));
const preflight = json(`${native}/mcp251-preflight.txt`);
const holder = json(profile.pgReceiptPath), holderProcess = json(`${matrix}/holders/${run}-pg-single/process.private.json`);
json(profile.pgConfigPath); // Hash only; no connection or provisioning operation.
check('released gate is issue-specific with only single runtime enabled', gate.status === 'RELEASED' && gate.revision === revision && gate.runId === run && same(gate.runtimeCells,['pg-single']) && gate.fullMatrixReleased === false && copiedGate.inputSha256 === files.get(`${native}/mcp251-targeted-gate.json`).sha256);
check('copy completion binds both source copies and exact revision', completion.revision === revision && completion.runId === run && same(completion.cells,['pg-single','pg-multi']) && completion.noProfileFixtureRuntimeOrBrowserStarted === true);
check('only fresh single profile was instantiated', !fs.existsSync(path.resolve(root,`${matrix}/${run}-pg-multi.profile.private.json`)) && profile.runId === run && profile.name === 'pg-single');
check('fresh profile uses required PostgreSQL single-user ports', profile.spec.mode === 'single_user' && profile.spec.engine === 'postgresql' && profile.spec.api === 18704 && profile.spec.web === 18784 && profile.preparationStatus === 'complete');
check('initialization completed before native wizard with distinct fresh root', initialized.status === 'completed' && initialized.code === 0 && initialized.at < save.start && profile.root.endsWith(`tldw-onboarding-uat-${run}-pg-single`));
check('profile/preflight source bindings agree', profile.sourceCommit === revision && preflight.sourceCommit === revision && profile.sourceRoot === preflight.sourceRoot && profile.python === preflight.python && profile.frontend === preflight.frontend && profile.sourceRoot.endsWith(`/repair-sources/${run}/pg-single`));
check('official PostgreSQL auth and content fixtures are held', holder.status === 'held' && holder.pid === 73403 && holderProcess.pid === holder.pid && holder.auth_fixture === 'pg_temp_db' && holder.content_fixture === 'pg_temp_db_session' && holder.run_id === run && holder.source_commit === revision);
check('runtime role is direct-login and restricted', holder.runtime_role.login === true && ['superuser','bypassrls','inherit','createdb','createrole','replication'].every(k => holder.runtime_role[k] === false) && holder.runtime_role.memberships === 0);
check('separate auth/content fixtures use the same restricted runtime role', holder.auth.database !== holder.content.database && holder.auth.user === holder.runtime_role.name && holder.content.user === holder.runtime_role.name && holder.provisioning.user !== holder.runtime_role.name);
check('safe preparation manifest records reused dependencies and no test/provider flags', safeProfile.reusedDependencies === true && safeProfile.providerEnvOverrides === 0 && safeProfile.testModeFlags === 0);

const sourceManifest = json(`${prep}/pg-single-source-manifest.json`);
json(`${prep}/pg-multi-source-manifest.json`);
const dependencies = json(`${prep}/pg-single-dependencies.json`); json(`${prep}/pg-multi-dependencies.json`);
const pythonReuse = json(`${prep}/python-reuse.json`);
check('Python reuse receipt identity is preserved', hash(read(pythonReuse.receipt)) === pythonReuse.receiptSha256 && pythonReuse.reused === true && pythonReuse.copiedOrModified === false && pythonReuse.pythonVenv === profile.pythonVenv);
const dependencyHashes = dependencies.packages.map(e => ({path:e.path,matches:hash(read(path.join(profile.sourceRoot,e.path)))===e.sha256}));
check('copied dependency package identities match and builds were not copied', dependencyHashes.every(e=>e.matches) && dependencies.reusedInstallation === true && dependencies.noNextBuildCopied === true);
const selectedSource = [production,'tldw_Server_API/app/main.py','apps/tldw-frontend/next.config.mjs','apps/tldw-frontend/scripts/validate-networking-config.mjs'];
const sourceHashes = selectedSource.map(p => { const entry=sourceManifest.files.find(f=>f.path===p);const current=hash(read(path.join(profile.sourceRoot,p)));return {path:p,sha256:current,matchesManifest:current===entry?.sha256}; });
check('selected runtime source matches archive manifest and reviewed MCP repair', sourceManifest.revision === revision && sourceHashes.every(e=>e.matchesManifest) && sourceHashes.find(e=>e.path===production).sha256===expectedProduction);
check('workspace and retained implementation snapshot match reviewed MCP source', hash(read(production))===expectedProduction && hash(read(`${packet}/mcp251/review-snapshot/${production}`))===expectedProduction);
// Stream the large archive hash without retaining its contents in memory.
const archiveHash = crypto.createHash('sha256'); let archiveBytes=0;
for await(const chunk of fs.createReadStream(completion.archive)){archiveHash.update(chunk);archiveBytes+=chunk.length;}
const actualArchiveHash=archiveHash.digest('hex');
files.set(relative(completion.archive),{path:relative(completion.archive),bytes:archiveBytes,sha256:actualArchiveHash});
check('copied source archive matches recorded completion hash', actualArchiveHash===completion.archiveSha256);

const previous = json(`${base}/upgrade254-native-review/audit.json`);
const oldProfiles = previous.results.map(old=>{
  const oldProfilePath=`${matrix}/${old.originalRun}-${old.cell}.profile.private.json`;
  const p=json(oldProfilePath);const initPath=path.join(p.root,'initialized.private.json');read(initPath);read(p.pgReceiptPath);
  const oldHolder=json(p.pgReceiptPath);
  return {cell:old.cell,profileUnchanged:files.get(oldProfilePath).sha256===old.originalRecordHashes.profile,
    initializationUnchanged:files.get(relative(initPath)).sha256===old.originalRecordHashes.initialization,
    holderUnchanged:files.get(relative(p.pgReceiptPath)).sha256===old.originalRecordHashes.holder,
    freshRootDistinct:p.root!==profile.root,
    freshDatabasesDistinct:oldHolder.auth.database!==holder.auth.database&&oldHolder.content.database!==holder.content.database};
});
check('original completed targeted profile initialization and holder records remain unchanged', oldProfiles.every(p=>p.profileUnchanged&&p.initializationUnchanged&&p.holderUnchanged&&p.freshRootDistinct&&p.freshDatabasesDistinct));

const runtime=[];
for(const action of ['backend','frontend']){
  const record=json(path.join(profile.root,`${action}-process.private.json`));const current=processInfo(record.pid);const env=environment(record.pid);
  const matchesCommand=current.command===`${record.command} ${record.args.join(' ')}`;
  const listeners=command('/usr/sbin/lsof',['-nP',`-iTCP:${action==='backend'?profile.spec.api:profile.spec.web}`,'-sTCP:LISTEN','-Fp']).text.split('\n').filter(s=>/^p\d+$/.test(s)).map(s=>Number(s.slice(1)));
  const listenerChecks=listeners.map(pid=>{const p=processInfo(pid);return {pid,ppid:p.ppid,owned:pid===record.pid||p.ppid===record.pid,cwd:p.cwd};});
  const safeEnv=Object.fromEntries(['PYTHONPATH','TLDW_CONFIG_FILE','TLDW_ENV_FILE','AUTH_MODE','TLDW_USER_DB_BACKEND','TLDW_CONTENT_DB_BACKEND','TLDW_NEXT_DIST_DIR','TLDW_INTERNAL_API_ORIGIN'].filter(k=>k in env.values).map(k=>[k,env.values[k]]));
  const envBound=action==='backend'
    ? env.values.PYTHONPATH===profile.sourcePaths.join(path.delimiter)&&env.values.AUTH_MODE==='single_user'&&env.values.TLDW_USER_DB_BACKEND==='postgresql'&&env.values.TLDW_CONTENT_DB_BACKEND==='postgresql'&&dbMatches(env.values.DATABASE_URL,holder.auth)&&dbMatches(env.values.TLDW_CONTENT_PG_DSN,holder.content)
    : env.values.TLDW_INTERNAL_API_ORIGIN==='http://127.0.0.1:18704'&&env.values.TLDW_NEXT_DIST_DIR===profile.nextDistDir;
  const log=read(record.logPath);files.get(relative(record.logPath)).mutableRuntimeLog=true;
  const startup=action==='backend'?log.includes('Application startup complete'): /Ready in|✓ Ready/.test(log);
  check(`live ${action} matches recorded command cwd and owned listener`,current.available&&matchesCommand&&current.cwd===record.cwd&&listenerChecks.length>0&&listenerChecks.every(p=>p.owned));
  check(`live ${action} environment uses fresh source and intended runtime bindings`,env.available&&envBound);
  check(`${action} native startup marker exists`,startup);
  runtime.push({action,pid:record.pid,startedAt:record.startedAt,processReadable:current.available,errorCode:current.errorCode,
    commandSha256:hash(current.command),matchesCommand,cwd:current.cwd,matchesCwd:current.cwd===record.cwd,
    safeEnv,environmentBindingsMatch:envBound,listeners:listenerChecks,startupMarkerPresent:startup});
}
const liveHolder=processInfo(holder.pid);
const liveHolderEnv=environment(holder.pid);
check('official fixture holder remains alive in the recorded source context',liveHolder.available&&liveHolder.command.includes('pytest')&&liveHolder.command.includes('test_hold_official_pg.py')&&liveHolderEnv.available&&liveHolderEnv.values.MATRIX_SOURCE_ROOT===profile.sourceRoot&&liveHolderEnv.values.MATRIX_SOURCE_COMMIT===revision&&liveHolderEnv.values.MATRIX_RUN_ID===run&&liveHolderEnv.values.MATRIX_CELL==='pg-single');

read(`${packet}/README.md`);read(`${packet}/review251/REVIEW.md`);
read('backlog/tasks/task-13260.193 - Fix-PostgreSQL-MCP-setup-permission-profile-lookup-UAT-251.md');
read(`${matrix}/matrix-launcher.mjs`);read(`${matrix}/pg_role_adapter.py`);
read(path.relative(root,fileURLToPath(import.meta.url)));
if(fs.existsSync(path.join(out,'REVIEW.md')))read(path.relative(root,path.join(out,'REVIEW.md')));
const failed=checks.filter(c=>!c.passed);
const audit={at:new Date().toISOString(),task:'TASK-13260.193',finding:'UAT251',verdict:failed.length?'REMAINING_GAP':'CLEAR',
  scope:'Issue-specific fresh PostgreSQL single-user MCP Save packs and reload acceptance',runId:run,source:revision,checks,
  native:{save:{at:save.at,status:save.response.status,profileId:applied.profile_id,assignmentId:applied.assignment_id,packs,toolCount:applied.effective_tool_count},
    sample:{at:sample.at,status:sample.response.status,validationState:validated.validation_state,runId:validated.last_validation_run_id,tool:validated.sample_tool_name,externalStatus:validated.external_status},
    reload:{at:reload.at,status:reload.response.status,stateStatus:state.status,completedSteps:state.completed_steps,mcpTools:state.mcp_tools}},
  sourceHashes,dependencyHashes,archive:{sha256:actualArchiveHash,bytes:archiveBytes},
  freshness:{root:profile.root,initializedAt:initialized.at,initializationStatus:initialized.status,originalProfiles:oldProfiles},
  officialFixture:{pid:holder.pid,status:holder.status,authFixture:holder.auth_fixture,contentFixture:holder.content_fixture,
    roleFlags:Object.fromEntries(['login','superuser','bypassrls','inherit','createdb','createrole','replication','memberships'].map(k=>[k,holder.runtime_role[k]])),
    processReadable:liveHolder.available,commandSha256:hash(liveHolder.command),safeEnv:Object.fromEntries(['MATRIX_SOURCE_ROOT','MATRIX_SOURCE_COMMIT','MATRIX_RUN_ID','MATRIX_CELL','MATRIX_PYTHON_VENV'].map(k=>[k,liveHolderEnv.values[k]]))},runtime,
  auditCorrections:['First audit expected initialization status complete; the actual receipt schema uses completed, with exit code zero.', 'First audit expected holder source path in command arguments; official holder binds source and run through MATRIX_SOURCE_ROOT/MATRIX_RUN_ID environment fields, now independently checked.'],
  limits:['The source manifest covers both copied cells, but only pg-single was instantiated for this acceptance.',
    'Original profile/init/holder hashes and distinct database identities were checked; this is not whole-database equivalence.',
    'Python/JavaScript dependencies were reused. This is a fresh application profile and official PostgreSQL fixture, not a clean OS installation.',
    'Native acceptance covers five default built-in read-only packs and mcp.tools.list. External addons, audio and first-chat completion were not tested.',
    'Setup remains in_progress at First chat. No full wizard completion or fresh full 48-cell matrix acceptance is claimed.',
    'Live role-to-DSN bindings match restricted official fixture receipts; no direct role query or database mutation was performed by this reviewer.',
    'Initial process inspection was sandbox-blocked (EPERM); a safe-projection read-only escalation supplies actual process checks.',
    'Prior independently reviewed implementation test results are retained context; no product tests were rerun here.'],
  reviewedFiles:[...files.values()].sort((a,b)=>a.path.localeCompare(b.path))};
fs.writeFileSync(path.join(out,'audit.json'),`${JSON.stringify(audit,null,2)}\n`);
console.log(JSON.stringify({verdict:audit.verdict,passed:checks.length-failed.length,failed,reviewedFiles:files.size,auditSha256:hash(fs.readFileSync(path.join(out,'audit.json')))},null,2));
if(failed.length)process.exitCode=1;
