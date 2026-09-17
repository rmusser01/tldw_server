// Private Stage4 adapter. No archive, dependency installation, or UAT execution.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import net from 'node:net';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawn, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';

process.umask(0o077);
export const privateRoot = path.dirname(fileURLToPath(import.meta.url));
const profilesRoot = path.join(privateRoot, 'profiles');
const specs = {
  'sqlite-single': { mode: 'single_user', engine: 'sqlite' },
  'sqlite-multi': { mode: 'multi_user', engine: 'sqlite' },
  'pg-single': { mode: 'single_user', engine: 'postgresql' },
  'pg-multi': { mode: 'multi_user', engine: 'postgresql' },
};
const baseKeys = ['PATH', 'HOME', 'USER', 'LOGNAME', 'SHELL', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ'];
export const baseEnv = () => Object.fromEntries(baseKeys.filter(k => typeof process.env[k] === 'string').map(k => [k, process.env[k]]));
export const json = p => JSON.parse(fs.readFileSync(p, 'utf8'));
export const writePrivate = (p, value) => {
  fs.mkdirSync(path.dirname(p), { recursive: true, mode: 0o700 });
  fs.writeFileSync(p, typeof value === 'string' ? value : JSON.stringify(value, null, 2) + '\n', { mode: 0o600 });
  fs.chmodSync(p, 0o600);
};
const sha = data => crypto.createHash('sha256').update(data).digest('hex');
const inside = (p, root) => p === root || p.startsWith(root + path.sep);
export const fail = message => { throw Object.assign(Error(message), { safeLauncherMessage: true }); };
export const reportFailure = error => console.error(error?.safeLauncherMessage ? error.message : 'Action failed; inspect the private inputs or log. No raw exception details are printed.');
export function parseOptions(args) {
  const options = {};
  const allowed = new Set(['run-id', 'source-root', 'python-venv', 'source-commit', 'api-port', 'web-port', 'pg-config', 'pg-receipt']);
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i]?.slice(2), value = args[i + 1];
    if (!args[i]?.startsWith('--') || !allowed.has(key) || !value || value.startsWith('--') || Object.hasOwn(options, key)) fail('Invalid or duplicate launcher option');
    options[key] = value;
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,90}$/.test(options['run-id'] || '')) fail('Specify a safe --run-id');
  return options;
}
export function specFor(name) {
  if (!Object.hasOwn(specs, name)) fail('Unknown matrix cell');
  return specs[name];
}
export const recordPath = (name, runId) => path.join(privateRoot, `${runId}-${name}.profile.private.json`);
export function privateJson(file) {
  if ((fs.statSync(file).mode & 0o777) !== 0o600) fail('Private input must be mode0600');
  return json(file);
}
function requireUnboundSource(sourceRoot) {
  for (const name of fs.readdirSync(privateRoot).filter(name => name.endsWith('.profile.private.json'))) {
    const owner = privateJson(path.join(privateRoot, name));
    if (typeof owner.sourceRoot !== 'string') fail('Existing profile has no source ownership record');
    const ownedRoot = fs.existsSync(owner.sourceRoot) ? fs.realpathSync(owner.sourceRoot) : path.resolve(owner.sourceRoot);
    if (ownedRoot === sourceRoot) fail('Source root is already bound to a run/cell; preserve that owner and use a fresh archive');
  }
}
const preparationHash = p => sha(JSON.stringify(p));
function requireInitialized(p) {
  const file = path.join(p.root, 'initialized.private.json');
  if (!fs.existsSync(file)) fail('Matching completed initialization is required before startup');
  const receipt = privateJson(file);
  if (receipt.status !== 'completed' || receipt.code !== 0 || !receipt.token || receipt.preparationHash !== preparationHash(p)) fail('Matching completed initialization is required before startup');
}

const pythonOriginProbe = String.raw`
import importlib.util, json, os, sys
from pathlib import Path
root = Path(os.environ['MATRIX_SOURCE_ROOT']).resolve()
venv = Path(os.environ['MATRIX_PYTHON_VENV']).resolve()
if Path(sys.prefix).resolve() != venv: raise RuntimeError('Relocated Python prefix mismatch')
if any('__editable__' in getattr(f, '__module__', '') for f in sys.meta_path): raise RuntimeError('Editable finder is still active')
if any('/.worktrees/' in str(p) for p in sys.path): raise RuntimeError('Worktree fallback remains active')
origins = {}
for name in ('tldw_Server_API', 'mcp_unified', 'tldw_profile_core', 'uvicorn', 'fastapi', 'pydantic', 'psycopg'):
    spec = importlib.util.find_spec(name)
    if not spec or not spec.origin: raise RuntimeError('Required package is unavailable')
    origin = Path(spec.origin).resolve()
    expected = root if name in ('tldw_Server_API', 'mcp_unified', 'tldw_profile_core') else venv
    if not origin.is_relative_to(expected): raise RuntimeError('Package origin escaped the prepared root')
    origins[name] = str(origin)
print(json.dumps({'prefix': str(venv), 'origins': origins}))
`;

// Read-only preflight: never imports the app, starts a service, or touches a DB.
export function inspectInputs(options, { frontendRequired = true } = {}) {
  if (!/^[a-f0-9]{40}$/.test(options['source-commit'] || '')) fail('Specify the released full --source-commit');
  if (!options['source-root'] || !options['python-venv']) fail('Specify --source-root and --python-venv');
  const sourceRoot = fs.realpathSync(options['source-root']);
  const pythonVenv = fs.realpathSync(options['python-venv']);
  if (sourceRoot === privateRoot || !inside(sourceRoot, privateRoot) || !inside(pythonVenv, privateRoot)) fail('Use prepared source and dependency copies beneath this private packet');
  if (fs.existsSync(path.join(sourceRoot, '.git'))) fail('Expected a frozen source archive, not a mutable checkout');
  const sourcePaths = [sourceRoot, path.join(sourceRoot, 'apps/mcp-unified/src'), path.join(sourceRoot, 'packages/tldw_profile_core/src')];
  const sourceFiles = ['pyproject.toml', 'tldw_Server_API/app/main.py', 'tldw_Server_API/app/core/Utils/Utils.py', 'tldw_Server_API/Config_Files/config.txt', 'tldw_Server_API/Config_Files/mcp_modules.yaml', 'apps/tldw-frontend/next.config.mjs', 'apps/tldw-frontend/tsconfig.json', 'apps/tldw-frontend/scripts/live-tier-uat/profile.mjs', 'apps/tldw-frontend/scripts/onboarding-uat/profile.mjs'];
  const sourceHashes = Object.fromEntries(sourceFiles.map(file => {
    const resolved = fs.realpathSync(path.join(sourceRoot, file));
    if (!inside(resolved, sourceRoot)) fail('Frozen source file escaped its archive');
    return [file, sha(fs.readFileSync(resolved))];
  }));
  const python = path.join(pythonVenv, 'bin/python');
  const pythonEnv = { ...baseEnv(), PYTHONPATH: sourcePaths.join(path.delimiter), PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1', MATRIX_SOURCE_ROOT: sourceRoot, MATRIX_PYTHON_VENV: pythonVenv };
  const checked = spawnSync(python, ['-c', pythonOriginProbe], { cwd: privateRoot, env: pythonEnv, encoding: 'utf8', timeout: 30000, maxBuffer: 1024 * 1024 });
  if (checked.status !== 0) fail('Python origin preflight failed; no runtime was launched');
  const pythonOrigins = JSON.parse(checked.stdout.trim());
  const frontend = path.join(sourceRoot, 'apps/tldw-frontend');
  const requireFrontend = createRequire(path.join(frontend, 'package.json'));
  const dependencyOrigins = {};
  if (frontendRequired) {
    for (const name of ['next/package.json', 'react/package.json', 'react-dom/package.json', 'typescript/package.json']) {
      const resolved = fs.realpathSync(requireFrontend.resolve(name));
      if (!inside(resolved, sourceRoot)) fail('Frontend dependency escaped the cell');
      dependencyOrigins[name] = resolved;
    }
    if (fs.realpathSync(path.join(frontend, 'node_modules/@tldw/ui')) !== path.join(sourceRoot, 'apps/packages/ui')) fail('Shared UI escaped the cell');
    const checkLinks = dir => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, entry.name);
        if (entry.isSymbolicLink()) { if (!inside(fs.realpathSync(p), sourceRoot)) fail('Dependency symlink escaped the cell'); }
        else if (entry.isDirectory()) checkLinks(p);
      }
    };
    for (const dir of ['apps/node_modules', 'apps/tldw-frontend/node_modules', 'apps/packages/ui/node_modules']) checkLinks(path.join(sourceRoot, dir));
  }
  return { sourceRoot, sourceCommit: options['source-commit'], pythonVenv, python, sourcePaths, sourceHashes, pythonOrigins, frontend, dependencyOrigins, nextCli: frontendRequired ? requireFrontend.resolve('next/dist/bin/next') : undefined };
}

export function readPgConfig(file) {
  if (!file) fail('Specify the private --pg-config');
  const cfg = privateJson(file);
  if (!['127.0.0.1', 'localhost', '::1'].includes(cfg.host) || !Number.isInteger(Number(cfg.port)) || Number(cfg.port) < 1 || Number(cfg.port) > 65535 || !cfg.user || !cfg.password) fail('Expected the private owned local PG configuration');
  return cfg;
}
function readPgReceipt(p) {
  const receiptPath = fs.realpathSync(p.pgReceiptPath);
  if (!inside(receiptPath, privateRoot)) fail('PG receipt must belong to this private packet');
  const receipt = privateJson(receiptPath), cfg = readPgConfig(p.pgConfigPath);
  if (receipt.profile !== p.name || receipt.run_id !== p.runId || receipt.source_root !== p.sourceRoot || receipt.source_commit !== p.sourceCommit || receipt.python_venv !== p.pythonVenv || receipt.status !== 'held' || !Number.isInteger(receipt.pid)) fail('PG receipt ownership or lifecycle mismatch');
  process.kill(receipt.pid, 0);
  if (receipt.auth_fixture !== 'pg_temp_db' || receipt.content_fixture !== 'pg_temp_db_session') fail('Use official PostgreSQL fixtures');
  const role = receipt.runtime_role, provisioner = receipt.provisioning;
  if (cfg.purpose !== 'matrix-runtime' || cfg.cell !== p.name || cfg.run_id !== p.runId || !/^tldw_matrix_[a-f0-9]{16}$/.test(cfg.user) || !role || role.name !== cfg.user || role.login !== true || role.memberships !== 0 || ['superuser', 'bypassrls', 'inherit', 'createdb', 'createrole', 'replication'].some(flag => role[flag] !== false)) fail('Restricted PostgreSQL runtime role required');
  if (!provisioner?.user || provisioner.user === cfg.user || provisioner.host !== cfg.host || Number(provisioner.port) !== Number(cfg.port)) fail('Distinct matching PostgreSQL provisioning identity required');
  for (const db of [receipt.auth, receipt.content]) {
    if (!db || db.host !== cfg.host || Number(db.port) !== Number(cfg.port) || db.user !== cfg.user || db.password !== cfg.password || !/^tldw_test_[a-f0-9]{8}$/.test(db.database)) fail('PG fixture server mismatch');
  }
  if (receipt.auth.database === receipt.content.database) fail('Auth/content need distinct fixture databases');
  return receipt;
}
const dbUrl = db => `postgresql://${encodeURIComponent(db.user)}:${encodeURIComponent(db.password)}@${db.host.includes(':') ? `[${db.host}]` : db.host}:${db.port}/${db.database}`;
async function helpers(sourceRoot) {
  const live = await import(pathToFileURL(path.join(sourceRoot, 'apps/tldw-frontend/scripts/live-tier-uat/profile.mjs')));
  const onboarding = await import(pathToFileURL(path.join(sourceRoot, 'apps/tldw-frontend/scripts/onboarding-uat/profile.mjs')));
  return { ...live, ...onboarding };
}
function patchIni(text, section, key, value) {
  const lines = text.split(/\r?\n/); let begin = lines.findIndex(l => l.trim() === `[${section}]`);
  if (begin < 0) { lines.push(`[${section}]`, `${key} = ${value}`); return lines.join('\n'); }
  let end = lines.findIndex((l, i) => i > begin && /^\s*\[/.test(l)); if (end < 0) end = lines.length;
  const at = lines.findIndex((l, i) => i > begin && i < end && l.split('=')[0].trim().toLowerCase() === key.toLowerCase());
  if (at >= 0) lines[at] = `${key} = ${value}`; else lines.splice(end, 0, `${key} = ${value}`);
  return lines.join('\n');
}
function frontendEnv(p) {
  const masks = {};
  for (const name of ['.env', '.env.local', '.env.development', '.env.development.local']) {
    const file = path.join(p.frontend, name); if (!fs.existsSync(file)) continue;
    for (const match of fs.readFileSync(file, 'utf8').matchAll(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=/gm)) masks[match[1]] = '';
  }
  return { ...masks, ...baseEnv(), NODE_ENV: 'development', __NEXT_PROCESSED_ENV: 'true', NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: 'quickstart', TLDW_INTERNAL_API_ORIGIN: `http://127.0.0.1:${p.spec.api}`, NEXT_PUBLIC_API_URL: '', NEXT_PUBLIC_API_VERSION: 'v1', NEXT_PUBLIC_X_API_KEY: '', NEXT_PUBLIC_API_BEARER: '', TLDW_NEXT_DIST_DIR: p.nextDistDir, NEXT_TELEMETRY_DISABLED: '1', TMPDIR: path.join(p.root, 'tmp'), XDG_CACHE_HOME: path.join(p.root, 'cache') };
}
async function runtimeEnv(p) {
  const { buildLiveTierBackendEnv } = await helpers(p.sourceRoot);
  const env = buildLiveTierBackendEnv({ profile: p, mockPort: 9099, baseEnv: baseEnv() });
  for (const k of ['DEFAULT_LLM_PROVIDER', 'OPENAI_API_KEY', 'OPENAI_API_BASE_URL', 'CUSTOM_OPENAI_API_KEY', 'CUSTOM_OPENAI_API_IP', 'CUSTOM_OPENAI_API_MODEL', 'WORKFLOWS_EGRESS_BLOCK_PRIVATE', 'WORKFLOWS_EGRESS_ALLOWED_PORTS']) delete env[k];
  const credentials = privateJson(p.credentialsPath), d = p.databaseDir, cache = path.join(p.root, 'cache');
  Object.assign(env, {
    TLDW_ENV_FILE_EXCLUSIVE: 'true', AUTH_MODE: p.spec.mode, SINGLE_USER_API_KEY: credentials.apiKey, JWT_SECRET_KEY: credentials.jwtSecret, API_KEY_HASH_SECRET: credentials.apiHashSecret,
    ENABLE_REGISTRATION: 'false', BYOK_ENABLED: 'false', PYTHONPATH: p.sourcePaths.join(path.delimiter), PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1', PYTHONUNBUFFERED: '1',
    TMPDIR: path.join(p.root, 'tmp'), TEMP: path.join(p.root, 'tmp'), TMP: path.join(p.root, 'tmp'), XDG_CACHE_HOME: cache,
    HF_HOME: path.join(cache, 'huggingface'), HF_HUB_CACHE: path.join(cache, 'huggingface/hub'), TORCH_HOME: path.join(cache, 'torch'), NEMO_CACHE_DIR: path.join(cache, 'nemo'), MPLCONFIGDIR: path.join(cache, 'matplotlib'),
    USER_DATA_BASE_PATH: p.userDatabasesDir, CHROMADB_BASE_PATH: path.join(d, 'chroma_db'), TLDW_USER_DB_BACKEND: p.spec.engine, TLDW_CONTENT_DB_BACKEND: p.spec.engine,
    TLDW_CONTENT_SQLITE_PATH: path.join(d, 'server_media_summary.db'), TLDW_DB_BACKUP_PATH: path.join(p.root, 'backups'), TLDW_DB_ALLOWED_BASE_DIRS: d, MEDIA_DB_PATH: path.join(p.userDatabasesDir, '1/Media_DB_v2.db'),
    AUDIT_SHARED_DB_PATH: path.join(d, 'audit_shared.db'), CONSENT_DB_PATH: path.join(d, 'consent.db'), CIRCUIT_BREAKER_REGISTRY_DB_PATH: path.join(d, 'circuit_breakers.db'), JOBS_AUDIT_DB_PATH: path.join(d, 'jobs_audit.db'),
    MODERATION_REVIEW_DB_PATH: path.join(d, 'moderation_review.db'), RESEARCH_SESSIONS_DB_PATH: path.join(d, 'research_sessions.db'), SANDBOX_STORE_DB_PATH: path.join(d, 'sandbox.db'), EVALUATIONS_DB_PATH: p.databasePaths.evaluations,
    SCHEDULER_DATABASE_URL: `sqlite:///${path.join(d, 'scheduler.db')}`, SCHEDULER_BASE_PATH: path.join(d, 'scheduler'), WORKFLOWS_FILE_BASE_DIR: path.join(p.root, 'workflows'), WORKFLOWS_ARTIFACTS_DIR: path.join(p.root, 'workflow-artifacts'),
    CODEGRAPH_INDEX_BASE_DIR: path.join(p.root, 'codegraph'), CODEGRAPH_JOBS_INDEX_BASE_DIR: path.join(p.root, 'codegraph-jobs'), RAG_CACHE_DIR: path.join(cache, 'rag'), RAG_SEMANTIC_CACHE_DIR: path.join(cache, 'rag-semantic'), RAG_FLASHRANK_CACHE_DIR: path.join(cache, 'flashrank'),
    ALLOWED_ORIGINS: `http://localhost:${p.spec.web},http://127.0.0.1:${p.spec.web}`,
  });
  if (p.spec.engine === 'postgresql') { const receipt = readPgReceipt(p); env.DATABASE_URL = dbUrl(receipt.auth); env.TLDW_CONTENT_PG_DSN = dbUrl(receipt.content); }
  if (Object.keys(env).some(k => /^(?:TEST_MODE|PYTEST_|TLDW_TEST_|CHAT_FORCE_MOCK)/.test(k))) fail('Test flags leaked into application environment');
  return env;
}
async function prepare(name, options) {
  const spec = { ...specFor(name), api: Number(options['api-port']), web: Number(options['web-port']) };
  if (![spec.api, spec.web].every(n => Number.isInteger(n) && n > 1024 && n < 65536) || spec.api === spec.web) fail('Specify distinct --api-port and --web-port');
  const runId = options['run-id'], record = recordPath(name, runId);
  if (fs.existsSync(record)) fail('Profile already exists; preparation never resets or rebinds it');
  const inspected = inspectInputs(options), ownerId = `${runId}-${name}`;
  requireUnboundSource(inspected.sourceRoot);
  const expectedRoot = path.join(profilesRoot, `tldw-onboarding-uat-${ownerId}`);
  if (fs.existsSync(expectedRoot)) fail('Refusing to overwrite an existing runtime directory');
  if (spec.engine === 'postgresql' && (!options['pg-config'] || !options['pg-receipt'])) fail('Prepare PG only after the separate official holder provides --pg-config and --pg-receipt');
  const pg = spec.engine === 'postgresql' ? { pgConfigPath: path.resolve(options['pg-config']), pgReceiptPath: path.resolve(options['pg-receipt']) } : {};
  if (spec.engine === 'sqlite' && (options['pg-config'] || options['pg-receipt'])) fail('SQLite cell cannot attach a PG receipt');
  if (spec.engine === 'postgresql') readPgReceipt({ ...inspected, ...pg, name, runId });
  const preparationId = crypto.randomUUID();
  writePrivate(record, { ...inspected, ...pg, name, spec, runId, root: expectedRoot, preparationId, preparationStatus: 'preparing' });
  const { buildLiveTierProfile, scrubHostConfigurationValues } = await helpers(inspected.sourceRoot);
  const p = { ...buildLiveTierProfile({ repoRoot: inspected.sourceRoot, frontendRoot: inspected.frontend, runId: ownerId, mockPort: 9099, baseTmpDir: profilesRoot, pythonCommand: inspected.python }), ...inspected, ...pg, name, spec, runId, preparationId, preparationStatus: 'complete' };
  for (const dir of ['tmp', 'cache', 'backups', 'models', 'workflows', 'workflow-artifacts', 'codegraph', 'codegraph-jobs']) fs.mkdirSync(path.join(p.root, dir), { recursive: true, mode: 0o700 });
  Object.assign(p, { credentialsPath: path.join(p.root, 'credentials.private.json'), safeManifestPath: path.join(p.root, 'safe-profile-manifest.json'), nextDistDir: `.next-live-tier-${ownerId}`, browserSessionName: ownerId });
  const password = () => `Uat-${crypto.randomBytes(24).toString('base64url')}!a7`;
  writePrivate(p.credentialsPath, { apiKey: `tldw_uat_${crypto.randomBytes(32).toString('hex')}`, jwtSecret: crypto.randomBytes(48).toString('base64url'), apiHashSecret: crypto.randomBytes(32).toString('hex'), accounts: Object.fromEntries(['admin', 'alice', 'bob'].map(user => [user, { username: user === 'admin' ? 'uat_admin' : user, email: `${user}-${name}@example.com`, password: password() }])), accountStatus: 'planned-not-created' });
  const env = await runtimeEnv(p);
  let text = scrubHostConfigurationValues(fs.readFileSync(path.join(p.sourceRoot, 'tldw_Server_API/Config_Files/config.txt'), 'utf8'), { mockBaseUrl: '' });
  const overrides = {
    Setup: { enable_first_time_setup: 'true', setup_completed: 'false' }, AuthNZ: { auth_mode: spec.mode, single_user_api_key: env.SINGLE_USER_API_KEY, database_url: env.DATABASE_URL },
    'TTS-Settings': { USER_DB_BASE_DIR: p.userDatabasesDir }, Files: { ingestion_source_allowed_roots: p.fixtureRoot }, Audit: { shared_db_path: env.AUDIT_SHARED_DB_PATH },
    Database: { type: spec.engine, sqlite_path: env.TLDW_CONTENT_SQLITE_PATH, backup_path: env.TLDW_DB_BACKUP_PATH, chroma_db_path: env.CHROMADB_BASE_PATH, prompts_db_path: path.join(p.databaseDir, 'prompts.db'), rag_qa_db_path: path.join(p.databaseDir, 'RAG_QA_Chat.db'), character_db_path: path.join(p.databaseDir, 'chatDB.db') },
    Embeddings: { onnx_model_path: path.join(p.root, 'models/onnx'), model_dir: path.join(p.root, 'models/embeddings') }, RAG: { flashrank_cache_dir: env.RAG_FLASHRANK_CACHE_DIR, default_llm_provider: '', default_llm_model: '' }, 'STT-Settings': { nemo_cache_dir: env.NEMO_CACHE_DIR },
    Logging: { log_file: path.join(p.logsDir, 'app.json'), log_metrics_file: path.join(p.logsDir, 'metrics.json'), system_log_file_path: p.systemLogFilePath },
  };
  for (const [section, values] of Object.entries(overrides)) for (const [key, value] of Object.entries(values)) text = patchIni(text, section, key, value);
  writePrivate(p.configPath, text);
  writePrivate(p.envPath, Object.entries(env).filter(([k]) => !baseKeys.includes(k)).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join('\n') + '\n');
  writePrivate(record, p);
  writePrivate(p.safeManifestPath, { name, runId, sourceRoot: p.sourceRoot, declaredSourceCommit: p.sourceCommit, sourceHashes: p.sourceHashes, pythonOrigins: p.pythonOrigins, dependencyOrigins: p.dependencyOrigins, mode: spec.mode, engine: spec.engine, apiPort: spec.api, webPort: spec.web, browserSessionName: p.browserSessionName, mutableRoots: [p.root, path.join(p.sourceRoot, 'Databases'), path.join(p.frontend, p.nextDistDir)], reusedDependencies: true, originalArchiveManifestRequired: true, noAccountsOrApplicationDatabasesCreatedByPrepare: true, providerEnvOverrides: 0, testModeFlags: 0, limitations: ['Prepared state is not startup/native acceptance.', 'Source commit is declared; parent archive manifest remains authoritative.', 'Optional ACP helper stub is inherited and unexercised; this does not certify real ACP.'] });
  console.log(JSON.stringify({ prepared: name, runId, manifest: p.safeManifestPath }));
}
function portFree(port) {
  return new Promise((resolve, reject) => { const server = net.createServer(); server.once('error', e => reject(Error(`Port preflight failed: ${e.code || 'unknown'}`))); server.listen(port, '127.0.0.1', () => server.close(resolve)); });
}
async function launch(action, name, options) {
  if (Object.keys(options).some(key => key !== 'run-id')) fail('Launch uses only --run-id; preparation owns the immutable source and port settings');
  specFor(name); const p = privateJson(recordPath(name, options['run-id']));
  if (p.name !== name || p.runId !== options['run-id'] || p.spec.mode !== specs[name].mode || p.spec.engine !== specs[name].engine) fail('Prepared profile ownership mismatch');
  if (p.preparationStatus !== 'complete' || !p.preparationId) fail('Matching completed preparation is required');
  if (action !== 'initialize') requireInitialized(p);
  if (action === 'bootstrap-admin') {
    if (p.spec.mode !== 'multi_user') fail('Admin bootstrap requires a multi-user cell');
    if (fs.existsSync(path.join(p.root, 'admin-bootstrapped.private.json'))) fail('Admin bootstrap already completed; this action does not reset passwords');
  }
  const checked = inspectInputs({ 'source-root': p.sourceRoot, 'python-venv': p.pythonVenv, 'source-commit': p.sourceCommit });
  if (JSON.stringify(checked.sourceHashes) !== JSON.stringify(p.sourceHashes)) fail('Prepared source fingerprints changed');
  const env = await runtimeEnv(p);
  if (p.spec.engine === 'postgresql') {
    const probe = spawnSync(p.python, [path.join(privateRoot, 'pg_role_adapter.py'), p.pgConfigPath, p.pgReceiptPath], { cwd: p.root, env, encoding: 'utf8', timeout: 30000, maxBuffer: 1024 * 1024 });
    if (probe.status !== 0) fail('PostgreSQL runtime role live verification failed; no process launched');
  }
  let command = p.python, args, cwd = p.root, initializationRequest, adminRequest;
  if (action === 'initialize') {
    if (fs.existsSync(path.join(p.root, 'initialized.private.json'))) fail('Initialization already completed; do not reset a cell');
    const token = crypto.randomUUID();
    initializationRequest = { token, preparationHash: preparationHash(p), sourceRoot: p.sourceRoot, receiptPath: path.join(p.root, `initialize-${token}.completed.private.json`) };
    env.MATRIX_INIT_REQUEST = JSON.stringify(initializationRequest);
    args = [path.join(privateRoot, 'initialize-cell.py')];
  } else if (action === 'bootstrap-admin') {
    const token = crypto.randomUUID();
    adminRequest = { token, preparationHash: preparationHash(p), sourceRoot: p.sourceRoot, credentialsPath: p.credentialsPath, receiptPath: path.join(p.root, `bootstrap-admin-${token}.completed.private.json`) };
    env.MATRIX_ADMIN_REQUEST = JSON.stringify(adminRequest);
    args = [path.join(privateRoot, 'bootstrap-admin-cell.py')];
  } else if (action === 'backend') {
    await portFree(p.spec.api); args = ['-m', 'uvicorn', 'tldw_Server_API.app.main:app', '--host', '127.0.0.1', '--port', String(p.spec.api)];
  } else {
    await portFree(p.spec.web);
    const dist = path.join(p.frontend, p.nextDistDir), receiptFile = path.join(p.root, 'frontend-build-root.private.json');
    if (fs.existsSync(dist) && (fs.lstatSync(dist).isSymbolicLink() || !fs.existsSync(receiptFile) || privateJson(receiptFile).path !== dist)) fail('Refusing an unowned or symlinked Next build root');
    fs.mkdirSync(dist, { recursive: true, mode: 0o700 }); writePrivate(receiptFile, { path: dist, name, runId: p.runId });
    command = process.execPath; args = [checked.nextCli, 'dev', '--hostname', '127.0.0.1', '--port', String(p.spec.web)]; cwd = p.frontend;
  }
  const logPath = path.join(p.logsDir, `${action}-${Date.now()}-${crypto.randomUUID()}.private.log`), fd = fs.openSync(logPath, 'wx', 0o600);
  const child = spawn(command, args, { cwd, env: action === 'frontend' ? frontendEnv(p) : env, stdio: ['ignore', fd, fd] });
  const receipt = path.join(p.root, `${action}-process.private.json`);
  writePrivate(receipt, { pid: child.pid, action, name, runId: p.runId, startedAt: new Date().toISOString(), command, args, cwd, logPath });
  console.log(JSON.stringify({ action, name, runId: p.runId, pid: child.pid, logPath }));
  let interrupted = false;
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { interrupted = true; child.kill(signal); });
  child.once('error', () => { fs.closeSync(fd); process.exitCode = 1; console.error('Runtime failed to spawn; inspect private receipt.'); });
  child.once('exit', (code, signal) => {
    fs.closeSync(fd);
    let resultCode = code ?? 1;
    if (action === 'initialize' && code === 0) {
      try {
        if (interrupted || signal) fail('Initialization was interrupted; inspect the private log before retrying');
        const proof = privateJson(initializationRequest.receiptPath);
        if (proof.status !== 'completed' || proof.token !== initializationRequest.token || proof.preparationHash !== initializationRequest.preparationHash) fail('Initialization completion proof does not match this attempt');
        writePrivate(path.join(p.root, 'initialized.private.json'), { ...proof, at: new Date().toISOString(), code });
      } catch (error) { resultCode = 1; reportFailure(error); }
    }
    if (action === 'bootstrap-admin' && code === 0) {
      try {
        if (interrupted || signal) fail('Admin bootstrap was interrupted; inspect the private log before retrying');
        const proof = privateJson(adminRequest.receiptPath);
        if (proof.status !== 'completed' || proof.action !== 'bootstrap-admin' || proof.token !== adminRequest.token || proof.preparationHash !== adminRequest.preparationHash) fail('Admin bootstrap completion proof does not match this attempt');
        writePrivate(path.join(p.root, 'admin-bootstrapped.private.json'), { ...proof, at: new Date().toISOString(), code });
      } catch (error) { resultCode = 1; reportFailure(error); }
    }
    writePrivate(receipt, { ...privateJson(receipt), endedAt: new Date().toISOString(), code, signal, resultCode });
    console.log(JSON.stringify({ action, name, code, signal, resultCode })); process.exitCode = resultCode;
  });
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const [action, name, ...args] = process.argv.slice(2), options = parseOptions(args); specFor(name);
    if (action === 'check-inputs') console.log(JSON.stringify(inspectInputs(options), null, 2));
    else if (action === 'prepare') await prepare(name, options);
    else if (['initialize', 'bootstrap-admin', 'backend', 'frontend'].includes(action)) await launch(action, name, options);
    else fail('Use check-inputs|prepare|initialize|bootstrap-admin|backend|frontend CELL --run-id ID and explicit preparation paths');
  } catch (error) { reportFailure(error); process.exitCode = 1; }
}
