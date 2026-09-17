// Separate, opt-in official fixture lifecycle. Never imported by app runtime.
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { privateRoot, parseOptions, inspectInputs, readPgConfig, baseEnv, writePrivate, fail, reportFailure } from './matrix-launcher.mjs';

try {
  const [name, ...args] = process.argv.slice(2), options = parseOptions(args);
  if (!['pg-single', 'pg-multi'].includes(name)) fail('Specify a PG cell');
  const inspected = inspectInputs(options, { frontendRequired: false }), cfg = readPgConfig(options['pg-config']);
  const runId = options['run-id'], holderRoot = path.join(privateRoot, 'holders', `${runId}-${name}`);
  if (fs.existsSync(holderRoot)) fail('Refusing to overwrite an existing holder');
  fs.mkdirSync(holderRoot, { recursive: true, mode: 0o700 });
  const receiptPath = path.join(holderRoot, `${name}.pg-receipt.private.json`);
  const env = {
    ...baseEnv(), PYTHONPATH: inspected.sourcePaths.join(path.delimiter), PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1', PYTEST_DISABLE_PLUGIN_AUTOLOAD: '1',
    POSTGRES_TEST_HOST: cfg.host, POSTGRES_TEST_PORT: String(cfg.port), POSTGRES_TEST_USER: cfg.user, POSTGRES_TEST_PASSWORD: cfg.password, POSTGRES_TEST_DB: cfg.defaultDb || 'postgres',
    TLDW_TEST_POSTGRES_REQUIRED: '1', TLDW_TEST_NO_DOCKER: '1', TLDW_TEST_PG_CONTAINER_NAME: cfg.container, TLDW_TEST_PG_ALT_PORT: String(cfg.port),
    MATRIX_PG_CONTAINER: cfg.container, MATRIX_HOLDER_ROOT: holderRoot, MATRIX_CELL: name, MATRIX_RUN_ID: runId, MATRIX_SOURCE_ROOT: inspected.sourceRoot, MATRIX_SOURCE_COMMIT: inspected.sourceCommit, MATRIX_PYTHON_VENV: inspected.pythonVenv,
  };
  const pytestArgs = ['-m', 'pytest', '-p', 'tldw_Server_API.tests._plugins.postgres', '--confcutdir', privateRoot, '--rootdir', privateRoot, '-c', path.join(privateRoot, 'pytest-holder.ini'), '-q', '-s', path.join(privateRoot, 'test_hold_official_pg.py')];
  const logPath = path.join(holderRoot, 'holder.private.log'), fd = fs.openSync(logPath, 'wx', 0o600);
  const child = spawn(inspected.python, pytestArgs, { cwd: holderRoot, env, stdio: ['ignore', fd, fd] });
  writePrivate(path.join(holderRoot, 'process.private.json'), { pid: child.pid, name, runId, sourceRoot: inspected.sourceRoot, sourceCommit: inspected.sourceCommit, pythonVenv: inspected.pythonVenv, sourceHashes: inspected.sourceHashes, logPath, startedAt: new Date().toISOString() });
  console.log(JSON.stringify({ name, runId, pid: child.pid, receiptPath, runtimeConfigPath: path.join(holderRoot, 'runtime.pg-config.private.json'), logPath, officialFixturesOnly: true }));
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
  child.once('error', () => { fs.closeSync(fd); process.exitCode = 1; console.error('Official fixture holder could not start; inspect private log.'); });
  child.once('exit', (code, signal) => { fs.closeSync(fd); console.log(JSON.stringify({ name, runId, code, signal })); process.exitCode = code ?? 1; });
} catch (error) { reportFailure(error); process.exitCode = 1; }
