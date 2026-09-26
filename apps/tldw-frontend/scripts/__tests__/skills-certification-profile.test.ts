import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import * as profileModule from '../skills-certification/profile.mjs';

const {
  SKILLS_CERT_API_KEY,
  SKILLS_CERT_NAMES,
  buildSkillsCertificationCommands,
  buildSkillsCertificationEnvironments,
  createSkillsCertificationProfile,
  isConfirmedBindConflict,
} = profileModule;

const temporaryRoots: string[] = [];

const sourceConfig = `[Setup]
enable_first_time_setup = false
setup_completed = false

[AuthNZ]
auth_mode = multi_user
single_user_api_key = copied-auth-key

[Providers]
OPENAI_API_KEY = host-provider-secret
anthropic-token = host-anthropic-token
service secret = host-service-secret
database.password = host-database-password
search_engine_api_key_baidu = host-baidu-api-key
api_secret_key = host-api-secret-key
# commented_api_key = host-comment-secret
llama_api_IP = http://192.168.2.235:5000/v1
custom_openai_api_ip = https://api.example.test/v1
provider_name = retained-provider
max_tokens = 4096

[TTS-Settings]
USER_DB_BASE_DIR = /host/user-databases

[Logging]
system_log_file_path = Databases/system_logs.jsonl
`;

function writeExecutable(filePath: string) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, '#!/bin/sh\nexit 0\n', 'utf8');
  chmodSync(filePath, 0o755);
}

function createFixture({ localPython = true } = {}) {
  const fixtureRoot = mkdtempSync(path.join(tmpdir(), 'skills-certification-profile-test-'));
  temporaryRoots.push(fixtureRoot);

  const repoRoot = path.join(fixtureRoot, 'repo');
  const frontendRoot = path.join(repoRoot, 'apps/tldw-frontend');
  const extensionRoot = path.join(repoRoot, 'apps/extension');
  const temporaryBase = path.join(fixtureRoot, 'runtime-base');
  const configDir = path.join(repoRoot, 'tldw_Server_API/Config_Files');
  const localPythonPath = path.join(repoRoot, '.venv/bin/python');

  mkdirSync(frontendRoot, { recursive: true });
  mkdirSync(extensionRoot, { recursive: true });
  mkdirSync(temporaryBase, { recursive: true });
  mkdirSync(configDir, { recursive: true });
  writeFileSync(path.join(configDir, 'config.txt'), sourceConfig, 'utf8');
  if (localPython) {
    writeExecutable(localPythonPath);
  }

  return {
    extensionRoot,
    fixtureRoot,
    frontendRoot,
    localPythonPath,
    repoRoot,
    temporaryBase,
  };
}

function parseEnvFile(text: string) {
  return Object.fromEntries(
    text
      .trim()
      .split('\n')
      .map((line) => {
        const separator = line.indexOf('=');
        return [line.slice(0, separator), line.slice(separator + 1)];
      })
  );
}

function createProfile(fixture: ReturnType<typeof createFixture>) {
  return createSkillsCertificationProfile({
    repoRoot: fixture.repoRoot,
    temporaryBase: fixture.temporaryBase,
  });
}

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { force: true, recursive: true });
  }
});

describe('Skills certification profile', () => {
  it('exports only the stable profile entry points and fixed names', () => {
    expect(Object.keys(profileModule).sort()).toEqual([
      'SKILLS_CERT_API_KEY',
      'SKILLS_CERT_NAMES',
      'buildSkillsCertificationCommands',
      'buildSkillsCertificationEnvironments',
      'createSkillsCertificationProfile',
      'isConfirmedBindConflict',
    ]);
    expect(SKILLS_CERT_API_KEY).toBe('THIS-IS-A-SECURE-KEY-123-UAT');
    expect(SKILLS_CERT_NAMES).toEqual({
      extension: 'skills-cert-extension',
      webui: 'skills-cert-web',
    });
    expect(Object.isFrozen(SKILLS_CERT_NAMES)).toBe(true);
  });

  it('creates one marker-protected runtime with absolute isolated paths', () => {
    const fixture = createFixture();
    const profile = createProfile(fixture);

    expect(path.dirname(profile.root)).toBe(path.resolve(fixture.temporaryBase));
    expect(readdirSync(fixture.temporaryBase)).toHaveLength(1);
    expect(existsSync(profile.markerPath)).toBe(true);

    for (const value of Object.values(profile)) {
      expect(path.isAbsolute(value)).toBe(true);
    }
    for (const directory of [
      profile.configDir,
      profile.databaseDir,
      profile.userDatabasesDir,
      profile.homeDir,
      profile.tmpDir,
      profile.extensionProfileDir,
    ]) {
      expect(existsSync(directory)).toBe(true);
    }
    expect(path.dirname(profile.usersDbPath)).toBe(profile.databaseDir);
    expect(profile.systemLogPath).toBe(
      path.join(profile.databaseDir, 'system_logs.jsonl')
    );
    expect(JSON.stringify(profile)).not.toContain(SKILLS_CERT_API_KEY);
  });

  it('returns the resolved supplied temporary base for strict runtime cleanup', () => {
    const fixture = createFixture();
    const profile = createProfile(fixture);

    expect(profile.baseRoot).toBe(path.resolve(fixture.temporaryBase));
    expect(path.dirname(profile.root)).toBe(profile.baseRoot);
  });

  it('removes the newly created runtime root when profile setup fails after allocation', () => {
    const fixture = createFixture();
    rmSync(path.join(fixture.repoRoot, 'tldw_Server_API/Config_Files/config.txt'));

    expect(() => createProfile(fixture)).toThrow();
    expect(readdirSync(fixture.temporaryBase)).toEqual([]);
  });

  it('retains the allocated runtime descriptor when rollback fails', () => {
    const fixture = createFixture();
    rmSync(path.join(fixture.repoRoot, 'tldw_Server_API/Config_Files/config.txt'));
    const removeRoot = () => {
      throw new Error('rollback failed');
    };

    try {
      createSkillsCertificationProfile({
        repoRoot: fixture.repoRoot,
        temporaryBase: fixture.temporaryBase,
        removeRoot,
      });
      throw new Error('expected profile setup to fail');
    } catch (error) {
      expect(error).toBeInstanceOf(AggregateError);
      expect(error.runtime).toMatchObject({ baseRoot: fixture.temporaryBase });
      expect(existsSync(error.runtime.root)).toBe(true);
    }
  });

  it('scrubs copied credentials and provider endpoints before setting completed single-user auth', () => {
    const fixture = createFixture();
    const profile = createProfile(fixture);
    const configText = readFileSync(profile.configPath, 'utf8');

    expect(configText).toContain('enable_first_time_setup = true');
    expect(configText).toContain('setup_completed = true');
    expect(configText).toContain('auth_mode = single_user');
    expect(configText).toContain(`single_user_api_key = ${SKILLS_CERT_API_KEY}`);
    expect(configText.match(new RegExp(SKILLS_CERT_API_KEY, 'g'))).toHaveLength(1);
    expect(configText).toContain('OPENAI_API_KEY =');
    expect(configText).toContain('anthropic-token =');
    expect(configText).toContain('service secret =');
    expect(configText).toContain('database.password =');
    expect(configText).toMatch(/^search_engine_api_key_baidu =\s*$/m);
    expect(configText).toMatch(/^api_secret_key =\s*$/m);
    expect(configText).toContain('# commented_api_key =');
    expect(configText).toMatch(/^llama_api_IP =\s*$/m);
    expect(configText).toMatch(/^custom_openai_api_ip =\s*$/m);
    expect(configText).toContain('provider_name = retained-provider');
    expect(configText).toContain('max_tokens = 4096');
    expect(configText).toContain(`USER_DB_BASE_DIR = ${profile.userDatabasesDir}`);
    expect(configText).toContain(`system_log_file_path = ${profile.systemLogPath}`);
    expect(configText).not.toContain('system_log_file_path = Databases/system_logs.jsonl');
    expect(configText).not.toContain('host-provider-secret');
    expect(configText).not.toContain('host-anthropic-token');
    expect(configText).not.toContain('host-service-secret');
    expect(configText).not.toContain('host-database-password');
    expect(configText).not.toContain('host-baidu-api-key');
    expect(configText).not.toContain('host-api-secret-key');
    expect(configText).not.toContain('host-comment-secret');
    expect(configText).not.toContain('192.168.2.235');
    expect(configText).not.toContain('api.example.test');
    expect(configText).not.toContain('copied-auth-key');
  });

  it('writes only single-user auth, audit mode, and isolated paths to .env', () => {
    const fixture = createFixture();
    const profile = createProfile(fixture);
    const envText = readFileSync(profile.envPath, 'utf8');
    const env = parseEnvFile(envText);

    expect(Object.keys(env).sort()).toEqual([
      'AUTH_MODE',
      'CONTEXT_INTEGRITY_MODE',
      'DATABASE_URL',
      'SINGLE_USER_API_KEY',
      'TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS',
      'USER_DB_BASE_DIR',
      'USER_DB_BASE_DIR_ALLOWED_ROOTS',
    ]);
    expect(env).toEqual({
      AUTH_MODE: 'single_user',
      CONTEXT_INTEGRITY_MODE: 'audit_only',
      DATABASE_URL: `sqlite:///${profile.usersDbPath}`,
      SINGLE_USER_API_KEY: SKILLS_CERT_API_KEY,
      TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS: profile.databaseDir,
      USER_DB_BASE_DIR: profile.userDatabasesDir,
      USER_DB_BASE_DIR_ALLOWED_ROOTS: profile.databaseDir,
    });
    expect(path.isAbsolute(env.DATABASE_URL.slice('sqlite:///'.length))).toBe(true);
    for (const key of [
      'USER_DB_BASE_DIR',
      'USER_DB_BASE_DIR_ALLOWED_ROOTS',
      'TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS',
    ]) {
      expect(path.isAbsolute(env[key])).toBe(true);
    }
    expect(envText).not.toMatch(/OPENAI|ANTHROPIC|PROVIDER/i);
    expect(envText).not.toContain('host-provider-secret');
  });

  it('builds separate allowlisted child environments without host secrets', () => {
    const fixture = createFixture();
    const profile = createProfile(fixture);
    const environments = buildSkillsCertificationEnvironments({
      baseEnv: {
        ANTHROPIC_API_KEY: 'host-anthropic',
        CONTEXT_INTEGRITY_MODE: 'hardened',
        HOME: '/host/home',
        HOST_PROVIDER_SECRET: 'host-provider-secret',
        LANG: 'en_US.UTF-8',
        OPENAI_API_KEY: 'host-openai',
        PATH: '/usr/bin:/bin',
        PLAYWRIGHT_BROWSERS_PATH: '/host/playwright-browsers',
        SSL_CERT_FILE: '/host/cert.pem',
        TESTING: '1',
        TEST_MODE: '1',
        TMPDIR: '/host/tmp',
        UNSAFE_UNRELATED_VALUE: 'must-not-pass',
        VIRTUAL_ENV: path.join(fixture.repoRoot, '.venv'),
      },
      ports: { backend: 18992, web: 18991 },
      profile,
    });

    const childEnvironments = Object.values(environments) as Array<Record<string, string>>;
    expect(new Set(childEnvironments).size).toBe(childEnvironments.length);
    for (const [name, childEnv] of Object.entries(environments) as Array<
      [string, Record<string, string>]
    >) {
      expect(childEnv.PATH).toBe('/usr/bin:/bin');
      expect(childEnv.LANG).toBe('en_US.UTF-8');
      expect(childEnv.PLAYWRIGHT_BROWSERS_PATH).toBe('/host/playwright-browsers');
      expect(childEnv.SSL_CERT_FILE).toBe('/host/cert.pem');
      expect(childEnv).not.toHaveProperty('OPENAI_API_KEY');
      expect(childEnv).not.toHaveProperty('ANTHROPIC_API_KEY');
      expect(childEnv).not.toHaveProperty('HOST_PROVIDER_SECRET');
      expect(childEnv).not.toHaveProperty('TESTING');
      expect(childEnv).not.toHaveProperty('TEST_MODE');
      expect(childEnv).not.toHaveProperty('UNSAFE_UNRELATED_VALUE');
      if (name === 'backendEnv') {
        expect(childEnv.CONTEXT_INTEGRITY_MODE).toBe('audit_only');
      } else {
        expect(childEnv).not.toHaveProperty('CONTEXT_INTEGRITY_MODE');
      }
    }

    expect(environments.authInitEnv.HOME).toBe(profile.homeDir);
    expect(environments.authInitEnv.TMPDIR).toBe(profile.tmpDir);
    expect(environments.backendEnv.HOME).toBe(profile.homeDir);
    expect(environments.backendEnv.TMPDIR).toBe(profile.tmpDir);
    expect(environments.backendEnv.USER_DB_BASE_DIR).toBe(profile.userDatabasesDir);
    expect(environments.backendEnv.TLDW_ENV_FILE_EXCLUSIVE).toBe('1');
    expect(environments.authInitEnv.TLDW_ENV_FILE_EXCLUSIVE).toBe('1');
    expect(environments.backendEnv.JOBS_DB_PATH).toBe(
      path.join(profile.databaseDir, 'jobs.db')
    );
    expect(environments.backendEnv.JOBS_DB_URL).toBe(
      `sqlite:///${path.join(profile.databaseDir, 'jobs.db')}`
    );
    expect(environments.backendEnv.MEDIA_DB_PATH).toBe(
      path.join(profile.userDatabasesDir, '1', 'Media_DB_v2.db')
    );
    expect(environments.backendEnv.SYSTEM_LOG_FILE_PATH).toBe(profile.systemLogPath);
    expect(environments.backendEnv.MCP_MODULES_CONFIG).toBe(
      path.join(profile.configDir, 'mcp-modules-disabled.yaml')
    );
    expect(environments.backendEnv.MCP_MODULES).toBe(
      'skills=tldw_Server_API.app.core.MCP_unified.modules.implementations.skills_module:SkillsModule'
    );
    expect(environments.backendEnv.SINGLE_USER_API_KEY).toBe(SKILLS_CERT_API_KEY);
    expect(environments.frontendEnv.NEXT_PUBLIC_X_API_KEY).toBe(SKILLS_CERT_API_KEY);
    expect(environments.webuiPlaywrightEnv.TLDW_E2E_API_KEY).toBe(SKILLS_CERT_API_KEY);
    expect(environments.extensionPlaywrightEnv.TLDW_E2E_API_KEY).toBe(SKILLS_CERT_API_KEY);
    expect(JSON.stringify(environments.extensionBuildEnv)).not.toContain(SKILLS_CERT_API_KEY);
    expect(JSON.stringify(environments.webuiChromiumProbeEnv)).not.toContain(SKILLS_CERT_API_KEY);
    expect(JSON.stringify(environments.extensionChromiumProbeEnv)).not.toContain(
      SKILLS_CERT_API_KEY
    );
    expect(environments.extensionPlaywrightEnv.TLDW_E2E_SKIP_EXTENSION_BUILD).toBe('1');
    expect(environments.webuiPlaywrightEnv.TLDW_SKILLS_CERT_SKILL_NAME).toBe('skills-cert-web');
    expect(environments.extensionPlaywrightEnv.TLDW_SKILLS_CERT_SKILL_NAME).toBe(
      'skills-cert-extension'
    );
  });

  it('builds fixed commands and package-local finite Chromium probes', () => {
    const fixture = createFixture();
    const profile = createProfile(fixture);
    const commands = buildSkillsCertificationCommands({
      baseEnv: { PATH: '/usr/bin:/bin' },
      extensionRoot: fixture.extensionRoot,
      frontendRoot: fixture.frontendRoot,
      ports: { backend: 18992, web: 18991 },
      profile,
      repoRoot: fixture.repoRoot,
    });

    expect(Object.keys(commands).sort()).toEqual([
      'authInit',
      'backend',
      'extensionBuild',
      'extensionChromiumProbe',
      'extensionPlaywright',
      'frontend',
      'webuiChromiumProbe',
      'webuiPlaywright',
    ]);
    for (const command of Object.values(commands)) {
      expect(path.isAbsolute(command.cwd)).toBe(true);
      expect(Array.isArray(command.args)).toBe(true);
      expect(command.args.every((arg) => typeof arg === 'string')).toBe(true);
      expect(command.args.join(' ')).not.toContain(SKILLS_CERT_API_KEY);
    }

    expect(commands.authInit).toMatchObject({
      command: fixture.localPythonPath,
      cwd: fixture.repoRoot,
    });
    expect(commands.authInit.args).toEqual([
      '-m',
      'tldw_Server_API.app.core.AuthNZ.initialize',
      '--non-interactive',
    ]);
    expect(commands.backend).toMatchObject({
      command: fixture.localPythonPath,
      cwd: fixture.repoRoot,
    });
    expect(commands.backend.args).toEqual([
      '-m',
      'uvicorn',
      'tldw_Server_API.app.main:app',
      '--host',
      '127.0.0.1',
      '--port',
      '18992',
    ]);
    expect(commands.frontend).toMatchObject({
      command: 'bun',
      cwd: fixture.frontendRoot,
    });
    expect(commands.frontend.args).toEqual(['run', 'dev', '--', '-p', '18991']);
    expect(commands.webuiPlaywright).toMatchObject({
      command: 'bunx',
      cwd: fixture.frontendRoot,
    });
    expect(commands.webuiPlaywright.args).toEqual([
      'playwright',
      'test',
      '-c',
      'e2e/skills-certification/playwright.config.ts',
    ]);
    expect(commands.extensionBuild).toMatchObject({
      command: 'bun',
      cwd: fixture.extensionRoot,
    });
    expect(commands.extensionBuild.args).toEqual(['run', 'build:chrome:prod']);
    expect(commands.extensionPlaywright).toMatchObject({
      command: 'bunx',
      cwd: fixture.extensionRoot,
    });
    expect(commands.extensionPlaywright.args).toEqual([
      'playwright',
      'test',
      '-c',
      'playwright.skills-certification.config.ts',
    ]);
    expect(commands.extensionPlaywright.env.TLDW_E2E_SKIP_EXTENSION_BUILD).toBe('1');

    for (const [probe, cwd] of [
      [commands.webuiChromiumProbe, fixture.frontendRoot],
      [commands.extensionChromiumProbe, fixture.extensionRoot],
    ] as const) {
      expect(probe.command).toBe(process.execPath);
      expect(probe.cwd).toBe(cwd);
      expect(probe.args.slice(0, 2)).toEqual(['--input-type=module', '--eval']);
      expect(probe.args[2]).toContain('import("@playwright/test")');
      expect(probe.args[2]).toContain('chromium.launch({ headless: true })');
      expect(probe.args[2]).toContain('browser.close()');
      expect(probe.args[2]).not.toMatch(/https?:|goto\(|install|download/i);
    }
  });

  it('prefers active VIRTUAL_ENV Python, falls back locally, and otherwise fails', () => {
    const fixture = createFixture();
    const profile = createProfile(fixture);
    const activeVenv = path.join(fixture.fixtureRoot, 'active-venv');
    const activePython = path.join(activeVenv, 'bin/python');
    writeExecutable(activePython);
    const options = {
      extensionRoot: fixture.extensionRoot,
      frontendRoot: fixture.frontendRoot,
      ports: { backend: 18992, web: 18991 },
      profile,
      repoRoot: fixture.repoRoot,
    };

    expect(
      buildSkillsCertificationCommands({
        ...options,
        baseEnv: { NODE_ENV: 'test', PATH: '/usr/bin:/bin', VIRTUAL_ENV: activeVenv },
      }).backend.command
    ).toBe(activePython);
    expect(
      buildSkillsCertificationCommands({
        ...options,
        baseEnv: { NODE_ENV: 'test', PATH: '/usr/bin:/bin' },
      }).backend.command
    ).toBe(fixture.localPythonPath);

    const missingPythonFixture = createFixture({ localPython: false });
    const missingPythonProfile = createProfile(missingPythonFixture);
    expect(() =>
      buildSkillsCertificationCommands({
        baseEnv: { NODE_ENV: 'test', PATH: '/usr/bin:/bin' },
        extensionRoot: missingPythonFixture.extensionRoot,
        frontendRoot: missingPythonFixture.frontendRoot,
        ports: { backend: 18992, web: 18991 },
        profile: missingPythonProfile,
        repoRoot: missingPythonFixture.repoRoot,
      })
    ).toThrow(/Python.*VIRTUAL_ENV.*\.venv/i);
  });
});

describe('Skills certification bind-conflict classification', () => {
  it.each([
    'listen EADDRINUSE: address already in use 127.0.0.1:18992',
    'Address already in use',
    "ERROR: [Errno 48] error while attempting to bind on address ('127.0.0.1', 18992)",
    'uvicorn startup failed while binding: [Errno 98]',
  ])('accepts confirmed bind-conflict output: %s', (text) => {
    expect(isConfirmedBindConflict(text)).toBe(true);
  });

  it.each([
    "ModuleNotFoundError: No module named 'uvicorn'",
    '401 Unauthorized while initializing single-user auth',
    'backend health check timed out',
    'configuration error: TLDW_CONFIG_FILE is invalid',
    'uvicorn startup failed: [Errno 2] missing file',
    '',
  ])('rejects non-bind startup output: %s', (text) => {
    expect(isConfirmedBindConflict(text)).toBe(false);
  });
});
