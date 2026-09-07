import {
  accessSync,
  constants,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';

export const SKILLS_CERT_API_KEY = 'THIS-IS-A-SECURE-KEY-123-UAT';
export const SKILLS_CERT_NAMES = Object.freeze({
  webui: 'skills-cert-web',
  extension: 'skills-cert-extension',
});

const runtimeMarker = '.skills-certification-runtime';
const safeBaseEnvKeys = Object.freeze([
  'APPDATA',
  'BUN_INSTALL',
  'COMSPEC',
  'CURL_CA_BUNDLE',
  'DBUS_SESSION_BUS_ADDRESS',
  'DISPLAY',
  'DYLD_FALLBACK_LIBRARY_PATH',
  'DYLD_LIBRARY_PATH',
  'HOME',
  'LANG',
  'LC_ALL',
  'LC_CTYPE',
  'LD_LIBRARY_PATH',
  'LOCALAPPDATA',
  'LOGNAME',
  'NODE_EXTRA_CA_CERTS',
  'PATH',
  'PATHEXT',
  'PLAYWRIGHT_BROWSERS_PATH',
  'REQUESTS_CA_BUNDLE',
  'SHELL',
  'SSL_CERT_DIR',
  'SSL_CERT_FILE',
  'SYSTEMROOT',
  'TEMP',
  'TMP',
  'TMPDIR',
  'USER',
  'USERPROFILE',
  'VIRTUAL_ENV',
  'WINDIR',
  'XDG_CACHE_HOME',
  'XDG_CONFIG_HOME',
  'XDG_RUNTIME_DIR',
]);
const chromiumProbeSource =
  'const { chromium } = await import("@playwright/test"); const browser = await chromium.launch({ headless: true }); await browser.close()';

function safeBaseEnvironment(baseEnv) {
  const env = {};
  for (const key of safeBaseEnvKeys) {
    if (typeof baseEnv[key] === 'string') {
      env[key] = baseEnv[key];
    }
  }
  return env;
}

function normalizedIniKey(key) {
  return key
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function scrubHostConfigurationValues(text) {
  return text
    .split(/\r?\n/)
    .map((line) => {
      const match = line.match(/^(\s*(?:[#;]\s*)?)([^=]+?)(\s*=\s*)(.*)$/);
      if (!match) {
        return line;
      }
      const key = normalizedIniKey(match[2]);
      if (!/(?:^|_)api(?:_[a-z0-9]+)*_key(?:_|$)|api_?key$|api_ip$|(?:token|secret|password)$/.test(key)) {
        return line;
      }
      return `${match[1]}${match[2]}${match[3]}`;
    })
    .join('\n');
}

function patchIniValue(text, sectionName, key, value) {
  const lines = text.split(/\r?\n/);
  const sectionIndex = lines.findIndex((line) => {
    const match = line.match(/^\s*\[([^\]]+)]\s*$/);
    return match?.[1].toLowerCase() === sectionName.toLowerCase();
  });

  if (sectionIndex < 0) {
    if (lines.at(-1)?.trim()) {
      lines.push('');
    }
    lines.push(`[${sectionName}]`, `${key} = ${value}`);
    return lines.join('\n');
  }

  let sectionEnd = lines.length;
  for (let index = sectionIndex + 1; index < lines.length; index += 1) {
    if (/^\s*\[[^\]]+]\s*$/.test(lines[index])) {
      sectionEnd = index;
      break;
    }
  }

  for (let index = sectionIndex + 1; index < sectionEnd; index += 1) {
    const match = lines[index].match(/^\s*([^#;][^=]*?)\s*=/);
    if (match?.[1].trim().toLowerCase() === key.toLowerCase()) {
      lines[index] = `${key} = ${value}`;
      return lines.join('\n');
    }
  }

  lines.splice(sectionEnd, 0, `${key} = ${value}`);
  return lines.join('\n');
}

function sqliteUrl(filePath) {
  return `sqlite:///${filePath}`;
}

function writeProfileEnv(profile) {
  writeFileSync(
    profile.envPath,
    [
      'AUTH_MODE=single_user',
      'CONTEXT_INTEGRITY_MODE=audit_only',
      `SINGLE_USER_API_KEY=${SKILLS_CERT_API_KEY}`,
      `DATABASE_URL=${sqliteUrl(profile.usersDbPath)}`,
      `USER_DB_BASE_DIR=${profile.userDatabasesDir}`,
      `USER_DB_BASE_DIR_ALLOWED_ROOTS=${profile.databaseDir}`,
      `TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS=${profile.databaseDir}`,
      '',
    ].join('\n'),
    'utf8'
  );
}

function isExecutable(filePath) {
  try {
    accessSync(filePath, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function resolvePythonExecutable(repoRoot, baseEnv) {
  const candidates = [];
  if (baseEnv.VIRTUAL_ENV) {
    const activeVenv = path.resolve(baseEnv.VIRTUAL_ENV);
    candidates.push(
      path.join(activeVenv, 'bin/python'),
      path.join(activeVenv, 'bin/python3'),
      path.join(activeVenv, 'Scripts/python.exe')
    );
  }

  const localVenv = path.join(repoRoot, '.venv');
  candidates.push(
    path.join(localVenv, 'bin/python'),
    path.join(localVenv, 'bin/python3'),
    path.join(localVenv, 'Scripts/python.exe')
  );

  const executable = candidates.find(isExecutable);
  if (!executable) {
    throw new Error(
      'Python preflight failed: no executable in active VIRTUAL_ENV or worktree-local .venv'
    );
  }
  return executable;
}

/**
 * Create a disposable single-user backend profile below a supplied temporary base.
 */
export function createSkillsCertificationProfile({ repoRoot, temporaryBase, removeRoot = rmSync }) {
  if (!repoRoot) {
    throw new Error('createSkillsCertificationProfile requires repoRoot');
  }
  if (!temporaryBase) {
    throw new Error('createSkillsCertificationProfile requires temporaryBase');
  }

  const resolvedRepoRoot = path.resolve(repoRoot);
  const resolvedTemporaryBase = path.resolve(temporaryBase);
  mkdirSync(resolvedTemporaryBase, { recursive: true });

  const root = mkdtempSync(path.join(resolvedTemporaryBase, 'tldw-skills-certification-'));
  const profile = {
    baseRoot: resolvedTemporaryBase,
    root,
    markerPath: path.join(root, runtimeMarker),
    configDir: path.join(root, 'Config_Files'),
    configPath: path.join(root, 'Config_Files/config.txt'),
    envPath: path.join(root, 'Config_Files/.env'),
    databaseDir: path.join(root, 'Databases'),
    usersDbPath: path.join(root, 'Databases/users.db'),
    userDatabasesDir: path.join(root, 'Databases/user_databases'),
    systemLogPath: path.join(root, 'Databases/system_logs.jsonl'),
    homeDir: path.join(root, 'home'),
    tmpDir: path.join(root, 'tmp'),
    extensionProfileDir: path.join(root, 'extension-profile'),
  };
  try {
    for (const directory of [
      profile.configDir,
      profile.databaseDir,
      profile.userDatabasesDir,
      profile.homeDir,
      profile.tmpDir,
      profile.extensionProfileDir,
    ]) {
      mkdirSync(directory, { recursive: true });
    }
    writeFileSync(profile.markerPath, 'tldw-skills-certification-runtime\n', {
      encoding: 'utf8',
      flag: 'wx',
    });

    const sourceConfigPath = path.join(resolvedRepoRoot, 'tldw_Server_API/Config_Files/config.txt');
    let configText = scrubHostConfigurationValues(readFileSync(sourceConfigPath, 'utf8'));
    configText = patchIniValue(configText, 'Setup', 'enable_first_time_setup', 'true');
    configText = patchIniValue(configText, 'Setup', 'setup_completed', 'true');
    configText = patchIniValue(configText, 'AuthNZ', 'auth_mode', 'single_user');
    configText = patchIniValue(configText, 'AuthNZ', 'single_user_api_key', SKILLS_CERT_API_KEY);
    configText = patchIniValue(
      configText,
      'TTS-Settings',
      'USER_DB_BASE_DIR',
      profile.userDatabasesDir
    );
    configText = patchIniValue(
      configText,
      'Logging',
      'system_log_file_path',
      profile.systemLogPath
    );
    writeFileSync(profile.configPath, configText, 'utf8');
    writeProfileEnv(profile);

    return profile;
  } catch (error) {
    try {
      removeRoot(root, { force: true, recursive: true });
    } catch (cleanupError) {
      const aggregate = new AggregateError(
        [error, cleanupError],
        'Skills certification profile setup failed'
      );
      aggregate.runtime = profile;
      throw aggregate;
    }
    throw error;
  }
}

/**
 * Build independent allowlisted environments for every certification child.
 *
 * @param {{
 *   profile: Record<string, string>,
 *   ports: { backend: number, web: number },
 *   baseEnv?: Record<string, string | undefined>,
 * }} input
 */
export function buildSkillsCertificationEnvironments({ profile, ports, baseEnv = process.env }) {
  if (!profile || !ports?.backend || !ports?.web) {
    throw new Error(
      'buildSkillsCertificationEnvironments requires profile, backend port, and web port'
    );
  }

  const backendUrl = `http://127.0.0.1:${ports.backend}`;
  const webUrl = `http://127.0.0.1:${ports.web}`;
  const jobsDbPath = path.join(profile.databaseDir, 'jobs.db');
  const backendSettings = {
    AUTH_MODE: 'single_user',
    DATABASE_URL: sqliteUrl(profile.usersDbPath),
    HOME: profile.homeDir,
    JOBS_DB_PATH: jobsDbPath,
    JOBS_DB_URL: sqliteUrl(jobsDbPath),
    MCP_MODULES:
      'skills=tldw_Server_API.app.core.MCP_unified.modules.implementations.skills_module:SkillsModule',
    MCP_MODULES_CONFIG: path.join(profile.configDir, 'mcp-modules-disabled.yaml'),
    MEDIA_DB_PATH: path.join(profile.userDatabasesDir, '1', 'Media_DB_v2.db'),
    SINGLE_USER_API_KEY: SKILLS_CERT_API_KEY,
    SYSTEM_LOG_FILE_PATH: profile.systemLogPath,
    TEMP: profile.tmpDir,
    TLDW_CONFIG_FILE: profile.configPath,
    TLDW_ENV_FILE: profile.envPath,
    TLDW_ENV_FILE_EXCLUSIVE: '1',
    TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS: profile.databaseDir,
    TMP: profile.tmpDir,
    TMPDIR: profile.tmpDir,
    USER_DB_BASE_DIR: profile.userDatabasesDir,
    USER_DB_BASE_DIR_ALLOWED_ROOTS: profile.databaseDir,
  };

  return {
    authInitEnv: {
      ...safeBaseEnvironment(baseEnv),
      ...backendSettings,
    },
    backendEnv: {
      ...safeBaseEnvironment(baseEnv),
      ...backendSettings,
      CONTEXT_INTEGRITY_MODE: 'audit_only',
    },
    frontendEnv: {
      ...safeBaseEnvironment(baseEnv),
      NEXT_PUBLIC_API_URL: backendUrl,
      NEXT_PUBLIC_API_VERSION: 'v1',
      NEXT_PUBLIC_X_API_KEY: SKILLS_CERT_API_KEY,
      TLDW_SERVER_URL: backendUrl,
      TLDW_WEB_URL: webUrl,
    },
    webuiPlaywrightEnv: {
      ...safeBaseEnvironment(baseEnv),
      TLDW_E2E_API_KEY: SKILLS_CERT_API_KEY,
      TLDW_E2E_SERVER_URL: backendUrl,
      TLDW_SERVER_URL: backendUrl,
      TLDW_SKILLS_CERT_SKILL_NAME: SKILLS_CERT_NAMES.webui,
      TLDW_WEB_URL: webUrl,
    },
    extensionBuildEnv: {
      ...safeBaseEnvironment(baseEnv),
      TLDW_BUILD_PROFILE: 'production',
      TLDW_E2E_SERVER_URL: backendUrl,
    },
    extensionPlaywrightEnv: {
      ...safeBaseEnvironment(baseEnv),
      TLDW_E2E_API_KEY: SKILLS_CERT_API_KEY,
      TLDW_E2E_SERVER_URL: backendUrl,
      TLDW_E2E_SKIP_EXTENSION_BUILD: '1',
      TLDW_SKILLS_CERT_EXTENSION_PROFILE_ROOT: profile.extensionProfileDir,
      TLDW_SKILLS_CERT_SKILL_NAME: SKILLS_CERT_NAMES.extension,
    },
    webuiChromiumProbeEnv: safeBaseEnvironment(baseEnv),
    extensionChromiumProbeEnv: safeBaseEnvironment(baseEnv),
  };
}

/**
 * Build fixed executable, argv, cwd, and environment records for certification.
 *
 * @param {{
 *   repoRoot: string,
 *   frontendRoot?: string,
 *   extensionRoot?: string,
 *   profile: Record<string, string>,
 *   ports: { backend: number, web: number },
 *   baseEnv?: Record<string, string | undefined>,
 * }} input
 */
export function buildSkillsCertificationCommands({
  repoRoot,
  frontendRoot,
  extensionRoot,
  profile,
  ports,
  baseEnv = process.env,
}) {
  if (!repoRoot) {
    throw new Error('buildSkillsCertificationCommands requires repoRoot');
  }

  const resolvedRepoRoot = path.resolve(repoRoot);
  const resolvedFrontendRoot = path.resolve(
    frontendRoot ?? path.join(resolvedRepoRoot, 'apps/tldw-frontend')
  );
  const resolvedExtensionRoot = path.resolve(
    extensionRoot ?? path.join(resolvedRepoRoot, 'apps/extension')
  );
  const python = resolvePythonExecutable(resolvedRepoRoot, baseEnv);
  const environments = buildSkillsCertificationEnvironments({
    profile,
    ports,
    baseEnv,
  });

  return {
    authInit: {
      name: 'auth-init',
      command: python,
      args: ['-m', 'tldw_Server_API.app.core.AuthNZ.initialize', '--non-interactive'],
      cwd: resolvedRepoRoot,
      env: environments.authInitEnv,
    },
    backend: {
      name: 'backend',
      command: python,
      args: [
        '-m',
        'uvicorn',
        'tldw_Server_API.app.main:app',
        '--host',
        '127.0.0.1',
        '--port',
        String(ports.backend),
      ],
      cwd: resolvedRepoRoot,
      env: environments.backendEnv,
    },
    frontend: {
      name: 'frontend',
      command: 'bun',
      args: ['run', 'dev', '--', '-p', String(ports.web)],
      cwd: resolvedFrontendRoot,
      env: environments.frontendEnv,
    },
    webuiPlaywright: {
      name: 'webui-playwright',
      command: 'bunx',
      args: ['playwright', 'test', '-c', 'e2e/skills-certification/playwright.config.ts'],
      cwd: resolvedFrontendRoot,
      env: environments.webuiPlaywrightEnv,
    },
    extensionBuild: {
      name: 'extension-build',
      command: 'bun',
      args: ['run', 'build:chrome:prod'],
      cwd: resolvedExtensionRoot,
      env: environments.extensionBuildEnv,
    },
    extensionPlaywright: {
      name: 'extension-playwright',
      command: 'bunx',
      args: ['playwright', 'test', '-c', 'playwright.skills-certification.config.ts'],
      cwd: resolvedExtensionRoot,
      env: environments.extensionPlaywrightEnv,
    },
    webuiChromiumProbe: {
      name: 'webui-chromium-probe',
      command: process.execPath,
      args: ['--input-type=module', '--eval', chromiumProbeSource],
      cwd: resolvedFrontendRoot,
      env: environments.webuiChromiumProbeEnv,
    },
    extensionChromiumProbe: {
      name: 'extension-chromium-probe',
      command: process.execPath,
      args: ['--input-type=module', '--eval', chromiumProbeSource],
      cwd: resolvedExtensionRoot,
      env: environments.extensionChromiumProbeEnv,
    },
  };
}

/** Return true only for explicit socket bind-conflict diagnostics. */
export function isConfirmedBindConflict(text) {
  if (typeof text !== 'string' || !text) {
    return false;
  }
  if (/\bEADDRINUSE\b|address already in use/i.test(text)) {
    return true;
  }
  return (
    /\[(?:Errno (?:48|98|10048)|WinError 10048)]/i.test(text) &&
    /uvicorn|bind(?:ing)?|listen|socket/i.test(text)
  );
}
