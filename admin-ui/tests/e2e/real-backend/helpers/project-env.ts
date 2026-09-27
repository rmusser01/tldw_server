export const REAL_BACKEND_PROJECTS = ['chromium-real-jwt', 'chromium-real-single-user'] as const;

export type RealBackendProjectName = (typeof REAL_BACKEND_PROJECTS)[number];
export type RealBackendAuthMode = 'multi_user' | 'single_user';

export type RealBackendProjectEnv = {
  projectName: RealBackendProjectName;
  authMode: RealBackendAuthMode;
  uiPort: number;
  apiPort: number;
  uiBaseUrl: string;
  apiBaseUrl: string;
  serverLabel: string;
};

const PROJECT_ENV: Record<RealBackendProjectName, RealBackendProjectEnv> = {
  'chromium-real-jwt': {
    projectName: 'chromium-real-jwt',
    authMode: 'multi_user',
    uiPort: 3101,
    apiPort: 8101,
    uiBaseUrl: 'http://127.0.0.1:3101',
    apiBaseUrl: 'http://127.0.0.1:8101',
    serverLabel: 'admin-ui-real-jwt',
  },
  'chromium-real-single-user': {
    projectName: 'chromium-real-single-user',
    authMode: 'single_user',
    uiPort: 3102,
    apiPort: 8102,
    uiBaseUrl: 'http://127.0.0.1:3102',
    apiBaseUrl: 'http://127.0.0.1:8102',
    serverLabel: 'admin-ui-real-single-user',
  },
};

type ProjectEnvOverrides = Record<string, string | undefined>;

export const isRealBackendProjectName = (value: string): value is RealBackendProjectName =>
  REAL_BACKEND_PROJECTS.includes(value as RealBackendProjectName);

export const getRequestedRealBackendProjects = (
  argv: readonly string[] = process.argv,
): RealBackendProjectName[] => {
  const requestedProjects: RealBackendProjectName[] = [];

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const projectName = argument.startsWith('--project=')
      ? argument.slice('--project='.length)
      : argument === '--project'
        ? argv[index + 1]
        : undefined;

    if (
      projectName
      && isRealBackendProjectName(projectName)
      && !requestedProjects.includes(projectName)
    ) {
      requestedProjects.push(projectName);
    }
  }

  return requestedProjects;
};

const normalizeLoopbackUrl = (url: string): string => url.replace('localhost', '127.0.0.1');

const getProjectApiOverride = (
  projectName: RealBackendProjectName,
  env: ProjectEnvOverrides,
): string | undefined => {
  if (projectName === 'chromium-real-jwt') {
    return env.TLDW_ADMIN_E2E_JWT_API_URL;
  }

  return env.TLDW_ADMIN_E2E_SINGLE_USER_API_URL;
};

export const shouldManageBackend = (
  projectName: RealBackendProjectName,
  env: ProjectEnvOverrides = process.env,
): boolean => {
  if (env.TLDW_ADMIN_E2E_AUTOSTART_BACKEND === 'false') {
    return false;
  }

  return !getProjectApiOverride(projectName, env);
};

export const getProjectEnv = (
  projectName: string,
  env: ProjectEnvOverrides = process.env,
): RealBackendProjectEnv => {
  if (!isRealBackendProjectName(projectName)) {
    throw new Error(`Unsupported real-backend project: ${projectName}`);
  }

  const baseProjectEnv = PROJECT_ENV[projectName];
  const apiBaseUrl = getProjectApiOverride(projectName, env);

  if (!apiBaseUrl) {
    return baseProjectEnv;
  }

  return {
    ...baseProjectEnv,
    apiBaseUrl: normalizeLoopbackUrl(apiBaseUrl),
  };
};
