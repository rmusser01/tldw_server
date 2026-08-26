import {
  getProjectEnv,
  getRequestedRealBackendProjects,
  shouldManageBackend,
} from './project-env';
import { stopManagedBackend } from './backend-lifecycle';

export default async function globalTeardown(): Promise<void> {
  const requestedRealBackendProjects = getRequestedRealBackendProjects(process.argv);
  if (requestedRealBackendProjects.length === 0) {
    return;
  }

  for (const projectName of requestedRealBackendProjects) {
    if (!shouldManageBackend(projectName)) {
      continue;
    }
    await stopManagedBackend(getProjectEnv(projectName)).catch(() => undefined);
  }
}
