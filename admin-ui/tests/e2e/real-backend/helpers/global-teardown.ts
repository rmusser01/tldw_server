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

  const cleanupErrors: unknown[] = [];
  for (const projectName of requestedRealBackendProjects) {
    if (!shouldManageBackend(projectName)) {
      continue;
    }
    try {
      await stopManagedBackend(getProjectEnv(projectName));
    } catch (error) {
      cleanupErrors.push(error);
    }
  }

  if (cleanupErrors.length > 0) {
    throw new AggregateError(
      cleanupErrors,
      `Failed to stop ${cleanupErrors.length} managed real-backend process${cleanupErrors.length === 1 ? '' : 'es'}`,
    );
  }
}
