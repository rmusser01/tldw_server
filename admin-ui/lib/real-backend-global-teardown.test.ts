import { beforeEach, describe, expect, it, vi } from 'vitest';

const { stopManagedBackend } = vi.hoisted(() => ({
  stopManagedBackend: vi.fn(),
}));

vi.mock('../tests/e2e/real-backend/helpers/project-env', () => ({
  getRequestedRealBackendProjects: () => [
    'chromium-real-jwt',
    'chromium-real-single-user',
  ],
  shouldManageBackend: () => true,
  getProjectEnv: (projectName: string) => ({ projectName }),
}));

vi.mock('../tests/e2e/real-backend/helpers/backend-lifecycle', () => ({
  stopManagedBackend,
}));

import globalTeardown from '../tests/e2e/real-backend/helpers/global-teardown';

describe('real-backend global teardown', () => {
  beforeEach(() => {
    stopManagedBackend.mockReset();
  });

  it('attempts every managed cleanup before reporting failures', async () => {
    stopManagedBackend
      .mockRejectedValueOnce(new Error('jwt cleanup failed'))
      .mockResolvedValueOnce(undefined);

    await expect(globalTeardown()).rejects.toThrow(
      'Failed to stop 1 managed real-backend process',
    );
    expect(stopManagedBackend).toHaveBeenCalledTimes(2);
  });
});
