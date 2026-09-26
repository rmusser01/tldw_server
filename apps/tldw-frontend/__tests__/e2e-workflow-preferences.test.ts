import type { Page, TestInfo } from '@playwright/test';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createSafeStorage } from '@/utils/safe-storage';
import { seedAuth } from '../e2e/utils/helpers';
import { test as workflowTest } from '../e2e/utils/fixtures';

// Keep fixture initialization real; substitute only Playwright's registration
// boundary so Vitest can invoke it without launching a second test runner.
vi.mock('@playwright/test', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@playwright/test')>();
  return { ...actual, test: { extend: (fixtures: unknown) => fixtures } };
});

type FixtureRegistration = {
  authedPage: (
    dependencies: { page: Page },
    use: (page: Page) => Promise<void>,
    info: TestInfo
  ) => Promise<void>;
};

/** Capture browser initialization scripts and run their actual storage writes. */
function browserBoundary() {
  const scripts: Array<() => void> = [];
  const page = {
    context: () => ({ grantPermissions: async () => undefined }),
    addInitScript: async (fn: (arg: unknown) => void, arg: unknown) => {
      scripts.push(() => fn(arg));
    },
    route: async () => undefined,
    on: () => undefined,
  } as unknown as Page;
  return { page, navigate: () => scripts.forEach((run) => run()) };
}

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal('chrome', undefined);
  vi.stubGlobal('browser', undefined);
});

afterEach(() => {
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe('returning-user workflow preferences', () => {
  it('marks the Notes tutorial seen before yielding the authenticated page', async () => {
    const { page, navigate } = browserBoundary();
    const fixtures = workflowTest as unknown as FixtureRegistration;
    await fixtures.authedPage(
      { page },
      async () => {
        navigate();
        const storage = createSafeStorage({ area: 'local' });
        expect(await storage.get('notes-tutorial-shown')).toBe('1');
      },
      { status: 'passed' } as TestInfo
    );
  });

  it('leaves first-visit tutorial state untouched when seeding authentication alone', async () => {
    const { page, navigate } = browserBoundary();
    await seedAuth(page);
    navigate();
    const storage = createSafeStorage({ area: 'local' });
    expect(await storage.get('notes-tutorial-shown')).toBeUndefined();
  });
});
