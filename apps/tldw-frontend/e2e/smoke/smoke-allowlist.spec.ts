import { test, expect } from '@playwright/test';
import { classifySmokeIssues, getCriticalIssues } from './smoke.setup';

const optionalListUrl =
  'http://127.0.0.1:18323/api/v1/moderation/review/items?status=needs_review&sort=newest&limit=50';

for (const scenario of [
  {
    name: 'rejects an unstubbed moderation list miss',
    route: '/moderation',
    url: optionalListUrl,
    unexpected: 1,
  },
  {
    name: 'rejects another resource miss on the moderation page',
    route: '/moderation',
    url: 'http://127.0.0.1:18323/api/v1/auth/me',
    unexpected: 1,
  },
  {
    name: 'rejects an unlocated resource miss on the moderation page',
    route: '/moderation',
    url: undefined,
    unexpected: 1,
  },
  {
    name: 'rejects the moderation list miss on another page',
    route: '/unrelated-route',
    url: optionalListUrl,
    unexpected: 1,
  },
]) {
  test(scenario.name, () => {
    const issues = getCriticalIssues({
      console: [
        {
          type: 'error',
          text: 'Failed to load resource: the server responded with a status of 404 (Not Found)',
          ...(scenario.url ? { location: { url: scenario.url, lineNumber: 0 } } : {}),
        },
      ],
      pageErrors: [],
      requestFailures: [],
    });

    expect(classifySmokeIssues(scenario.route, issues).unexpectedConsoleErrors).toHaveLength(
      scenario.unexpected
    );
  });
}

for (const text of [
  'Warning: [antd: Drawer] `width` is deprecated. Please use `size` instead.',
  'The above error occurred in the <ForcedRouteErrorProbe> component',
  '[RouteErrorBoundary:kanban] Error: Forced route boundary error for kanban',
]) {
  test(`ordinary route rejects ${text}`, () => {
    const issues = getCriticalIssues({
      console: [{ type: 'error', text }], pageErrors: [], requestFailures: [],
    });
    expect(classifySmokeIssues('/kanban', issues).unexpectedConsoleErrors).toHaveLength(1);
  });
}
