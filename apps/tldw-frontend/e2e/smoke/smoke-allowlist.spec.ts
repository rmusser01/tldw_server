import { test, expect } from '@playwright/test';
import { classifySmokeIssues, getCriticalIssues } from './smoke.setup';

const missingRouteDocumentUrl = 'http://localhost:8080/__wayfinding-missing-route__';

for (const scenario of [
  {
    name: 'allows the deliberately missing wayfinding route document 404',
    route: '/__wayfinding-missing-route__',
    url: missingRouteDocumentUrl,
    unexpected: 0,
  },
  {
    name: 'rejects another resource miss on the wayfinding route',
    route: '/__wayfinding-missing-route__',
    url: 'http://127.0.0.1:18323/api/v1/auth/me',
    unexpected: 1,
  },
  {
    name: 'rejects an unlocated resource miss on the wayfinding route',
    route: '/__wayfinding-missing-route__',
    url: undefined,
    unexpected: 1,
  },
  {
    name: 'rejects the missing-route document 404 on another page',
    route: '/unrelated-route',
    url: missingRouteDocumentUrl,
    unexpected: 1,
  },
  {
    name: 'rejects the retired minimal-backend moderation list miss',
    route: '/moderation',
    url: 'http://127.0.0.1:18323/api/v1/moderation/review/items?status=needs_review&sort=newest&limit=50',
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
