import AxeBuilder from '@axe-core/playwright';
import { test, expect, seedAuth, getCriticalIssues } from './smoke.setup';
import {
  getRedirectDispositionForA11yScan,
  getStage4HighRiskRouteGovernanceProblems,
  type Stage4HighRiskRoute,
} from './stage4-axe-high-risk-routes.helpers';
import { waitForAppShell, waitForVisualSettle } from '../utils/helpers';
import { getRouteMetadata } from '../../../packages/ui/src/routes/route-metadata';

const LOAD_TIMEOUT = 30_000;
const HOSTED_MODE =
  String(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE || '')
    .trim()
    .toLowerCase() === 'hosted';

const HIGH_RISK_ROUTES: Stage4HighRiskRoute[] = [
  {
    path: '/',
    name: 'Home',
    rationale: 'Default entry route controls setup, health, and primary navigation state.',
  },
  {
    path: '/login',
    name: 'Login',
    rationale: 'Authentication forms and setup fallbacks are high-risk for labels and recovery.',
    mayRedirectWhenUnavailable: true,
    acceptablePaths: HOSTED_MODE ? ['/login'] : ['/login', '/settings/tldw'],
    requiresSeededAuth: false,
  },
  {
    path: '/chat',
    name: 'Chat',
    rationale: 'Primary assistant workflow combines composer, history, and live status controls.',
  },
  { path: "/document-workspace", name: "Document Workspace" },
  { path: "/workflow-editor", name: "Workflow Editor" },
  { path: "/collections", name: "Collections" },
  { path: "/data-tables", name: "Data Tables" },
  { path: "/watchlists", name: "Watchlists" },
  { path: "/evaluations", name: "Evaluations" },
  { path: "/knowledge", name: "Knowledge QA" },
  { path: "/companion", name: "Companion" },
  { path: "/admin/mlx", name: "Admin MLX" },
  { path: "/quick-chat-popout", name: "Quick Chat Popout" },
  { path: "/research-workspace", name: "Research Workspace" },
  { path: "/settings/image-generation", name: "Image Generation Settings" }
]

const STAGE4_A11Y_RULES = [
  'landmark-one-main',
  'region',
  'link-name',
  'image-alt',
  'input-image-alt',
  'select-name',
  'aria-command-name',
  'aria-toggle-field-name',
];

type A11yAnalysisResult =
  | {
      type: 'results';
      results: Awaited<ReturnType<AxeBuilder['analyze']>>;
    }
  | {
      type: 'skip';
      message: string;
    };

async function waitForRouteToSettle(
  page: Parameters<typeof seedAuth>[0],
  expectedPath: string,
  mayRedirectWhenUnavailable: boolean | undefined
): Promise<void> {
  if (mayRedirectWhenUnavailable) {
    try {
      await page.waitForURL((url) => new URL(url.toString()).pathname !== expectedPath, {
        timeout: 1_500,
      });
    } catch {}
  }

  await waitForVisualSettle(page, LOAD_TIMEOUT);
}

async function analyzeA11yWithRetry(
  page: Parameters<typeof seedAuth>[0],
  routePath: string,
  mayRedirectWhenUnavailable: boolean | undefined
): Promise<A11yAnalysisResult> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    let navigationObservedDuringScan = false;
    const onFrameNavigated = (frame: unknown) => {
      if (frame === page.mainFrame()) {
        navigationObservedDuringScan = true;
      }
    };
    page.on('framenavigated', onFrameNavigated);

    try {
      await waitForVisualSettle(page, LOAD_TIMEOUT);
      const results = await new AxeBuilder({ page })
        .withRules(STAGE4_A11Y_RULES)
        .disableRules(['color-contrast'])
        .analyze();
      return {
        type: 'results' as const,
        results,
      };
    } catch (error) {
      lastError = error;
      const message = error instanceof Error ? error.message : String(error);
      if (!message.includes('Execution context was destroyed')) {
        throw error;
      }

      await waitForAppShell(page, LOAD_TIMEOUT);
      await waitForRouteToSettle(page, routePath, mayRedirectWhenUnavailable);

      const disposition = getRedirectDispositionForA11yScan({
        routePath,
        finalPath: new URL(page.url()).pathname,
        mayRedirectWhenUnavailable,
        navigationObservedDuringScan,
      });
      if (disposition.shouldSkip) {
        return {
          type: 'skip' as const,
          message: disposition.message,
        };
      }

      if (attempt === 1) {
        throw error;
      }
    } finally {
      page.off('framenavigated', onFrameNavigated);
    }
  }

  throw lastError instanceof Error ? lastError : new Error(`Axe scan failed for ${routePath}`);
}

async function clearSeededAuth(page: Parameters<typeof seedAuth>[0]): Promise<void> {
  await page.addInitScript(() => {
    try {
      localStorage.removeItem('tldwConfig');
      localStorage.removeItem('__tldw_first_run_complete');
      localStorage.removeItem('__tldw_allow_offline');
    } catch {}
  });
}

function formatAxeViolations(
  routePath: string,
  violations: Awaited<ReturnType<AxeBuilder['analyze']>>['violations']
): string {
  if (violations.length === 0) return `${routePath}: no violations`;
  return [
    `${routePath}: ${violations.length} serious/critical Axe violations`,
    ...violations.map((violation) => {
      const nodes = violation.nodes
        .slice(0, 3)
        .map((node) => node.target.join(' '))
        .join(' | ');
      return `- ${violation.id} [${violation.impact ?? 'unknown'}] -> ${nodes}`;
    }),
  ].join('\n');
}

async function waitForHighRiskRouteReady(
  page: Parameters<typeof seedAuth>[0],
  route: Stage4HighRiskRoute
): Promise<void> {
  if (route.path !== '/login') return;

  await expect
    .poll(
      async () => {
        const loginHeadingVisible = await page
          .getByRole('heading', { name: /^sign in$/i })
          .isVisible()
          .catch(() => false);
        const serverUrlVisible = await page
          .getByLabel(/server url/i)
          .isVisible()
          .catch(() => false);
        const apiKeyVisible = await page
          .getByLabel(/api key/i)
          .isVisible()
          .catch(() => false);
        const loginButtonVisible = await page
          .getByRole('button', { name: /^(login|sign in|verify & login)$/i })
          .isVisible()
          .catch(() => false);

        return loginHeadingVisible || serverUrlVisible || apiKeyVisible || loginButtonVisible;
      },
      {
        timeout: LOAD_TIMEOUT,
        message:
          'Login route did not render either the hosted sign-in form or the shared tldw settings form.',
      }
    )
    .toBe(true);
}

test.describe('Stage 4 Axe high-risk routes', () => {
  test('high-risk route list is metadata-aligned and rationale-backed', () => {
    expect(getStage4HighRiskRouteGovernanceProblems(HIGH_RISK_ROUTES, getRouteMetadata)).toEqual(
      []
    );
  });

  for (const route of HIGH_RISK_ROUTES) {
    test(`${route.name} (${route.path}) passes Stage 4 Axe checks`, async ({
      page,
      diagnostics,
    }) => {
      if (route.requiresSeededAuth === false) {
        await clearSeededAuth(page);
      } else {
        await seedAuth(page);
      }

      const response = await page.goto(route.path, {
        waitUntil: 'domcontentloaded',
        timeout: LOAD_TIMEOUT,
      });
      await waitForAppShell(page, LOAD_TIMEOUT);
      await waitForRouteToSettle(page, route.path, route.mayRedirectWhenUnavailable);

      const status = response?.status() ?? 0;
      test.skip(status >= 400, `Route unavailable in smoke runtime (status ${status})`);

      const acceptablePaths = route.acceptablePaths ?? [route.path];
      const finalPath = new URL(page.url()).pathname;

      if (route.mayRedirectWhenUnavailable && finalPath !== route.path) {
        expect(
          acceptablePaths,
          `Route ${route.path} redirected to unexpected path ${finalPath}`
        ).toContain(finalPath);
        test.skip(
          true,
          `Route ${route.path} redirected to ${finalPath}; feature is unavailable in this runtime`
        );
      }

      const preScanDisposition = getRedirectDispositionForA11yScan({
        routePath: route.path,
        finalPath,
        mayRedirectWhenUnavailable: route.mayRedirectWhenUnavailable,
      });
      if (preScanDisposition.shouldSkip) {
        test.skip(true, preScanDisposition.message);
      }
      expect(
        acceptablePaths,
        `Route ${route.path} resolved to unexpected path ${finalPath}`
      ).toContain(finalPath);
      await waitForHighRiskRouteReady(page, route);

      const issues = getCriticalIssues(diagnostics);
      expect(issues.pageErrors, `Uncaught page errors while scanning ${route.path}`).toHaveLength(
        0
      );

      const analysis = await analyzeA11yWithRetry(
        page,
        route.path,
        route.mayRedirectWhenUnavailable
      );
      if (analysis.type === 'skip') {
        test.skip(true, analysis.message);
        return;
      }
      const results = analysis.results;

      const blockingViolations = results.violations.filter(
        (violation) => violation.impact === 'serious' || violation.impact === 'critical'
      );

      expect(blockingViolations, formatAxeViolations(route.path, blockingViolations)).toEqual([]);
    });
  }
});
