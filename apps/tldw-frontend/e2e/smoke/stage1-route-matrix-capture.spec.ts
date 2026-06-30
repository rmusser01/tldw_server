import fs from 'node:fs/promises';
import path from 'node:path';
import { test, expect, seedAuth, AUTH_CONFIG } from './smoke.setup';
import { waitForAppShell } from '../utils/helpers';
import {
  hasRuntimeOverlayBodySignal,
  hasRuntimeOverlayConsoleSignal,
} from './runtime-overlay';

const LOAD_TIMEOUT = 30_000;
const CHROME_RUNTIME_PATTERNS = [
  /chrome is not defined/i,
  /chrome\.storage/i,
  /Cannot read properties of undefined \(reading ['"]storage['"]\)/i,
];
const MAX_DEPTH_PATTERN = /Maximum update depth exceeded/i;
const TEMPLATE_LEAK_PATTERN = /\{\{[^{}\n]{1,120}\}\}/;

type ManifestRoute = {
  route: string;
};

type ManifestData = {
  routes: ManifestRoute[];
};

type RouteSmokeResult = {
  route: string;
  status: number;
  finalPath: string;
  redirected: boolean;
  hasErrorOverlay: boolean;
  hasChromeRuntimeError: boolean;
  consoleErrorCount: number;
  maxDepthEvents: number;
  hasTemplateLeak: boolean;
  bodySnippet: string;
  consoleErrors: string[];
  pageErrors: string[];
};

function hasPatternMatch(input: string, patterns: RegExp[]): boolean {
  return patterns.some((pattern) => pattern.test(input));
}

function getFinalPath(rawUrl: string): string {
  try {
    const parsed = new URL(rawUrl);
    return `${parsed.pathname}${parsed.search}`;
  } catch {
    return '';
  }
}

function toBodySnippet(bodyText: string): string {
  return bodyText.replace(/\s+/g, ' ').trim().slice(0, 260);
}

test.describe('Stage 1 route matrix capture', () => {
  test('captures audited manifest routes and writes Stage 1 artifact', async ({
    page,
    diagnostics,
  }, testInfo) => {
    test.setTimeout(20 * 60_000);

    const manifestPath = path.resolve(process.cwd(), '../../ux-audit/screenshots-v2/manifest.json');
    const manifestRaw = await fs.readFile(manifestPath, 'utf8');
    const manifest = JSON.parse(manifestRaw) as ManifestData;
    const routes = manifest.routes.map((entry) => entry.route);

    const outputDate = process.env.TLDW_STAGE1_OUTPUT_DATE || new Date().toISOString().slice(0, 10);
    const outputSuffix = process.env.TLDW_STAGE1_OUTPUT_SUFFIX || 'kickoff';
    const outputFileName = `stage1_route_smoke_results_${outputDate}_${outputSuffix}.json`;
    const outputPath = path.resolve(process.cwd(), `../../Docs/Plans/artifacts/${outputFileName}`);

    const results: RouteSmokeResult[] = [];
    await seedAuth(page);

    for (const route of routes) {
      diagnostics.console.length = 0;
      diagnostics.pageErrors.length = 0;
      diagnostics.requestFailures.length = 0;

      let status = 0;
      try {
        const response = await page.goto(route, {
          waitUntil: 'domcontentloaded',
          timeout: LOAD_TIMEOUT,
        });
        status = response?.status() ?? 0;
      } catch {
        status = 0;
      }

      await waitForAppShell(page, LOAD_TIMEOUT);

      const finalPath = getFinalPath(page.url());
      const redirected = finalPath !== '' && finalPath !== route;
      const bodyText = await page.evaluate(() => document.body?.innerText ?? '').catch(() => '');
      const bodySnippet = toBodySnippet(bodyText);

      const consoleErrors = diagnostics.console
        .filter((entry) => entry.type === 'error' || entry.type === 'warning')
        .map((entry) => entry.text);
      const pageErrors = diagnostics.pageErrors.map((entry) => entry.message);

      const maxDepthEvents = [...consoleErrors, ...pageErrors].filter((entry) =>
        MAX_DEPTH_PATTERN.test(entry)
      ).length;
      const hasErrorOverlay =
        [...consoleErrors, ...pageErrors].some(hasRuntimeOverlayConsoleSignal) ||
        hasRuntimeOverlayBodySignal(bodyText);
      const hasChromeRuntimeError = [...consoleErrors, ...pageErrors].some((entry) =>
        hasPatternMatch(entry, CHROME_RUNTIME_PATTERNS)
      );
      const hasTemplateLeak = TEMPLATE_LEAK_PATTERN.test(bodyText);

      results.push({
        route,
        status,
        finalPath,
        redirected,
        hasErrorOverlay,
        hasChromeRuntimeError,
        consoleErrorCount: consoleErrors.length,
        maxDepthEvents,
        hasTemplateLeak,
        bodySnippet,
        consoleErrors,
        pageErrors,
      });
    }

    const summary = {
      totalRoutes: results.length,
      successful: results.filter((entry) => entry.status >= 200 && entry.status < 400).length,
      failed: results.filter((entry) => entry.status < 200 || entry.status >= 400).length,
      redirected: results.filter((entry) => entry.redirected).length,
      withErrorOverlay: results.filter((entry) => entry.hasErrorOverlay).length,
      withConsoleErrors: results.filter((entry) => entry.consoleErrorCount > 0).length,
      withChromeRuntimeErrors: results.filter((entry) => entry.hasChromeRuntimeError).length,
      http404: results.filter((entry) => entry.status === 404).length,
      timeoutStatus0: results.filter((entry) => entry.status === 0).length,
      maxDepthRoutes: results.filter((entry) => entry.maxDepthEvents > 0).length,
      maxDepthEvents: results.reduce((total, entry) => total + entry.maxDepthEvents, 0),
      templateLeakRoutes: results.filter((entry) => entry.hasTemplateLeak).length,
    };

    const artifact = {
      generatedAt: new Date().toISOString(),
      baseUrl: testInfo.project.use.baseURL ?? '',
      backendUrl: AUTH_CONFIG.serverUrl,
      summary,
      routes: results,
    };

    await fs.writeFile(outputPath, `${JSON.stringify(artifact, null, 2)}\n`, 'utf8');
    testInfo.annotations.push({
      type: 'stage1-route-smoke-artifact',
      description: outputPath,
    });

    expect(results.length, 'Manifest route list should not be empty').toBeGreaterThan(0);
  });
});
