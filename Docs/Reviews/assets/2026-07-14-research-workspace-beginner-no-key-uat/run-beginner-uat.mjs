import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { chromium } from "/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/research-workspace-beginner-uat/apps/tldw-frontend/node_modules/playwright/index.mjs";

const CDP_URL = "http://127.0.0.1:18162";
const WEB_URL = "http://127.0.0.1:18161/research-workspace";
const API_ORIGIN = "http://127.0.0.1:18160";
const WEB_ORIGIN = new URL(WEB_URL).origin;
const EVIDENCE_DIR = "/private/tmp/task12968-research-workspace-uat";
const BACKEND_LOG = `${EVIDENCE_DIR}/backend-durable-live.log`;
const ADD_URL_VALUE = "https://example.com/research-workspace-task12968";
const UAT_RUN_ID = `task12968-${randomUUID()}`;
const markerUrl = (stage) => {
  const url = new URL("/api/v1/health", API_ORIGIN);
  url.searchParams.set(`task12968_uat_${stage}`, UAT_RUN_ID);
  return url.toString();
};
const START_MARKER_URL = markerUrl("start");
const END_MARKER_URL = markerUrl("end");

const checkpoints = [];
const diagnostics = {
  startedAt: new Date().toISOString(),
  cdpUrl: CDP_URL,
  webUrl: WEB_URL,
  apiOrigin: API_ORIGIN,
  uatRunId: UAT_RUN_ID,
  markerUrls: { start: START_MARKER_URL, end: END_MARKER_URL },
  browserVersion: null,
  backgroundTargets: [],
  contexts: {},
};

const sanitizeUrl = (value) => {
  try {
    const url = new URL(value);
    return `${url.origin}${url.pathname}${url.search}`;
  } catch {
    return String(value).slice(0, 500);
  }
};

const assertionError = (error) => ({
  name: error instanceof Error ? error.name : "Error",
  message: error instanceof Error ? error.message : String(error),
});

const createContextDiagnostics = () => ({
  console: [],
  pageErrors: [],
  requestFailures: [],
  httpErrors: [],
  apiRequests: [],
  serviceWorkerTargets: [],
  backgroundPageTargets: [],
});

const classifyTrackedApiRequest = (value) => {
  try {
    const url = new URL(value);
    if (url.origin === API_ORIGIN && url.pathname.startsWith("/api/v1/")) {
      return "direct_api";
    }
    if (url.origin !== WEB_ORIGIN) return null;
    if (url.pathname.startsWith("/api/v1/")) return "same_origin_api";
    if (url.pathname.startsWith("/api/proxy/")) return "hosted_proxy";
    return null;
  } catch {
    return null;
  }
};

const isWorkspaceMigrationRequest = (value) => {
  try {
    const pathname = new URL(value).pathname;
    return (
      pathname.startsWith("/api/v1/workspaces/migrations") ||
      pathname.startsWith("/api/proxy/workspaces/migrations")
    );
  } catch {
    return false;
  }
};

const attachDiagnostics = (context, page, bucket) => {
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      bucket.console.push({ type: message.type(), text: message.text().slice(0, 2_000) });
    }
  });
  page.on("pageerror", (error) => {
    bucket.pageErrors.push({ name: error.name, message: error.message.slice(0, 2_000) });
  });
  context.on("requestfailed", (request) => {
    bucket.requestFailures.push({
      method: request.method(),
      url: sanitizeUrl(request.url()),
      failure: request.failure()?.errorText ?? "unknown",
    });
  });
  context.on("request", (request) => {
    const transport = classifyTrackedApiRequest(request.url());
    if (!transport) return;
    const headers = request.headers();
    const names = Object.keys(headers).map((name) => name.toLowerCase());
    bucket.apiRequests.push({
      at: new Date().toISOString(),
      method: request.method(),
      url: sanitizeUrl(request.url()),
      transport,
      hasAuthorization: names.includes("authorization"),
      hasXApiKey: names.includes("x-api-key"),
    });
  });
  context.on("response", (response) => {
    if (response.status() < 400) return;
    bucket.httpErrors.push({
      status: response.status(),
      method: response.request().method(),
      url: sanitizeUrl(response.url()),
    });
  });
  const recordServiceWorker = (worker) => {
    bucket.serviceWorkerTargets.push(sanitizeUrl(worker.url()));
  };
  const recordBackgroundPage = (backgroundPage) => {
    bucket.backgroundPageTargets.push(sanitizeUrl(backgroundPage.url()));
  };
  for (const worker of context.serviceWorkers()) recordServiceWorker(worker);
  for (const backgroundPage of context.backgroundPages()) {
    recordBackgroundPage(backgroundPage);
  }
  context.on("serviceworker", recordServiceWorker);
  context.on("backgroundpage", recordBackgroundPage);
};

const installBackgroundTargetObserver = async (browser) => {
  const client = await browser.newBrowserCDPSession();
  const targetsById = new Map();
  const backgroundTypes = new Set([
    "background_page",
    "service_worker",
    "shared_worker",
  ]);
  const recordTarget = (targetInfo) => {
    if (
      !backgroundTypes.has(targetInfo.type) &&
      !targetInfo.url.startsWith("chrome-extension://")
    ) {
      return;
    }
    targetsById.set(targetInfo.targetId, {
      targetId: targetInfo.targetId,
      type: targetInfo.type,
      url: sanitizeUrl(targetInfo.url),
    });
  };

  client.on("Target.targetCreated", ({ targetInfo }) => recordTarget(targetInfo));
  await client.send("Target.setDiscoverTargets", { discover: true });
  const { targetInfos } = await client.send("Target.getTargets");
  for (const targetInfo of targetInfos) recordTarget(targetInfo);

  return {
    client,
    snapshot: () => Array.from(targetsById.values()),
  };
};

const installReadinessObserver = async (context) => {
  await context.addInitScript(() => {
    window.__TASK12968_UAT_OBSERVATIONS = { readiness: [], published: [] };
    const started = performance.now();
    let lastSignature = null;
    let lastPublishedSignature = null;
    const sample = () => {
      const gate = Array.from(document.querySelectorAll('main[role="status"]')).find(
        (candidate) =>
          /Checking server readiness|Retrying server readiness/i.test(
            candidate.textContent ?? "",
          ),
      );
      if (gate) {
        const signature = gate.textContent?.trim() ?? "";
        if (signature !== lastSignature) {
          lastSignature = signature;
          window.__TASK12968_UAT_OBSERVATIONS.readiness.push({
            atMs: Math.round(performance.now() - started),
            text: signature,
          });
        }
      }
      const published = window.__tldwServerReadinessState;
      const publishedSignature = published ? JSON.stringify(published) : null;
      if (publishedSignature && publishedSignature !== lastPublishedSignature) {
        lastPublishedSignature = publishedSignature;
        window.__TASK12968_UAT_OBSERVATIONS.published.push({
          atMs: Math.round(performance.now() - started),
          ...published,
        });
      }
    };
    window.addEventListener("tldw:server-readiness-state", (event) => {
      const detail = event.detail;
      const signature = detail ? JSON.stringify(detail) : null;
      if (!signature || signature === lastPublishedSignature) return;
      lastPublishedSignature = signature;
      window.__TASK12968_UAT_OBSERVATIONS.published.push({
        atMs: Math.round(performance.now() - started),
        ...detail,
      });
    });
    const timer = window.setInterval(sample, 10);
    window.setTimeout(() => window.clearInterval(timer), 30_000);
  });
};

const inspectSafetyState = async (page) =>
  page.evaluate(() => {
    const isVisible = (element) => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        Number(style.opacity || "1") !== 0 &&
        rect.width > 0 &&
        rect.height > 0
      );
    };
    const portalDetails = Array.from(document.querySelectorAll("nextjs-portal")).map(
      (portal) => {
        const shadowText = portal.shadowRoot?.textContent?.trim() ?? "";
        const hostText = portal.textContent?.trim() ?? "";
        const hasDialogOverlay = Boolean(
          portal.shadowRoot?.querySelector(
            '[data-nextjs-dialog-overlay], [data-nextjs-dialog], [role="dialog"]',
          ),
        );
        const hasRuntimeCopy = /Unhandled Runtime Error|Build Error|Runtime Error|Application error/i.test(
          `${hostText} ${shadowText}`,
        );
        return {
          hostVisible: isVisible(portal),
          hostTextLength: hostText.length,
          shadowTextLength: shadowText.length,
          hasDialogOverlay,
          hasRuntimeCopy,
        };
      },
    );
    const globalBackendDialogs = Array.from(
      document.querySelectorAll('[role="dialog"]'),
    ).filter(
      (dialog) =>
        isVisible(dialog) &&
        /Can['’]t reach your tldw server/i.test(dialog.textContent ?? ""),
    );
    return {
      url: location.href,
      globalBackendDialogCount: globalBackendDialogs.length,
      globalBackendCopyVisible:
        document.body.innerText.includes("Can't reach your tldw server") ||
        document.body.innerText.includes("Can’t reach your tldw server"),
      nextPortalHostCount: portalDetails.length,
      runtimeOverlayCount: portalDetails.filter(
        (portal) => portal.hasDialogOverlay || portal.hasRuntimeCopy,
      ).length,
      portalDetails,
    };
  });

const inspectMobileHeaderLayout = async (page) =>
  page.evaluate(() => {
    const box = (element) => {
      if (!(element instanceof HTMLElement)) return null;
      const rect = element.getBoundingClientRect();
      return {
        x: rect.x,
        y: rect.y,
        right: rect.right,
        bottom: rect.bottom,
        width: rect.width,
        height: rect.height,
      };
    };
    const overlaps = (left, right) =>
      left.x < right.right &&
      left.right > right.x &&
      left.y < right.bottom &&
      left.bottom > right.y;
    const header = document.querySelector('[data-testid="workspace-header"]');
    const contextIndicator = document.querySelector(
      '[data-testid="workspace-server-context-indicator"]',
    );
    const actions = document.querySelector('[data-testid="workspace-header-actions"]');
    const actionButtons = Array.from(actions?.querySelectorAll("button") ?? [])
      .filter((element) => getComputedStyle(element).display !== "none")
      .map((element) => ({
        label:
          element.getAttribute("aria-label") ?? element.textContent?.trim() ?? "",
        box: box(element),
      }))
      .filter((entry) => entry.box);
    const actionOverlaps = [];
    for (let leftIndex = 0; leftIndex < actionButtons.length; leftIndex += 1) {
      for (
        let rightIndex = leftIndex + 1;
        rightIndex < actionButtons.length;
        rightIndex += 1
      ) {
        if (overlaps(actionButtons[leftIndex].box, actionButtons[rightIndex].box)) {
          actionOverlaps.push([
            actionButtons[leftIndex].label,
            actionButtons[rightIndex].label,
          ]);
        }
      }
    }
    return {
      viewportWidth: innerWidth,
      documentWidth: document.documentElement.scrollWidth,
      header: box(header),
      contextIndicator: box(contextIndicator),
      actions: box(actions),
      actionButtons,
      actionOverlaps,
      allActionsInsideViewport: actionButtons.every(
        ({ box: actionBox }) => actionBox.x >= 0 && actionBox.right <= innerWidth,
      ),
    };
  });

const captureCheckpoint = async (
  page,
  name,
  screenshotName,
  details = {},
  screenshotOptions = {},
) => {
  const screenshot = `${EVIDENCE_DIR}/${screenshotName}`;
  if (screenshotOptions.target) {
    await screenshotOptions.target.screenshot({ path: screenshot });
  } else {
    await page.screenshot({
      path: screenshot,
      fullPage: screenshotOptions.fullPage ?? true,
    });
  }
  const safety = await inspectSafetyState(page);
  assert.equal(
    safety.globalBackendDialogCount,
    0,
    `${name}: unexpected global backend-unreachable dialog`,
  );
  assert.equal(
    safety.globalBackendCopyVisible,
    false,
    `${name}: unexpected global backend-unreachable copy`,
  );
  assert.equal(safety.runtimeOverlayCount, 0, `${name}: Next.js runtime overlay visible`);
  checkpoints.push({
    name,
    status: "pass",
    screenshot,
    safety,
    ...details,
  });
  return screenshot;
};

const captureFailure = async (page, name, error) => {
  const screenshot = `${EVIDENCE_DIR}/FAIL-${name.replaceAll(/[^a-z0-9]+/gi, "-")}.png`;
  await page.screenshot({ path: screenshot, fullPage: true }).catch(() => {});
  checkpoints.push({
    name,
    status: "fail",
    screenshot,
    error: assertionError(error),
    safety: await inspectSafetyState(page).catch(() => null),
  });
};

const waitForFocus = async (page, locator, message) => {
  const element = await locator.elementHandle();
  assert.ok(element, `${message}: element was not attached`);
  await page.waitForFunction(
    (candidate) => candidate === document.activeElement,
    element,
    { timeout: 5_000 },
  );
};

const assertCredentialFree = async (context, page, label) => {
  const cookies = await context.cookies();
  assert.equal(
    cookies.length,
    0,
    `${label}: browser cookies must be empty (${cookies
      .map((cookie) => `${cookie.name}@${cookie.domain}${cookie.path}`)
      .join(", ")})`,
  );
  const state = await page.evaluate(async () => {
    const localKeys = Array.from({ length: localStorage.length }, (_, index) =>
      localStorage.key(index),
    ).filter(Boolean);
    const sessionKeys = Array.from({ length: sessionStorage.length }, (_, index) =>
      sessionStorage.key(index),
    ).filter(Boolean);
    const configRaw = localStorage.getItem("tldwConfig");
    let config = null;
    try {
      config = configRaw ? JSON.parse(configRaw) : null;
    } catch {
      config = { parseError: true };
    }
    return {
      localCredentialKeys: localKeys.filter((key) =>
        /api.?key|access.?token|refresh.?token|bearer|auth.?token/i.test(key),
      ),
      sessionCredentialKeys: sessionKeys.filter((key) =>
        /api.?key|access.?token|refresh.?token|bearer|auth.?token/i.test(key),
      ),
      configHasApiKey: Boolean(
        config &&
          typeof config === "object" &&
          ["apiKey", "api_key", "bearer", "accessToken", "access_token"].some(
            (key) => typeof config[key] === "string" && config[key].length > 0,
          ),
      ),
      serviceWorkerScopes:
        "serviceWorker" in navigator
          ? (await navigator.serviceWorker.getRegistrations()).map(
              (registration) => registration.scope,
            )
          : [],
      extensionRuntimeIdPresent: Boolean(
        globalThis.chrome?.runtime?.id || globalThis.browser?.runtime?.id,
      ),
    };
  });
  assert.deepEqual(state.localCredentialKeys, [], `${label}: local credential key found`);
  assert.deepEqual(
    state.sessionCredentialKeys,
    [],
    `${label}: session credential key found`,
  );
  assert.equal(state.configHasApiKey, false, `${label}: tldwConfig contains a credential`);
  assert.deepEqual(state.serviceWorkerScopes, [], `${label}: service worker registered`);
  assert.equal(
    state.extensionRuntimeIdPresent,
    false,
    `${label}: extension runtime could hide background API traffic`,
  );
  return { cookieCount: cookies.length, ...state };
};

const inspectFreshWorkspaceState = async (page, bucket, label) => {
  await page.waitForFunction(() => {
    const raw = localStorage.getItem("tldw-workspace");
    if (!raw) return false;
    try {
      const envelope = JSON.parse(raw);
      const state = envelope?.state;
      return (
        state &&
        typeof state.workspaceId === "string" &&
        state.workspaceId.length > 0 &&
        Array.isArray(state.workspaceIds) &&
        state.workspaceIds.length === 1 &&
        Array.isArray(state.savedWorkspaces) &&
        state.savedWorkspaces.length === 1
      );
    } catch {
      return false;
    }
  }, { timeout: 15_000 });

  const state = await page.evaluate(() => {
    const raw = localStorage.getItem("tldw-workspace");
    const envelope = raw ? JSON.parse(raw) : null;
    const persisted = envelope?.state ?? {};
    const workspaceIds = Array.isArray(persisted.workspaceIds)
      ? persisted.workspaceIds.filter((id) => typeof id === "string")
      : [];
    const savedWorkspaceIds = Array.isArray(persisted.savedWorkspaces)
      ? persisted.savedWorkspaces
          .map((workspace) => workspace?.id)
          .filter((id) => typeof id === "string")
      : [];
    const visibleText = document.body.innerText;
    const migrationStatusMessages = [
      "Checking local workspace data",
      "Legacy workspace data found",
      "Server receipt saved",
      "Local data retained",
      "Migration failed before local deletion",
      "Legacy workspace data moved",
    ].filter((message) => visibleText.includes(message));
    return {
      storageSchema: envelope?.schema ?? "monolithic",
      activeWorkspaceId: persisted.workspaceId ?? null,
      workspaceIds,
      savedWorkspaceIds,
      migrationStatusMessages,
    };
  });
  const migrationApiRequests = bucket.apiRequests.filter((request) =>
    isWorkspaceMigrationRequest(request.url),
  );

  assert.equal(state.workspaceIds.length, 1, `${label}: expected one workspace`);
  assert.equal(
    new Set(state.workspaceIds).size,
    1,
    `${label}: duplicate persisted workspace IDs`,
  );
  assert.equal(
    state.savedWorkspaceIds.length,
    1,
    `${label}: expected one saved workspace`,
  );
  assert.equal(
    state.activeWorkspaceId,
    state.workspaceIds[0],
    `${label}: active workspace does not match persisted workspace`,
  );
  assert.equal(
    state.savedWorkspaceIds[0],
    state.activeWorkspaceId,
    `${label}: saved workspace does not match active workspace`,
  );
  assert.deepEqual(
    state.migrationStatusMessages,
    [],
    `${label}: false migration status is visible`,
  );
  assert.deepEqual(
    migrationApiRequests,
    [],
    `${label}: fresh initialization called the migration API`,
  );

  return {
    ...state,
    workspaceCount: state.workspaceIds.length,
    savedWorkspaceCount: state.savedWorkspaceIds.length,
    migrationApiRequestCount: migrationApiRequests.length,
  };
};

const auditBackendMigrationRequests = async () => {
  const startMarkerNeedle = `task12968_uat_start=${UAT_RUN_ID}`;
  const endMarkerNeedle = `task12968_uat_end=${UAT_RUN_ID}`;
  let raw = "";
  for (let attempt = 0; attempt < 50; attempt += 1) {
    raw = await readFile(BACKEND_LOG, "utf8");
    if (raw.includes(startMarkerNeedle) && raw.includes(endMarkerNeedle)) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  const lines = raw.split(/\r?\n/);
  const startIndex = lines.findIndex((line) => line.includes(startMarkerNeedle));
  let endIndex = -1;
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    if (lines[index].includes(endMarkerNeedle)) {
      endIndex = index;
      break;
    }
  }
  assert.ok(startIndex >= 0, "backend access log is missing the browser start marker");
  assert.ok(endIndex > startIndex, "backend access log is missing the browser end marker");

  const correlatedLines = lines.slice(startIndex, endIndex + 1);
  const timestampPattern = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})/;
  const timestampFor = (line) => {
    const match = line.match(timestampPattern);
    if (!match) return null;
    const at = new Date(match[1].replace(" ", "T"));
    return Number.isFinite(at.getTime()) ? at.toISOString() : null;
  };
  const apiRequestLines = [];
  const migrationLines = [];
  for (const line of correlatedLines) {
    if (line.includes("/api/v1/")) apiRequestLines.push(line.slice(0, 2_000));
    if (line.includes("/api/v1/workspaces/migrations")) {
      migrationLines.push(line.slice(0, 2_000));
    }
  }
  return {
    logPath: BACKEND_LOG,
    runId: UAT_RUN_ID,
    startMarkerUrl: START_MARKER_URL,
    endMarkerUrl: END_MARKER_URL,
    startMarkerLineCount: correlatedLines.filter((line) =>
      line.includes(startMarkerNeedle),
    ).length,
    endMarkerLineCount: correlatedLines.filter((line) =>
      line.includes(endMarkerNeedle),
    ).length,
    windowStartedAt: timestampFor(correlatedLines[0]),
    windowCompletedAt: timestampFor(correlatedLines[correlatedLines.length - 1]),
    correlatedLineCount: correlatedLines.length,
    apiRequestLineCount: apiRequestLines.length,
    apiRequestLines,
    migrationRequestCount: migrationLines.length,
    migrationLines,
  };
};

const prepareContext = async (browser, options, name) => {
  const context = await browser.newContext(options);
  await context.clearCookies();
  await installReadinessObserver(context);
  const page = await context.newPage();
  const client = await context.newCDPSession(page);
  await client.send("Network.clearBrowserCookies");
  await client.send("Storage.clearDataForOrigin", {
    origin: new URL(WEB_URL).origin,
    storageTypes: "all",
  });
  const bucket = createContextDiagnostics();
  diagnostics.contexts[name] = bucket;
  attachDiagnostics(context, page, bucket);
  return { context, page, client, bucket };
};

const emitBackendMarker = async (browser, stage) => {
  const context = await browser.newContext();
  const page = await context.newPage();
  const bucket = createContextDiagnostics();
  diagnostics.contexts[`marker_${stage}`] = bucket;
  attachDiagnostics(context, page, bucket);
  try {
    const url = stage === "start" ? START_MARKER_URL : END_MARKER_URL;
    await page.goto(url, { waitUntil: "load", timeout: 60_000 });
    assert.ok(
      bucket.apiRequests.some((request) => request.url === sanitizeUrl(url)),
      `browser context did not observe the ${stage} marker request`,
    );
    assert.ok(
      bucket.apiRequests.every(
        (request) => !request.hasAuthorization && !request.hasXApiKey,
      ),
      `${stage} marker unexpectedly carried an API credential`,
    );
    assert.deepEqual(
      bucket.serviceWorkerTargets,
      [],
      `${stage} marker context observed a service worker target`,
    );
    assert.deepEqual(
      bucket.backgroundPageTargets,
      [],
      `${stage} marker context observed an extension background page`,
    );
  } finally {
    await context.close();
  }
};

const runDesktop = async (browser) => {
  const { context, page, bucket } = await prepareContext(
    browser,
    { viewport: { width: 1440, height: 1000 }, colorScheme: "dark" },
    "desktop",
  );
  let currentStep = "desktop-direct-entry";
  try {
    const preNavigationCookies = await context.cookies();
    assert.equal(preNavigationCookies.length, 0, "desktop pre-navigation cookies");

    const navigationStartedAt = Date.now();
    await page.goto(WEB_URL, { waitUntil: "commit", timeout: 60_000 });
    const gate = page
      .locator('main[role="status"]')
      .filter({ hasText: /Checking server readiness|Retrying server readiness/i });
    let readinessScreenshot = `${EVIDENCE_DIR}/desktop-01-readiness-gate.png`;
    let readinessGateVisible = false;
    try {
      await gate.waitFor({ state: "visible", timeout: 10_000 });
      readinessGateVisible = true;
      await page.screenshot({ path: readinessScreenshot, fullPage: true });
    } catch {
      readinessScreenshot = `${EVIDENCE_DIR}/desktop-01-direct-entry.png`;
      await page.screenshot({ path: readinessScreenshot, fullPage: true });
    }

    await page.locator("#workspace-main-content").waitFor({ timeout: 60_000 });
    await page.waitForTimeout(750);
    const routeReadyMs = Date.now() - navigationStartedAt;
    const readinessObservations = await page.evaluate(
      () => window.__TASK12968_UAT_OBSERVATIONS ?? { readiness: [], published: [] },
    );
    assert.ok(
      readinessGateVisible ||
        readinessObservations.readiness.length > 0 ||
        readinessObservations.published.some((entry) => entry.state === "ready"),
      "server readiness lifecycle was not observed",
    );
    assert.ok(
      readinessObservations.published.some(
        (entry) => entry.state === "ready" && entry.httpStatus === 200,
      ),
      "server readiness did not publish a healthy HTTP 200 state",
    );
    const credentials = await assertCredentialFree(context, page, "desktop settled");
    assert.ok(
      bucket.apiRequests.length > 0,
      "desktop must make at least one request to the real API",
    );
    assert.ok(
      bucket.apiRequests.every((request) => !request.hasAuthorization && !request.hasXApiKey),
      "desktop API requests unexpectedly carried credentials",
    );
    const freshWorkspaceState = await inspectFreshWorkspaceState(
      page,
      bucket,
      "desktop direct entry",
    );
    await captureCheckpoint(page, currentStep, "desktop-02-settled-workspace.png", {
      readinessGateVisible,
      readinessObservations,
      readinessScreenshot,
      routeReadyMs,
      credentials,
      freshWorkspaceState,
    });

    currentStep = "desktop-sources-empty-state";
    const sources = page.locator("#workspace-sources-panel");
    await sources.getByText("No sources yet", { exact: true }).waitFor();
    await sources
      .getByText(/Add PDFs, web pages, videos, audio, or notes\./)
      .waitFor();
    const sourcesScreenshot = `${EVIDENCE_DIR}/desktop-03-sources-empty.png`;
    await sources.screenshot({ path: sourcesScreenshot });
    await captureCheckpoint(page, currentStep, "desktop-03-sources-empty-full.png", {
      paneScreenshot: sourcesScreenshot,
    });

    currentStep = "desktop-chat-empty-state";
    const chat = page.locator("#workspace-main-content");
    await chat.getByText("Start your research", { exact: true }).waitFor();
    await chat
      .getByText("Your research assistant — grounded in your sources", { exact: true })
      .waitFor();
    const chatScreenshot = `${EVIDENCE_DIR}/desktop-04-chat-empty.png`;
    await chat.screenshot({ path: chatScreenshot });
    await captureCheckpoint(page, currentStep, "desktop-04-chat-empty-full.png", {
      paneScreenshot: chatScreenshot,
    });

    currentStep = "desktop-studio-empty-state";
    const studio = page.locator("#workspace-studio-panel");
    await studio.getByText("Generate outputs from your sources", { exact: true }).waitFor();
    await studio.getByText("No outputs generated yet", { exact: true }).waitFor();
    const studioScreenshot = `${EVIDENCE_DIR}/desktop-05-studio-empty.png`;
    await studio.screenshot({ path: studioScreenshot });
    await captureCheckpoint(page, currentStep, "desktop-05-studio-empty-full.png", {
      paneScreenshot: studioScreenshot,
    });

    currentStep = "desktop-first-run-tour";
    await page.getByRole("button", { name: "Start tour", exact: true }).click();
    const tourTitles = [
      "Workspace Header",
      "Sources Pane",
      "Chat Workspace",
      "Studio Outputs",
      "Workspace Switcher",
    ];
    for (let index = 0; index < tourTitles.length; index += 1) {
      await page.getByText(tourTitles[index], { exact: true }).waitFor({ timeout: 15_000 });
      if (index === 0) {
        await page.waitForTimeout(600);
        await captureCheckpoint(page, currentStep, "desktop-06-first-run-tour.png", {
          stepTitle: tourTitles[index],
          totalSteps: tourTitles.length,
        });
      }
      const primaryTourAction = page.locator(
        'button[data-test-id="button-primary"]',
      );
      assert.match(
        (await primaryTourAction.getAttribute("aria-label")) ?? "",
        index === tourTitles.length - 1 ? /^Finish/ : /^Next/,
      );
      await primaryTourAction.click();
    }
    await page
      .getByText("Workspace Switcher", { exact: true })
      .waitFor({ state: "hidden", timeout: 15_000 });
    assert.equal(
      await page.getByRole("button", { name: "Start tour", exact: true }).count(),
      0,
      "first-run prompt persisted after tour",
    );
    assert.equal(
      await page.evaluate(() =>
        localStorage.getItem("tldw:research-workspace:onboarding-dismissed:v1"),
      ),
      "1",
      "first-run tour dismissal was not persisted",
    );
    await page.waitForTimeout(4_000);
    assert.equal(
      await page
        .getByText("Tour started. Follow the highlighted steps.", { exact: true })
        .count(),
      0,
      "tour feedback persisted in the workspace layout",
    );
    await captureCheckpoint(page, "desktop-completed-tour", "desktop-07-tour-complete.png", {
      completedSteps: tourTitles,
    });

    currentStep = "desktop-replay-tour";
    await page.getByTestId("workspace-settings-button").click();
    await page.getByText("Replay tour", { exact: true }).click();
    await page.getByText("Workspace Header", { exact: true }).waitFor({ timeout: 15_000 });
    await page.waitForTimeout(600);
    await captureCheckpoint(page, currentStep, "desktop-08-replay-tour.png", {
      replayStepTitle: "Workspace Header",
    });
    await page.getByRole("button", { name: "Skip tour" }).click();
    await page
      .getByText("Workspace Header", { exact: true })
      .waitFor({ state: "hidden", timeout: 15_000 });

    currentStep = "desktop-visible-search";
    const searchButton = page.getByTestId("workspace-search-button");
    await searchButton.click();
    const searchDialog = page.getByRole("dialog", { name: "Search workspace" });
    await searchDialog.waitFor();
    const searchInput = searchDialog.getByLabel("Search workspace");
    await searchInput.waitFor();
    await waitForFocus(page, searchInput, "visible search input focus");
    await captureCheckpoint(page, currentStep, "desktop-09-visible-search.png");
    await searchInput.press("Escape");
    await searchDialog.waitFor({ state: "hidden" });
    await waitForFocus(page, searchButton, "visible search focus restoration");

    currentStep = "desktop-shortcut-search";
    await searchButton.focus();
    await page.keyboard.press("Meta+K");
    await searchDialog.waitFor();
    await captureCheckpoint(page, currentStep, "desktop-10-shortcut-search.png");
    await searchDialog.getByLabel("Search workspace").press("Escape");
    await searchDialog.waitFor({ state: "hidden" });
    await waitForFocus(page, searchButton, "shortcut search focus restoration");

    currentStep = "desktop-add-url-no-key-recovery";
    await page.getByRole("button", { name: "Add Sources", exact: true }).first().click();
    const addSourceDialog = page.getByRole("dialog", { name: "Add Sources" });
    await addSourceDialog.waitFor();
    await addSourceDialog.getByRole("tab", { name: "URL", exact: true }).click();
    const urlInput = addSourceDialog.getByPlaceholder(
      "https://example.com/article or YouTube URL",
    );
    await urlInput.fill(ADD_URL_VALUE);
    const requestCountBeforeSubmit = bucket.apiRequests.length;
    await addSourceDialog.getByRole("button", { name: "Add URL", exact: true }).click();
    await addSourceDialog
      .getByText(
        "You do not have permission to add this source. Check your session and retry.",
        { exact: true },
      )
      .waitFor({ timeout: 30_000 });
    assert.equal(await urlInput.inputValue(), ADD_URL_VALUE, "URL was lost after auth failure");
    const submitRequests = bucket.apiRequests.slice(requestCountBeforeSubmit);
    assert.ok(
      submitRequests.every((request) => !request.hasAuthorization && !request.hasXApiKey),
      "Add URL unexpectedly carried an API credential",
    );
    const requestDisposition =
      submitRequests.length === 0 ? "client_auth_guard" : "backend_auth_response";
    await captureCheckpoint(page, currentStep, "desktop-11-add-url-auth-recovery.png", {
      retainedUrl: ADD_URL_VALUE,
      requestDisposition,
      sanitizedRequests: submitRequests,
    });
    await addSourceDialog.getByRole("button", { name: "Close" }).click();
    await addSourceDialog.waitFor({ state: "hidden" });

    const finalFreshWorkspaceState = await inspectFreshWorkspaceState(
      page,
      bucket,
      "desktop final state",
    );
    assert.deepEqual(
      bucket.serviceWorkerTargets,
      [],
      "desktop observed a service worker target",
    );
    assert.deepEqual(
      bucket.backgroundPageTargets,
      [],
      "desktop observed an extension background page",
    );
    await captureCheckpoint(
      page,
      "desktop-final-no-global-modal-or-runtime-overlay",
      "desktop-12-final.png",
      {
        apiRequestCount: bucket.apiRequests.length,
        credentialBearingRequestCount: bucket.apiRequests.filter(
          (request) => request.hasAuthorization || request.hasXApiKey,
        ).length,
        freshWorkspaceState: finalFreshWorkspaceState,
      },
    );
  } catch (error) {
    await captureFailure(page, currentStep, error);
    throw error;
  } finally {
    await context.close();
  }
};

const runMobile = async (browser) => {
  const { context, page, bucket } = await prepareContext(
    browser,
    {
      viewport: { width: 390, height: 844 },
      screen: { width: 390, height: 844 },
      isMobile: true,
      hasTouch: true,
      colorScheme: "dark",
    },
    "mobile",
  );
  let currentStep = "mobile-direct-entry";
  try {
    const navigationStartedAt = Date.now();
    await page.goto(WEB_URL, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await page.getByTestId("workspace-header").waitFor({ timeout: 60_000 });
    await page.getByRole("tablist").last().waitFor({ timeout: 60_000 });
    await page.waitForTimeout(750);
    const routeReadyMs = Date.now() - navigationStartedAt;
    const credentials = await assertCredentialFree(context, page, "mobile settled");
    assert.ok(bucket.apiRequests.length > 0, "mobile must reach the real API");
    assert.ok(
      bucket.apiRequests.every((request) => !request.hasAuthorization && !request.hasXApiKey),
      "mobile API requests unexpectedly carried credentials",
    );
    const freshWorkspaceState = await inspectFreshWorkspaceState(
      page,
      bucket,
      "mobile direct entry",
    );
    const headerLayout = await inspectMobileHeaderLayout(page);
    assert.ok(headerLayout.header, "mobile header geometry unavailable");
    assert.ok(headerLayout.contextIndicator, "mobile context geometry unavailable");
    assert.ok(headerLayout.actions, "mobile actions geometry unavailable");
    assert.ok(
      headerLayout.header.height <= 180,
      `mobile header is too tall: ${headerLayout.header.height}px`,
    );
    assert.ok(
      headerLayout.contextIndicator.width >= 350,
      `mobile context row is squeezed: ${headerLayout.contextIndicator.width}px`,
    );
    assert.ok(
      headerLayout.actions.width >= 350,
      `mobile action row is squeezed: ${headerLayout.actions.width}px`,
    );
    assert.deepEqual(headerLayout.actionOverlaps, [], "mobile header actions overlap");
    assert.equal(
      headerLayout.allActionsInsideViewport,
      true,
      "mobile header action escaped the viewport",
    );
    assert.ok(
      headerLayout.documentWidth <= headerLayout.viewportWidth + 1,
      `mobile page has horizontal overflow: ${headerLayout.documentWidth}/${headerLayout.viewportWidth}`,
    );
    await captureCheckpoint(
      page,
      currentStep,
      "mobile-01-direct-entry.png",
      {
        routeReadyMs,
        credentials,
        headerLayout,
        freshWorkspaceState,
        evidenceScope: "viewport",
      },
      { fullPage: false },
    );

    const dismiss = page.getByRole("button", { name: "Dismiss", exact: true });
    if (await dismiss.isVisible().catch(() => false)) {
      await dismiss.click();
    }

    const tablist = page.getByRole("tablist").last();
    const mobileTabEvidence = {
      Sources: {
        emptyState: "No sources yet",
        screenshot: "mobile-02-sources-tab.png",
      },
      Chat: {
        emptyState: "Start your research",
        screenshot: "mobile-03-chat-tab.png",
      },
      Studio: {
        emptyState: "Generate outputs from your sources",
        screenshot: "mobile-04-studio-tab.png",
      },
    };
    for (const tabName of ["Sources", "Chat", "Studio"]) {
      currentStep = `mobile-${tabName.toLowerCase()}-tab`;
      const tab = tablist.getByRole("tab", { name: new RegExp(`^${tabName}`) });
      await tab.click();
      await tab.waitFor();
      assert.equal(await tab.getAttribute("aria-selected"), "true");
      await page
        .getByText(mobileTabEvidence[tabName].emptyState, { exact: true })
        .first()
        .waitFor({ state: "visible", timeout: 15_000 });
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(400);
      const layout = await page.evaluate(() => ({
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
      }));
      assert.ok(
        layout.scrollWidth <= layout.clientWidth + 1,
        `${tabName} tab has horizontal overflow: ${layout.scrollWidth}/${layout.clientWidth}`,
      );
      await captureCheckpoint(
        page,
        currentStep,
        mobileTabEvidence[tabName].screenshot,
        { layout, evidenceScope: "active_tabpanel" },
        { target: page.locator('[role="tabpanel"]:visible') },
      );
    }

    currentStep = "mobile-search";
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.getByTestId("workspace-search-button").click();
    const searchDialog = page.getByRole("dialog", { name: "Search workspace" });
    await searchDialog.waitFor();
    await searchDialog.getByLabel("Search workspace").waitFor();
    const searchDialogHandle = await searchDialog.elementHandle();
    assert.ok(searchDialogHandle, "mobile search dialog was detached");
    await page.waitForFunction(
      (dialog) => Number.parseFloat(getComputedStyle(dialog).opacity || "0") >= 0.99,
      searchDialogHandle,
      { timeout: 5_000 },
    );
    await captureCheckpoint(
      page,
      currentStep,
      "mobile-05-search.png",
      { evidenceScope: "search_dialog" },
      { target: searchDialog },
    );
    await searchDialog.getByLabel("Search workspace").press("Escape");
    await searchDialog.waitFor({ state: "hidden" });
    await page.waitForTimeout(400);

    await captureCheckpoint(
      page,
      "mobile-final-no-global-modal-or-runtime-overlay",
      "mobile-06-final.png",
      {
        credentialBearingRequestCount: bucket.apiRequests.filter(
          (request) => request.hasAuthorization || request.hasXApiKey,
        ).length,
        evidenceScope: "active_tabpanel",
      },
      { target: page.locator('[role="tabpanel"]:visible') },
    );
    assert.deepEqual(
      bucket.serviceWorkerTargets,
      [],
      "mobile observed a service worker target",
    );
    assert.deepEqual(
      bucket.backgroundPageTargets,
      [],
      "mobile observed an extension background page",
    );
  } catch (error) {
    await captureFailure(page, currentStep, error);
    throw error;
  } finally {
    await context.close();
  }
};

let browser;
let backgroundTargetObserver;
let outcome = "pass";
let failure = null;
try {
  browser = await chromium.connectOverCDP(CDP_URL);
  diagnostics.browserVersion = await browser.version();
  backgroundTargetObserver = await installBackgroundTargetObserver(browser);
  await emitBackendMarker(browser, "start");
  await runDesktop(browser);
  await runMobile(browser);
  await emitBackendMarker(browser, "end");
  const browserApiRequests = Object.values(diagnostics.contexts).flatMap(
    (context) => context.apiRequests,
  );
  assert.ok(
    browserApiRequests.some((request) => request.url === sanitizeUrl(START_MARKER_URL)),
    "browser diagnostics did not observe the backend start marker",
  );
  assert.ok(
    browserApiRequests.some((request) => request.url === sanitizeUrl(END_MARKER_URL)),
    "browser diagnostics did not observe the backend end marker",
  );
  diagnostics.backendAccessLogAudit = await auditBackendMigrationRequests();
  assert.ok(
    diagnostics.backendAccessLogAudit.apiRequestLineCount > 0,
    "backend access log had no API coverage during the UAT window",
  );
  assert.equal(
    diagnostics.backendAccessLogAudit.migrationRequestCount,
    0,
    "backend access log recorded workspace migration traffic during UAT",
  );
  diagnostics.backgroundTargets = backgroundTargetObserver.snapshot();
  assert.deepEqual(
    diagnostics.backgroundTargets,
    [],
    "CDP observed a transient worker or extension background target during UAT",
  );
} catch (error) {
  outcome = "fail";
  failure = assertionError(error);
  process.exitCode = 1;
} finally {
  diagnostics.completedAt = new Date().toISOString();
  diagnostics.outcome = outcome;
  diagnostics.failure = failure;
  diagnostics.checkpointCount = checkpoints.length;
  diagnostics.failedCheckpointCount = checkpoints.filter(
    (checkpoint) => checkpoint.status !== "pass",
  ).length;
  if (backgroundTargetObserver) {
    diagnostics.backgroundTargets = backgroundTargetObserver.snapshot();
  }
  await writeFile(
    `${EVIDENCE_DIR}/checkpoints.json`,
    `${JSON.stringify({ outcome, failure, checkpoints }, null, 2)}\n`,
  );
  await writeFile(
    `${EVIDENCE_DIR}/diagnostics.json`,
    `${JSON.stringify(diagnostics, null, 2)}\n`,
  );
  if (backgroundTargetObserver) await backgroundTargetObserver.client.detach();
  if (browser) await browser.close();
}

process.stdout.write(
  `${JSON.stringify({ outcome, failure, checkpointCount: checkpoints.length }, null, 2)}\n`,
);
