/**
 * Network assertion helpers for E2E tests.
 * Intercepts API calls and lets tests verify buttons fire correct endpoints.
 */
import { type Page, type Request, type Response } from "@playwright/test"

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

export interface ApiCallMatcher {
  method?: string            // "GET" | "POST" | "PUT" | "DELETE" | "PATCH"
  url: string | RegExp       // substring or regex match against request URL
  bodyContains?: Record<string, unknown>  // partial match on JSON body
}

export interface CapturedApiCall {
  method: string
  url: string
  requestBody: unknown
  status: number
  responseBody: unknown
  timestamp: number
}

interface ApiCallResult {
  request: Request
  response: Response
}

/* ------------------------------------------------------------------ */
/* expectApiCall                                                        */
/* ------------------------------------------------------------------ */

/**
 * Returns a promise that resolves when a matching API call is made.
 * Call BEFORE the action that triggers the API call.
 *
 * @example
 * const apiCall = expectApiCall(page, { method: 'POST', url: '/api/v1/notes' });
 * await page.getByRole('button', { name: 'Create' }).click();
 * const { request, response } = await apiCall;
 * expect(response.status()).toBe(200);
 */
export function expectApiCall(
  page: Page,
  matcher: ApiCallMatcher,
  timeoutMs = 15_000
): Promise<ApiCallResult> {
  return new Promise<ApiCallResult>((resolve, reject) => {
    const timer = setTimeout(() => {
      page.removeListener("requestfinished", handler)
      reject(
        new Error(
          `Expected API call matching ${JSON.stringify(matcher)} but none was made within ${timeoutMs}ms`
        )
      )
    }, timeoutMs)

    const handler = async (request: Request) => {
      if (!matchesRequest(request, matcher)) return

      const response = await request.response()
      if (!response) return

      if (matcher.bodyContains) {
        try {
          const body = request.postDataJSON()
          if (!partialMatch(body, matcher.bodyContains)) return
        } catch {
          return
        }
      }

      clearTimeout(timer)
      page.removeListener("requestfinished", handler)
      resolve({ request, response })
    }

    page.on("requestfinished", handler)
  })
}

/* ------------------------------------------------------------------ */
/* expectNoApiCall                                                     */
/* ------------------------------------------------------------------ */

/**
 * Asserts that NO matching API call is made within the timeout.
 * Use to detect dead buttons.
 *
 * @example
 * await page.getByRole('button', { name: 'Broken' }).click();
 * await expectNoApiCall(page, { url: '/api/v1/' }, 3000);
 */
export function expectNoApiCall(
  page: Page,
  matcher: ApiCallMatcher,
  timeoutMs = 3_000
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      page.removeListener("requestfinished", handler)
      resolve()
    }, timeoutMs)

    const handler = async (request: Request) => {
      if (!matchesRequest(request, matcher)) return
      clearTimeout(timer)
      page.removeListener("requestfinished", handler)
      reject(
        new Error(
          `Expected NO API call matching ${JSON.stringify(matcher)} but one was made: ${request.method()} ${request.url()}`
        )
      )
    }

    page.on("requestfinished", handler)
  })
}

/* ------------------------------------------------------------------ */
/* assertApiSequence                                                   */
/* ------------------------------------------------------------------ */

/**
 * Asserts API calls happen in the specified order.
 *
 * @example
 * const sequence = assertApiSequence(page, [
 *   { method: 'POST', url: '/api/v1/media/process' },
 *   { method: 'GET', url: '/api/v1/media/' },
 * ]);
 * await page.getByRole('button', { name: 'Ingest' }).click();
 * await sequence;
 */
export function assertApiSequence(
  page: Page,
  matchers: ApiCallMatcher[],
  timeoutMs = 30_000
): Promise<ApiCallResult[]> {
  return new Promise<ApiCallResult[]>((resolve, reject) => {
    const results: ApiCallResult[] = []
    let currentIndex = 0

    const timer = setTimeout(() => {
      page.removeListener("requestfinished", handler)
      reject(
        new Error(
          `API sequence incomplete: matched ${currentIndex}/${matchers.length}. ` +
          `Waiting for: ${JSON.stringify(matchers[currentIndex])}`
        )
      )
    }, timeoutMs)

    const handler = async (request: Request) => {
      if (currentIndex >= matchers.length) return
      if (!matchesRequest(request, matchers[currentIndex])) return

      const response = await request.response()
      if (!response) return

      results.push({ request, response })
      currentIndex++

      if (currentIndex === matchers.length) {
        clearTimeout(timer)
        page.removeListener("requestfinished", handler)
        resolve(results)
      }
    }

    page.on("requestfinished", handler)
  })
}

/* ------------------------------------------------------------------ */
/* captureAllApiCalls                                                  */
/* ------------------------------------------------------------------ */

/**
 * Starts capturing all /api/ calls. Returns a handle to stop and retrieve.
 *
 * @example
 * const capture = captureAllApiCalls(page);
 * await doSomething();
 * const calls = await capture.stop();
 * expect(calls.length).toBeGreaterThan(0);
 */
export function captureAllApiCalls(page: Page): {
  stop: () => Promise<CapturedApiCall[]>
} {
  const calls: CapturedApiCall[] = []
  const inFlight = new Set<Promise<void>>()

  const captureRequest = async (request: Request): Promise<void> => {
    if (!request.url().includes("/api/")) return
    const response = await request.response()
    if (!response) return

    let requestBody: unknown = null
    try { requestBody = request.postDataJSON() } catch { /* no body */ }

    let responseBody: unknown = null
    try { responseBody = await response.json() } catch { /* non-json */ }

    calls.push({
      method: request.method(),
      url: request.url(),
      requestBody,
      status: response.status(),
      responseBody,
      timestamp: Date.now(),
    })
  }

  const handler = (request: Request): void => {
    const pending = captureRequest(request)
    inFlight.add(pending)
    void pending.finally(() => inFlight.delete(pending))
  }

  page.on("request", handler)

  return {
    stop: async () => {
      page.removeListener("request", handler)
      await Promise.allSettled([...inFlight])
      return calls
    },
  }
}

/* ------------------------------------------------------------------ */
/* getCapturedApiCalls (for fixture integration)                        */
/* ------------------------------------------------------------------ */

const pageCallsMap = new WeakMap<Page, CapturedApiCall[]>()

/**
 * Start auto-capturing API calls for a page (call in fixture setup).
 */
export function startApiCapture(page: Page): void {
  const calls: CapturedApiCall[] = []
  pageCallsMap.set(page, calls)

  page.on("requestfinished", async (request: Request) => {
    if (!request.url().includes("/api/")) return
    let response: Response | null = null
    try {
      response = await request.response()
    } catch {
      return
    }
    if (!response) return

    let requestBody: unknown = null
    try { requestBody = request.postDataJSON() } catch { /* no body */ }

    let responseBody: unknown = null
    try { responseBody = await response.json() } catch { /* non-json */ }

    calls.push({
      method: request.method(),
      url: request.url(),
      requestBody,
      status: response.status(),
      responseBody,
      timestamp: Date.now(),
    })
  })
}

/**
 * Get all captured API calls for a page (call in fixture teardown).
 */
export function getCapturedApiCalls(page: Page): CapturedApiCall[] {
  return pageCallsMap.get(page) ?? []
}

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

function matchesRequest(request: Request, matcher: ApiCallMatcher): boolean {
  if (matcher.method && request.method() !== matcher.method.toUpperCase()) {
    return false
  }
  const url = request.url()
  if (typeof matcher.url === "string") {
    if (!url.includes(matcher.url)) return false
  } else {
    if (!matcher.url.test(url)) return false
  }
  return true
}

function partialMatch(actual: unknown, expected: Record<string, unknown>): boolean {
  if (typeof actual !== "object" || actual === null) return false
  for (const [key, value] of Object.entries(expected)) {
    const actualValue = (actual as Record<string, unknown>)[key]
    if (typeof value === "object" && value !== null) {
      // Deep comparison for nested objects
      if (JSON.stringify(actualValue) !== JSON.stringify(value)) return false
    } else {
      if (actualValue !== value) return false
    }
  }
  return true
}
