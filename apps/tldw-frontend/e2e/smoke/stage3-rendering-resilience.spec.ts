import {
  test,
  expect,
  seedAuth,
  seedAdminFixtureProfile,
  getCriticalIssues
} from './smoke.setup'
import { waitForAppShell } from '../utils/helpers'
import type { Route } from '@playwright/test'

const LOAD_TIMEOUT = 30_000
const MAX_DEPTH_PATTERN = /Maximum update depth exceeded/i

type RouteDepthCheck = {
  path: string
  expectedPath?: string
}

const MAX_DEPTH_ROUTES: RouteDepthCheck[] = [
  { path: '/content-review' },
  { path: '/claims-review', expectedPath: '/content-review' },
  { path: '/watchlists' },
  { path: '/research-workspace' }
]

const fulfillJson = async (route: Route, status: number, data: unknown) => {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(data)
  })
}

test.describe('Stage 3 rendering resilience', () => {
  for (const route of MAX_DEPTH_ROUTES) {
    test(`does not emit max update depth warnings on ${route.path}`, async ({
      page,
      diagnostics
    }) => {
      await seedAuth(page)

      const response = await page.goto(route.path, {
        waitUntil: 'domcontentloaded',
        timeout: LOAD_TIMEOUT
      })

      const targetPath = route.expectedPath || route.path
      await page.waitForURL(
        (url) => url.pathname === targetPath,
        { timeout: LOAD_TIMEOUT }
      )
      await waitForAppShell(page, LOAD_TIMEOUT)

      const status = response?.status() ?? 0
      expect(status, `Expected ${route.path} to return HTTP 2xx/3xx`).toBeGreaterThanOrEqual(200)
      expect(status, `Expected ${route.path} to return HTTP 2xx/3xx`).toBeLessThan(400)

      const issues = getCriticalIssues(diagnostics)
      const maxDepthConsole = issues.consoleErrors.filter((entry) =>
        MAX_DEPTH_PATTERN.test(entry.text)
      )
      const maxDepthPageErrors = issues.pageErrors.filter((entry) =>
        MAX_DEPTH_PATTERN.test(entry.message)
      )

      expect(
        maxDepthConsole,
        `Unexpected max update depth console errors on ${route.path}: ${maxDepthConsole
          .map((entry) => entry.text)
          .join(' | ')}`
      ).toHaveLength(0)
      expect(
        maxDepthPageErrors,
        `Unexpected max update depth page errors on ${route.path}: ${maxDepthPageErrors
          .map((entry) => entry.message)
          .join(' | ')}`
      ).toHaveLength(0)
    })
  }

  test('admin stats timeout state shows retry and refetches on demand', async ({ page }, testInfo) => {
    const fixtureBaseUrl =
      typeof testInfo.project.use.baseURL === 'string'
        ? testInfo.project.use.baseURL
        : undefined
    await seedAdminFixtureProfile(page, fixtureBaseUrl)

    let forceStatsFailure = false
    let statsCalls = 0

    await page.route('**/api/v1/admin/stats**', async (route) => {
      statsCalls += 1
      if (forceStatsFailure) {
        await fulfillJson(route, 504, {
          detail: 'timeout while fetching admin stats'
        })
        return
      }
      await fulfillJson(route, 200, {
        users: {},
        storage: {},
        sessions: {}
      })
    })

    await page.route('**/api/v1/admin/users**', async (route) => {
      await fulfillJson(route, 200, {
        users: [
          {
            id: 101,
            uuid: 'fixture-user-101',
            username: 'fixture_admin_user',
            email: 'fixture_admin_user@example.local',
            role: 'admin',
            is_active: true,
            is_verified: true,
            created_at: '2026-02-16T00:00:00Z',
            storage_quota_mb: 5120,
            storage_used_mb: 64
          }
        ],
        total: 1,
        page: 1,
        limit: 20,
        pages: 1
      })
    })

    await page.route('**/api/v1/admin/roles**', async (route) => {
      if (route.request().method().toUpperCase() !== 'GET') {
        await route.continue()
        return
      }
      await fulfillJson(route, 200, [
        {
          id: 1,
          name: 'admin',
          description: 'Fixture admin role',
          is_system: true
        }
      ])
    })

    await page.route('**/api/v1/resource-governor/diag/media-budget**', async (route) => {
      await fulfillJson(route, 200, {
        user_id: 101,
        policy_id: 'media.default',
        limits: {
          daily_bytes_remaining: 2_147_483_648,
          retry_after: 0
        },
        usage: {
          daily_bytes_ingested: 0,
          daily_items_ingested: 0
        }
      })
    })

    await page.goto('/admin/server', {
      waitUntil: 'domcontentloaded',
      timeout: LOAD_TIMEOUT
    })

    await expect(
      page.getByRole('cell', { name: 'fixture_admin_user', exact: true })
    ).toBeVisible({
      timeout: LOAD_TIMEOUT
    })

    const systemStatsCard = page.locator('.ant-card').filter({
      hasText: /System statistics/i
    })
    const refreshButton = systemStatsCard.getByRole('button', { name: /refresh/i })

    forceStatsFailure = true
    statsCalls = 0
    await refreshButton.click()

    await expect.poll(() => statsCalls).toBe(1)

    const retryButton = systemStatsCard.getByRole('button', { name: /retry/i })
    await expect(retryButton).toBeVisible({ timeout: LOAD_TIMEOUT })

    forceStatsFailure = false
    await retryButton.click()

    await expect.poll(() => statsCalls).toBe(2)
    await expect(retryButton).toHaveCount(0)
  })

  test('stt model timeout state shows retry and reloads once per click', async ({ page }) => {
    await seedAuth(page)

    let shouldFailModels = true
    let modelCalls = 0
    await page.route('**/api/v1/media/transcription-models**', async (route) => {
      modelCalls += 1
      if (shouldFailModels) {
        await fulfillJson(route, 504, {
          detail: 'timeout while loading transcription models'
        })
        return
      }
      await fulfillJson(route, 200, {
        all_models: ['whisper-1', 'parakeet-tdt', 'canary']
      })
    })

    await page.goto('/stt', {
      waitUntil: 'domcontentloaded',
      timeout: LOAD_TIMEOUT
    })

    await expect
      .poll(() => modelCalls, {
        message:
          'Expected /api/v1/media/transcription-models interceptor to be hit at least once for /stt'
      })
      .toBeGreaterThan(0)

    const retryButton = page.getByRole('button', { name: /retry/i }).first()
    await expect(retryButton).toBeVisible({ timeout: LOAD_TIMEOUT })

    const callsBeforeRetry = modelCalls
    shouldFailModels = false
    await retryButton.click()

    await expect.poll(() => modelCalls).toBeGreaterThan(callsBeforeRetry)
    await expect(retryButton).toHaveCount(0)
  })

  test('speech stt timeout state shows retry and reloads once per click', async ({ page }) => {
    await seedAuth(page)

    let shouldFailModels = true
    let modelCalls = 0
    await page.route('**/api/v1/media/transcription-models**', async (route) => {
      modelCalls += 1
      if (shouldFailModels) {
        await fulfillJson(route, 504, {
          detail: 'timeout while loading speech transcription models'
        })
        return
      }
      await fulfillJson(route, 200, {
        all_models: ['whisper-1', 'parakeet-tdt', 'canary']
      })
    })

    await page.goto('/speech', {
      waitUntil: 'domcontentloaded',
      timeout: LOAD_TIMEOUT
    })

    await expect
      .poll(() => modelCalls, {
        message:
          'Expected /api/v1/media/transcription-models interceptor to be hit at least once for /speech'
      })
      .toBeGreaterThan(0)

    const retryButton = page.getByRole('button', { name: /retry/i }).first()
    await expect(retryButton).toBeVisible({ timeout: LOAD_TIMEOUT })

    const callsBeforeRetry = modelCalls
    shouldFailModels = false
    await retryButton.click()

    await expect.poll(() => modelCalls).toBeGreaterThan(callsBeforeRetry)
    await expect(retryButton).toHaveCount(0)
  })

  test('settings speech tldw catalog timeout shows retry and refetches once per click', async ({
    page
  }) => {
    await seedAuth(page)
    await page.addInitScript(() => {
      try {
        localStorage.setItem('ttsProvider', 'tldw')
      } catch {}
      try {
        localStorage.setItem('tldwTtsModel', 'kokoro')
      } catch {}
    })

    let providersCalls = 0
    await page.route('**/api/v1/audio/providers**', async (route) => {
      providersCalls += 1
      if (providersCalls <= 2) {
        await fulfillJson(route, 504, {
          detail: 'timeout while loading tldw providers'
        })
        return
      }
      await fulfillJson(route, 200, {
        providers: {
          kokoro: {
            provider_name: 'kokoro',
            formats: ['mp3'],
            languages: ['en'],
            supports_streaming: true,
            voices: [{ id: 'af_heart', name: 'AF Heart' }]
          }
        },
        voices: {
          kokoro: [{ id: 'af_heart', name: 'AF Heart' }]
        }
      })
    })

    await page.route('**/api/v1/audio/voices/catalog**', async (route) => {
      await fulfillJson(route, 200, {
        voices: [{ id: 'af_heart', name: 'AF Heart', provider: 'kokoro' }]
      })
    })

    await page.route('**/api/v1/audio/voices**', async (route) => {
      await fulfillJson(route, 200, {
        voices: [{ id: 'af_heart', name: 'AF Heart', provider: 'kokoro' }]
      })
    })

    await page.goto('/settings/speech', {
      waitUntil: 'domcontentloaded',
      timeout: LOAD_TIMEOUT
    })

    await expect
      .poll(() => providersCalls, {
        message: 'Expected tldw provider catalog endpoint to be called when settings load'
      })
      .toBeGreaterThan(0)

    const tldwCatalogAlert = page.locator('.ant-alert').filter({
      hasText: /tldw voice and model catalog|Unable to load tldw voices\/models/i
    })
    await expect(tldwCatalogAlert).toBeVisible({ timeout: LOAD_TIMEOUT })

    const retryButton = tldwCatalogAlert.getByRole('button', { name: /retry/i })
    const callsBeforeRetry = providersCalls
    await retryButton.click()

    await expect.poll(() => providersCalls).toBeGreaterThan(callsBeforeRetry)
    expect(providersCalls).toBeLessThanOrEqual(callsBeforeRetry + 2)
    await expect(tldwCatalogAlert).toHaveCount(0)
  })
})
