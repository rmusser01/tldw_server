import { test, expect, type BrowserContext, type Locator, type Page } from '@playwright/test'
import { launchWithExtensionOrSkip } from "./utils/real-server"
import path from 'path'

const serverUrl = process.env.TLDW_E2E_SERVER_URL || 'http://127.0.0.1:8000'
const apiKey = process.env.TLDW_E2E_API_KEY || ''
const seededWatchlistsConfig = {
  __tldw_first_run_complete: true,
  __tldw_allow_offline: true,
  tldwConfig: {
    serverUrl,
    authMode: 'single-user',
    ...(apiKey ? { apiKey } : {})
  }
}

const installWatchlistsRuntimeBridge = async (context: BrowserContext) => {
  await context.addInitScript(() => {
    if (typeof (window as any).__watchlistsBindBridge === 'function') {
      return
    }

    ;(window as any).__watchlistsBindBridge = (handleRequest, handleUpload) => {
      const patchRuntime = (runtime) => {
        if (!runtime?.sendMessage) return
        const original = runtime.sendMessage.bind(runtime)
        const handler = async (message) => {
          const bridgeHandler =
            message?.type === 'tldw:request'
              ? handleRequest
              : message?.type === 'tldw:upload'
                ? handleUpload
                : null

          if (bridgeHandler) {
            try {
              const payload = message.payload || {}
              const data = await bridgeHandler(payload)
              if (data == null && message?.type === 'tldw:request') {
                const path = String(payload?.path || '')
                const method = String(payload?.method || 'GET').toUpperCase()
                const [pathname, queryString] = path.split('?')
                const now = new Date().toISOString()
                const defaultWatchlist = {
                  id: 1,
                  name: 'E2E Watchlist',
                  description: 'Watchlists E2E default collection',
                  objective: 'Exercise Watchlists extension flows',
                  domain: 'news',
                  status: 'active',
                  priority: 'medium',
                  tags: ['e2e'],
                  created_at: now,
                  updated_at: now
                }

                if (pathname === '/api/v1/watchlists' && method === 'GET') {
                  const params = new URLSearchParams(queryString || '')
                  const page = Number(params.get('page') || 1)
                  const size = Number(params.get('size') || 20)
                  return {
                    ok: true,
                    status: 200,
                    data: {
                      items: [defaultWatchlist],
                      total: 1,
                      page,
                      size,
                      has_more: false,
                    },
                  }
                }

                const watchlistDetailMatch = pathname.match(/^\/api\/v1\/watchlists\/(\d+)$/)
                if (watchlistDetailMatch && method === 'GET') {
                  return {
                    ok: true,
                    status: 200,
                    data: { ...defaultWatchlist, id: Number(watchlistDetailMatch[1]) },
                  }
                }

                const contentAlertsMatch = pathname.match(/^\/api\/v1\/watchlists\/(\d+)\/alerts$/)
                if (contentAlertsMatch && method === 'GET') {
                  const params = new URLSearchParams(queryString || '')
                  const page = Number(params.get('page') || 1)
                  const size = Number(params.get('size') || 20)
                  return {
                    ok: true,
                    status: 200,
                    data: {
                      items: [],
                      total: 0,
                      page,
                      size,
                      has_more: false,
                    },
                  }
                }

                const contentAlertRulesMatch = pathname.match(/^\/api\/v1\/watchlists\/(\d+)\/content-alert-rules$/)
                if (contentAlertRulesMatch && method === 'GET') {
                  const params = new URLSearchParams(queryString || '')
                  const page = Number(params.get('page') || 1)
                  const size = Number(params.get('size') || 20)
                  return {
                    ok: true,
                    status: 200,
                    data: {
                      items: [],
                      total: 0,
                      page,
                      size,
                      has_more: false,
                    },
                  }
                }

                if (pathname === '/api/v1/watchlists/outputs' && method === 'GET') {
                  const params = new URLSearchParams(queryString || '')
                  const page = Number(params.get('page') || 1)
                  const size = Number(params.get('size') || 20)
                  return {
                    ok: true,
                    status: 200,
                    data: {
                      items: [],
                      total: 0,
                      page,
                      size,
                      has_more: false,
                    },
                  }
                }
              }
              if (data == null) {
                return { ok: false, status: 404, error: 'Not found' }
              }
              return { ok: true, status: 200, data }
            } catch (error) {
              return { ok: false, status: 500, error: String(error || '') }
            }
          }

          return original ? original(message) : { ok: true, status: 200, data: {} }
        }

        try {
          runtime.sendMessage = handler
          return
        } catch {}

        try {
          Object.defineProperty(runtime, 'sendMessage', {
            value: handler,
            configurable: true,
            writable: true
          })
        } catch {}
      }

      if (window.chrome?.runtime) {
        patchRuntime(window.chrome.runtime)
      }

      if (window.browser?.runtime) {
        patchRuntime(window.browser.runtime)
      }
    }
  })
}

const watchlistsDestinationConfig = (name: string) => {
  switch (name) {
    case 'Overview':
      return { key: 'overview', label: 'Overview' }
    case 'Feeds':
      return { key: 'sources', label: 'Feeds' }
    case 'Monitors':
      return {
        key: 'jobs',
        label: 'Monitors',
        parentLabel: 'Feeds',
        sectionTestId: 'watchlists-secondary-monitors',
        sectionLabel: 'Monitors'
      }
    case 'Activity':
      return {
        key: 'runs',
        label: 'Activity',
        parentLabel: 'Updates',
        sectionTestId: 'watchlists-secondary-activity',
        sectionLabel: 'Recent Activity'
      }
    case 'Articles':
    case 'Updates':
      return { key: 'items', label: 'Updates' }
    case 'Alerts':
      return { key: 'alerts', label: 'Alerts' }
    case 'Reports':
      return { key: 'outputs', label: 'Reports' }
    case 'Templates':
      return {
        key: 'templates',
        label: 'Templates',
        parentLabel: 'Reports',
        sectionTestId: 'watchlists-secondary-templates',
        sectionLabel: 'Templates'
      }
    case 'Settings':
      return { key: 'settings', label: 'Settings' }
    default:
      return { key: name.toLowerCase(), label: name }
  }
}

const isVisibleLocator = async (locator: Locator) =>
  (await locator.count()) > 0 && await locator.first().isVisible()

const watchlistsContentShell = (page: Page) =>
  page.getByTestId('watchlists-tab-content-shell')

const navigateWatchlistsDestination = async (page: Page, name: string) => {
  await expect(page.getByRole('heading', { name: 'Watchlists' })).toBeVisible({ timeout: 15_000 })
  const destination = watchlistsDestinationConfig(name)
  const experimentButton = page.getByTestId(`watchlists-experimental-tab-${destination.key}`)
  if (await isVisibleLocator(experimentButton)) {
    await experimentButton.first().click()
    return
  }

  const tab = page.getByRole('tab', { name: destination.label })
  if (await isVisibleLocator(tab)) {
    await tab.first().click()
    return
  }

  const shortcutButton = page.getByTestId(`watchlists-task-open-${destination.key}`)
  if (await isVisibleLocator(shortcutButton)) {
    await shortcutButton.first().click()
    return
  }

  const destinationButton = page.getByRole('button', { name: destination.label, exact: true })
  if (await isVisibleLocator(destinationButton)) {
    await destinationButton.first().click()
    return
  }

  if (destination.parentLabel && destination.sectionTestId) {
    const parentTab = page.getByRole('tab', { name: destination.parentLabel })
    if (await isVisibleLocator(parentTab)) {
      await parentTab.first().click()
      const section = page.getByTestId(destination.sectionTestId)
      await expect(section).toBeVisible()
      const toggle = section.getByRole('button', { name: new RegExp(destination.sectionLabel ?? destination.label) })
      if ((await toggle.first().getAttribute('aria-expanded')) !== 'true') {
        await toggle.first().click()
      }
      return
    }
  }

  const trigger = page.getByTestId('watchlists-constrained-nav-trigger')
  await expect(trigger).toBeVisible()
  await trigger.click()
  const drawer = page.getByTestId('watchlists-constrained-nav-drawer')
  await expect(drawer).toBeVisible()
  await drawer.getByRole('button', { name: destination.label }).click()
  await expect(drawer).toBeHidden()
}

const expectWatchlistsDestination = async (page: Page, name: string) => {
  await expect(page.getByRole('heading', { name: 'Watchlists' })).toBeVisible({ timeout: 15_000 })
  const destination = watchlistsDestinationConfig(name)
  const experimentButton = page.getByTestId(`watchlists-experimental-tab-${destination.key}`)
  if (await isVisibleLocator(experimentButton)) {
    await expect(experimentButton.first()).toHaveClass(/ant-btn-primary/)
    return
  }

  const tab = page.getByRole('tab', { name: destination.label })
  if (await isVisibleLocator(tab)) {
    await expect(tab.first()).toHaveAttribute('aria-selected', 'true')
    return
  }

  const destinationButton = page.getByRole('button', { name: destination.label, exact: true })
  if (await isVisibleLocator(destinationButton)) {
    await expect(watchlistsContentShell(page)).toBeVisible()
    return
  }

  if (destination.sectionTestId) {
    const section = page.getByTestId(destination.sectionTestId)
    if (await isVisibleLocator(section)) {
      await expect(section.getByRole('button').first()).toHaveAttribute('aria-expanded', 'true')
      return
    }
  }

  if (destination.key === 'overview') {
    await expect(watchlistsContentShell(page)).toBeVisible()
    return
  }

  await expect(page.getByTestId('watchlists-constrained-nav-trigger')).toContainText(destination.label)
}

const selectRowsAndAssertCount = async (page: Page, expectedCount: number) => {
  const tableCheckboxes = page.locator('.ant-table-tbody tr[data-row-key] .ant-checkbox')
  const tableCheckboxCount = await tableCheckboxes.count()
  if (tableCheckboxCount > 0) {
    await expect(tableCheckboxes).toHaveCount(expectedCount)

    for (let index = 0; index < expectedCount; index += 1) {
      const checkbox = tableCheckboxes.nth(index)
      await checkbox.scrollIntoViewIfNeeded()
      const alreadyChecked = await checkbox.evaluate((node) =>
        node.classList.contains('ant-checkbox-checked')
      )
      if (!alreadyChecked) {
        await checkbox.click({ force: true })
        await expect(checkbox).toHaveClass(/ant-checkbox-checked/)
      }
    }
  } else {
    const constrainedCheckboxes = page.locator('[data-testid^="watchlists-source-card-"] input[type="checkbox"]')
    await expect(constrainedCheckboxes).toHaveCount(expectedCount)
    for (let index = 0; index < expectedCount; index += 1) {
      const checkbox = constrainedCheckboxes.nth(index)
      await checkbox.scrollIntoViewIfNeeded()
      await checkbox.check({ force: true })
      await expect(checkbox).toBeChecked()
    }
  }

  await expect(page.getByText(`${expectedCount} selected`)).toBeVisible()
}

test.describe('Watchlists playground smoke', () => {
  test.describe.configure({ mode: 'serial' })

  test('loads tabs and key flows', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      localStorage.setItem('watchlists:quickSetup:autoshown:v1:1', '1')

      const now = () => new Date().toISOString()
      const sources = [
        {
          id: 1,
          name: 'Tech Daily',
          url: 'https://example.com/rss.xml',
          source_type: 'rss',
          active: true,
          tags: ['tech'],
          created_at: now(),
          updated_at: now(),
          last_scraped_at: now()
        },
        {
          id: 2,
          name: 'World News',
          url: 'https://example.com/world',
          source_type: 'site',
          active: true,
          tags: ['world'],
          created_at: now(),
          updated_at: now(),
          last_scraped_at: null
        }
      ]

      const tags = [
        { id: 1, name: 'tech' },
        { id: 2, name: 'world' }
      ]

      const groups = [
        { id: 10, name: 'News', description: null, parent_group_id: null }
      ]

      const jobs = [
        {
          id: 11,
          name: 'Morning Brief',
          description: 'Daily scan',
          active: true,
          scope: { sources: [1], groups: [], tags: ['tech'] },
          schedule_expr: '0 9 * * *',
          timezone: 'UTC',
          job_filters: { filters: [] },
          created_at: now(),
          updated_at: now(),
          last_run_at: now(),
          next_run_at: null
        }
      ]

      const runs = [
        {
          id: 101,
          job_id: 11,
          status: 'completed',
          started_at: now(),
          finished_at: now(),
          stats: {
            items_found: 2,
            items_ingested: 2,
            items_filtered: 0,
            items_errored: 0
          }
        }
      ]

      const runDetails = {
        id: 101,
        job_id: 11,
        status: 'completed',
        started_at: now(),
        finished_at: now(),
        stats: {
          items_found: 2,
          items_ingested: 2,
          items_filtered: 0,
          items_errored: 0
        },
        filter_tallies: { include: 2 },
        log_text: 'Processing 2 items\nCompleted successfully',
        log_path: null,
        truncated: false,
        filtered_sample: null,
        error_msg: null
      }

      const items = [
        {
          id: 501,
          run_id: 101,
          job_id: 11,
          source_id: 1,
          url: 'https://example.com/article-1',
          title: 'Example Item One',
          summary: 'Summary of item one',
          published_at: now(),
          tags: ['tech'],
          status: 'ingested',
          reviewed: false,
          created_at: now()
        },
        {
          id: 502,
          run_id: 101,
          job_id: 11,
          source_id: 1,
          url: 'https://example.com/article-2',
          title: 'Example Item Two',
          summary: 'Summary of item two',
          published_at: now(),
          tags: ['tech'],
          status: 'ingested',
          reviewed: true,
          created_at: now()
        }
      ]

      const outputs = [
        {
          id: 201,
          run_id: 101,
          job_id: 11,
          type: 'briefing',
          format: 'html',
          title: 'Morning Brief Output',
          version: 1,
          expired: false,
          metadata: {
            deliveries: [
              { channel: 'email', status: 'sent' },
              { channel: 'chatbook', status: 'stored', detail: 'Saved to Chatbook' },
              { channel: 'webhook', status: 'failed', detail: 'timeout' }
            ]
          },
          created_at: now(),
          expires_at: null
        }
      ]

      const templates = [
        {
          name: 'daily-brief',
          format: 'html',
          description: 'Daily summary template',
          updated_at: now()
        }
      ]

      const templateDetails = {
        name: 'daily-brief',
        format: 'html',
        description: 'Daily summary template',
        updated_at: now(),
        content: '<h1>{{ job.name }}</h1>'
      }

      const preview = {
        items: [
          {
            source_id: 1,
            source_type: 'rss',
            url: 'https://example.com/article-1',
            title: 'Example Item One',
            summary: 'Summary',
            published_at: now(),
            decision: 'ingest',
            matched_action: 'include',
            matched_filter_key: null,
            flagged: false
          },
          {
            source_id: 2,
            source_type: 'site',
            url: 'https://example.com/article-2',
            title: 'Example Item Two',
            summary: 'Summary',
            published_at: now(),
            decision: 'filtered',
            matched_action: 'exclude',
            matched_filter_key: null,
            flagged: false
          }
        ],
        total: 2,
        ingestable: 1,
        filtered: 1
      }

      const clusters = [
        {
          id: 301,
          summary: 'Cluster about solar energy',
          canonical_claim_text: null,
          member_count: 6,
          updated_at: now(),
          watchlist_count: 1
        },
        {
          id: 302,
          summary: 'Cluster about supply chain',
          canonical_claim_text: null,
          member_count: 3,
          updated_at: now(),
          watchlist_count: 0
        }
      ]

      const jobClusters = new Map([[11, [{ cluster_id: 301, created_at: now() }]]])

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const body = payload?.body || null
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        if (pathname === '/api/v1/watchlists/tags' && method === 'GET') {
          return paginate(tags, page, size)
        }

        if (pathname === '/api/v1/watchlists/groups' && method === 'GET') {
          return paginate(groups, page, size)
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate(jobs, page, size)
        }

        const jobPreviewMatch = pathname.match(/^\/api\/v1\/watchlists\/jobs\/(\d+)\/preview$/)
        if (jobPreviewMatch && method === 'POST') {
          return preview
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          const q = params.get('q')
          const filtered = q ? runs.filter((r) => r.status === q) : runs
          return paginate(filtered, page, size)
        }

        const jobRunsMatch = pathname.match(/^\/api\/v1\/watchlists\/jobs\/(\d+)\/runs$/)
        if (jobRunsMatch && method === 'GET') {
          const jobId = Number(jobRunsMatch[1])
          const filtered = runs.filter((r) => r.job_id === jobId)
          return paginate(filtered, page, size)
        }

        const runDetailsMatch = pathname.match(/^\/api\/v1\/watchlists\/runs\/(\d+)\/details$/)
        if (runDetailsMatch && method === 'GET') {
          return runDetails
        }

        if (pathname === '/api/v1/watchlists/items' && method === 'GET') {
          const runIdRaw = params.get('run_id')
          const runId =
            runIdRaw && runIdRaw.trim() !== '' ? Number(runIdRaw) : Number.NaN
          const filtered = Number.isNaN(runId)
            ? items
            : items.filter((item) => item.run_id === runId)
          return paginate(filtered, page, size)
        }

        if (pathname === '/api/v1/watchlists/outputs' && method === 'GET') {
          const jobIdRaw = params.get('job_id')
          const runIdRaw = params.get('run_id')
          const jobId =
            jobIdRaw && jobIdRaw.trim() !== '' ? Number(jobIdRaw) : Number.NaN
          const runId =
            runIdRaw && runIdRaw.trim() !== '' ? Number(runIdRaw) : Number.NaN
          let filtered = outputs
          if (!Number.isNaN(jobId)) {
            filtered = filtered.filter((output) => output.job_id === jobId)
          }
          if (!Number.isNaN(runId)) {
            filtered = filtered.filter((output) => output.run_id === runId)
          }
          return paginate(filtered, page, size)
        }

        const downloadMatch = pathname.match(/^\/api\/v1\/watchlists\/outputs\/(\d+)\/download$/)
        if (downloadMatch && method === 'GET') {
          return '<h1>Morning Brief</h1><p>Sample output</p>'
        }

        if (pathname === '/api/v1/watchlists/templates' && method === 'GET') {
          return { items: templates }
        }

        const templateMatch = pathname.match(/^\/api\/v1\/watchlists\/templates\/(.+)$/)
        if (templateMatch && method === 'GET') {
          return templateDetails
        }

        if (pathname === '/api/v1/watchlists/settings' && method === 'GET') {
          return {
            default_output_ttl_seconds: 86400,
            temporary_output_ttl_seconds: 3600
          }
        }

        const watchlistClustersMatch = pathname.match(/^\/api\/v1\/watchlists\/(\d+)\/clusters$/)
        if (watchlistClustersMatch && method === 'GET') {
          const jobId = Number(watchlistClustersMatch[1])
          return {
            watchlist_id: jobId,
            clusters: jobClusters.get(jobId) || []
          }
        }

        if (watchlistClustersMatch && method === 'POST') {
          const jobId = Number(watchlistClustersMatch[1])
          const existing = jobClusters.get(jobId) || []
          const clusterId = Number(body?.cluster_id)
          if (!existing.find((entry) => entry.cluster_id === clusterId)) {
            existing.push({ cluster_id: clusterId, created_at: now() })
            jobClusters.set(jobId, existing)
          }
          return { status: 'added' }
        }

        const watchlistClusterDeleteMatch = pathname.match(
          /^\/api\/v1\/watchlists\/(\d+)\/clusters\/(\d+)$/
        )
        if (watchlistClusterDeleteMatch && method === 'DELETE') {
          const jobId = Number(watchlistClusterDeleteMatch[1])
          const clusterId = Number(watchlistClusterDeleteMatch[2])
          const existing = jobClusters.get(jobId) || []
          jobClusters.set(
            jobId,
            existing.filter((entry) => entry.cluster_id !== clusterId)
          )
          return { status: 'removed' }
        }

        if (pathname === '/api/v1/claims/clusters' && method === 'GET') {
          return clusters
        }

        if (pathname === '/api/v1/watchlists/items/501' && method === 'PATCH') {
          return { ...items[0], reviewed: Boolean(body?.reviewed) }
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)

    })

    const page = await context.newPage()
    page.setDefaultTimeout(15_000)
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await expect(page.getByRole('heading', { name: 'Watchlists' })).toBeVisible()
    await expect(
      page.getByText('Create scheduled monitors to automatically scrape and process content.')
    ).toBeVisible()
    await expectWatchlistsDestination(page, 'Overview')
    await expect(page.getByText('At-a-glance watchlist health')).toBeVisible()

    await navigateWatchlistsDestination(page, 'Feeds')
    await expect(page.getByText('Tech Daily')).toBeVisible()

    const constrainedSources = page.getByTestId('watchlists-sources-constrained-list')
    if (await constrainedSources.isVisible().catch(() => false)) {
      await expect(page.getByTestId('watchlists-sources-constrained-list')).toContainText('World News')
    } else {
      await expect(page.locator('.ant-table-row')).toHaveCount(2)
    }

    await navigateWatchlistsDestination(page, 'Monitors')
    await expect(
      watchlistsContentShell(page).getByText('Morning Brief')
    ).toBeVisible()

    await navigateWatchlistsDestination(page, 'Activity')
    await expect(
      watchlistsContentShell(page).getByText('Showing core columns. Use advanced mode for run metrics.')
    ).toBeVisible()
    await page.getByTestId('watchlists-runs-advanced-toggle').click()
    await expect(
      watchlistsContentShell(page).getByText('Filter by monitor')
    ).toBeVisible()

    await navigateWatchlistsDestination(page, 'Reports')
    await expect(
      watchlistsContentShell(page).getByText('Showing core columns. Use advanced mode for format/run details.')
    ).toBeVisible()
    await expect(
      watchlistsContentShell(page).getByText('+2 more')
    ).toBeVisible()
    await page.getByTestId('watchlists-outputs-advanced-toggle').click()
    await expect(
      watchlistsContentShell(page).getByText('Filter by monitor')
    ).toBeVisible()
    await expect(
      watchlistsContentShell(page).getByText('chatbook Stored')
    ).toBeVisible()

    await context.close()
  })

  test('activity tab cancel action updates run status to cancelled', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true

      const now = () => new Date().toISOString()
      const jobs = [
        {
          id: 11,
          name: 'Morning Brief',
          description: 'Daily scan',
          active: true,
          scope: { sources: [1], groups: [], tags: ['tech'] },
          schedule_expr: '0 9 * * *',
          timezone: 'UTC',
          job_filters: { filters: [] },
          created_at: now(),
          updated_at: now(),
          last_run_at: now(),
          next_run_at: null
        }
      ]

      const runs = [
        {
          id: 101,
          job_id: 11,
          status: 'running',
          started_at: now(),
          finished_at: null,
          stats: {
            items_found: 2,
            items_ingested: 1,
            items_filtered: 0,
            items_errored: 0
          }
        }
      ]

      const runDetails = {
        id: 101,
        job_id: 11,
        status: 'running',
        started_at: now(),
        finished_at: null,
        stats: {
          items_found: 2,
          items_ingested: 1,
          items_filtered: 0,
          items_errored: 0
        },
        filter_tallies: null,
        log_text: 'Running...',
        log_path: null,
        truncated: false,
        filtered_sample: null,
        error_msg: null
      }

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate(jobs, page, size)
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          return paginate(runs, page, size)
        }

        const runDetailsMatch = pathname.match(/^\/api\/v1\/watchlists\/runs\/(\d+)\/details$/)
        if (runDetailsMatch && method === 'GET') {
          return runDetails
        }

        const runCancelMatch = pathname.match(/^\/api\/v1\/watchlists\/runs\/(\d+)\/cancel$/)
        if (runCancelMatch && method === 'POST') {
          const runId = Number(runCancelMatch[1])
          const target = runs.find((run) => run.id === runId)
          if (!target) {
            return { run_id: runId, status: 'missing', cancelled: false, message: 'run_not_found' }
          }
          target.status = 'cancelled'
          target.finished_at = now()
          runDetails.status = 'cancelled'
          runDetails.finished_at = target.finished_at
          runDetails.error_msg = 'cancelled_by_user'
          return { run_id: runId, status: 'cancelled', cancelled: true, message: 'cancel_requested' }
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await navigateWatchlistsDestination(page, 'Activity')
    await expect(page.getByText('Running')).toBeVisible()

    await page.getByTestId('watchlists-run-cancel-101').click()

    await expect(page.getByText('Cancelled', { exact: true }).first()).toBeVisible()

    await context.close()
  })

  test('activity run-details remediation can retry a failed run', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      ;(window as any).__watchlistsRetryRunCalls = 0
      ;(window as any).__watchlistsRetryRunJobIds = []
      ;(window as any).__watchlistsLastRunDetailsId = null

      const now = () => new Date().toISOString()
      const jobs = [
        {
          id: 11,
          name: 'Recovery Monitor',
          description: 'Retries failed runs',
          active: true,
          scope: { sources: [1], groups: [], tags: ['recovery'] },
          schedule_expr: '0 9 * * *',
          timezone: 'UTC',
          job_filters: { filters: [] },
          created_at: now(),
          updated_at: now(),
          last_run_at: now(),
          next_run_at: now()
        }
      ]

      const runs = [
        {
          id: 101,
          job_id: 11,
          status: 'failed',
          started_at: now(),
          finished_at: now(),
          error_msg: '403 forbidden while fetching source',
          stats: {
            items_found: 3,
            items_ingested: 1,
            items_filtered: 1,
            items_errored: 1
          }
        }
      ]

      const runDetailsById: Record<number, Record<string, any>> = {
        101: {
          id: 101,
          job_id: 11,
          status: 'failed',
          started_at: now(),
          finished_at: now(),
          error_msg: '403 forbidden while fetching source',
          stats: {
            items_found: 3,
            items_ingested: 1,
            items_filtered: 1,
            items_errored: 1
          },
          filter_tallies: { include: 1, exclude: 1 },
          log_text: '403 forbidden while fetching source',
          log_path: null,
          truncated: false,
          filtered_sample: [
            {
              title: 'Filtered candidate',
              status: 'filtered',
              matched_action: 'exclude'
            }
          ]
        },
        202: {
          id: 202,
          job_id: 11,
          status: 'pending',
          started_at: now(),
          finished_at: null,
          error_msg: null,
          stats: {
            items_found: 0,
            items_ingested: 0,
            items_filtered: 0,
            items_errored: 0
          },
          filter_tallies: null,
          log_text: 'Retry queued',
          log_path: null,
          truncated: false,
          filtered_sample: null
        }
      }

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate(jobs, page, size)
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          const q = params.get('q')
          const filtered = q ? runs.filter((run) => run.status === q) : runs
          return paginate(filtered, page, size)
        }

        const runDetailsMatch = pathname.match(/^\/api\/v1\/watchlists\/runs\/(\d+)\/details$/)
        if (runDetailsMatch && method === 'GET') {
          const runId = Number(runDetailsMatch[1])
          ;(window as any).__watchlistsLastRunDetailsId = runId
          return runDetailsById[runId] || null
        }

        const triggerRunMatch = pathname.match(/^\/api\/v1\/watchlists\/jobs\/(\d+)\/run$/)
        if (triggerRunMatch && method === 'POST') {
          const jobId = Number(triggerRunMatch[1])
          ;(window as any).__watchlistsRetryRunCalls += 1
          ;(window as any).__watchlistsRetryRunJobIds.push(jobId)
          const retryRun = {
            id: 202,
            job_id: jobId,
            status: 'pending',
            started_at: now(),
            finished_at: null,
            error_msg: null,
            stats: {
              items_found: 0,
              items_ingested: 0,
              items_filtered: 0,
              items_errored: 0
            }
          }
          runs.unshift(retryRun)
          return retryRun
        }

        if (pathname === '/api/v1/watchlists/items' && method === 'GET') {
          return paginate([], page, size)
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await navigateWatchlistsDestination(page, 'Activity')
    await expect(page.getByText('Failed', { exact: true }).first()).toBeVisible()

    await page.getByRole('button', { name: 'View Details' }).first().click()
    await expect(page.getByRole('dialog', { name: 'Run Details' })).toBeVisible()
    await expect(page.getByText('Suggested recovery steps')).toBeVisible()

    await page.getByRole('button', { name: 'Retry run' }).click()

    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsRetryRunCalls))
      .toBe(1)
    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsRetryRunJobIds))
      .toEqual([11])

    await context.close()
  })

  test('overview health and failed-run notification click-through', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      ;(window as any).__TLDW_WATCHLISTS_RUN_NOTIFICATIONS_POLL_MS = 200

      const now = () => new Date().toISOString()
      const sources = [
        {
          id: 1,
          name: 'Alert Feed',
          url: 'https://example.com/alerts.xml',
          source_type: 'rss',
          active: true,
          tags: ['alerts'],
          status: 'error',
          created_at: now(),
          updated_at: now(),
          last_scraped_at: now()
        }
      ]

      const jobs = [
        {
          id: 77,
          name: 'Alert Monitor',
          description: 'Monitors incident sources',
          active: true,
          scope: { sources: [1], groups: [], tags: ['alerts'] },
          schedule_expr: '*/5 * * * *',
          timezone: 'UTC',
          job_filters: { filters: [] },
          created_at: now(),
          updated_at: now(),
          last_run_at: now(),
          next_run_at: now()
        }
      ]

      const runningRun = {
        id: 101,
        job_id: 77,
        status: 'running',
        started_at: now(),
        finished_at: null,
        error_msg: null,
        stats: {
          items_found: 3,
          items_ingested: 1,
          items_filtered: 0,
          items_errored: 0
        }
      }

      const failedRun = {
        id: 101,
        job_id: 77,
        status: 'failed',
        started_at: now(),
        finished_at: now(),
        error_msg: 'Rate limit exceeded while fetching source',
        stats: {
          items_found: 3,
          items_ingested: 1,
          items_filtered: 0,
          items_errored: 1
        }
      }

      const runDetails = {
        ...failedRun,
        filter_tallies: { include: 1 },
        log_text: 'Rate limit exceeded while fetching source',
        log_path: null,
        truncated: false,
        filtered_sample: null
      }

      let runListCalls = 0

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate(jobs, page, size)
        }

        if (pathname === '/api/v1/watchlists/items' && method === 'GET') {
          if (params.get('reviewed') === 'false') {
            return {
              items: [],
              total: 9,
              page,
              size,
              has_more: false
            }
          }
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          const q = params.get('q')
          if (q === 'running') return paginate([runningRun], page, size)
          if (q === 'pending') return paginate([], page, size)
          if (q === 'failed') return paginate([failedRun], page, size)

          runListCalls += 1
          const list = runListCalls === 1 ? [runningRun] : [failedRun]
          return paginate(list, page, size)
        }

        const runDetailsMatch = pathname.match(/^\/api\/v1\/watchlists\/runs\/(\d+)\/details$/)
        if (runDetailsMatch && method === 'GET') {
          return runDetails
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)

    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await expect(page.getByRole('heading', { name: 'Watchlists' })).toBeVisible()
    await expectWatchlistsDestination(page, 'Overview')
    await expect(page.getByText('Setup complete')).toBeVisible()
    await expect(page.getByText('System requires attention')).toBeVisible()
    await expect(page.getByText('Recent Failed Runs')).toBeVisible()
    await expect(page.getByText('Alert Monitor')).toBeVisible()

    const failureNotice = page
      .locator('.ant-notification-notice')
      .filter({ hasText: 'Run failed' })
      .first()
    await expect(failureNotice).toBeVisible({ timeout: 10_000 })
    await expect(failureNotice).toContainText('rate-limiting requests')
    await failureNotice.getByRole('button', { name: 'View run' }).click()

    await expectWatchlistsDestination(page, 'Activity')
    const runDialog = page.getByRole('dialog', { name: 'Run Details' })
    await expect(runDialog).toBeVisible()
    await expect(runDialog.getByText('Rate limit exceeded while fetching source')).toBeVisible()

    await context.close()
  })

  test('overview quick setup callout drives first tab transition', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      try {
        localStorage.setItem('watchlists:quickSetup:autoshown:v1', '1')
      } catch {}

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/items' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          return paginate([], page, size)
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await expect(page.getByText('Add feeds to this Watchlist')).toBeVisible()
    await expect(page.getByText('Create a monitor')).toBeVisible()
    await expect(page.getByText('Review results')).toBeVisible()

    const autoQuickSetupDialog = page.getByRole('dialog', { name: 'Add initial collection' })
    await autoQuickSetupDialog.waitFor({ state: 'visible', timeout: 1_000 }).catch(() => {})
    if (await isVisibleLocator(autoQuickSetupDialog)) {
      await autoQuickSetupDialog.getByRole('button', { name: 'Cancel' }).click()
      await expect(autoQuickSetupDialog).toBeHidden()
    }

    const addFirstFeedButton = page.getByRole('button', { name: 'Add first feed' })
    await expect(addFirstFeedButton).toBeVisible()
    await addFirstFeedButton.click()
    await expectWatchlistsDestination(page, 'Feeds')

    await context.close()
  })

  test('guided quick setup creates feed and monitor in three steps', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      try {
        localStorage.setItem('watchlists:quickSetup:autoshown:v1', '1')
      } catch {}
      ;(window as any).__watchlistsQuickSetup = {
        sourceBody: null,
        jobBody: null,
        run: null
      }

      const now = () => new Date().toISOString()
      const sources: Array<Record<string, any>> = []
      const jobs: Array<Record<string, any>> = []
      const runs: Array<Record<string, any>> = []

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const body = payload?.body || null
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        if (pathname === '/api/v1/watchlists/sources' && method === 'POST') {
          ;(window as any).__watchlistsQuickSetup.sourceBody = body
          const created = {
            id: 501,
            name: String(body?.name || 'Untitled'),
            url: String(body?.url || ''),
            source_type: String(body?.source_type || 'rss'),
            active: body?.active !== false,
            tags: [],
            group_ids: [],
            status: 'healthy',
            created_at: now(),
            updated_at: now(),
            last_scraped_at: null
          }
          sources.unshift(created)
          return created
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate(jobs, page, size)
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'POST') {
          ;(window as any).__watchlistsQuickSetup.jobBody = body
          const created = {
            id: 601,
            name: String(body?.name || 'Untitled Monitor'),
            description: null,
            active: body?.active !== false,
            scope: body?.scope || { sources: [] },
            schedule_expr: body?.schedule_expr || null,
            timezone: body?.timezone || 'UTC',
            job_filters: null,
            output_prefs: null,
            created_at: now(),
            updated_at: now(),
            last_run_at: null,
            next_run_at: null
          }
          jobs.unshift(created)
          return created
        }

        const runTriggerMatch = pathname.match(/^\/api\/v1\/watchlists\/jobs\/(\d+)\/run$/)
        if (runTriggerMatch && method === 'POST') {
          const run = {
            id: 701,
            job_id: Number(runTriggerMatch[1]),
            status: 'running',
            started_at: now(),
            finished_at: null,
            stats: {
              items_found: 0,
              items_ingested: 0,
              items_filtered: 0,
              items_errored: 0
            }
          }
          ;(window as any).__watchlistsQuickSetup.run = run
          runs.unshift(run)
          return run
        }

        if (pathname === '/api/v1/watchlists/items' && method === 'GET') {
          if (params.get('reviewed') === 'false') {
            return { items: [], total: 0, page, size, has_more: false }
          }
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          const q = params.get('q')
          if (q === 'running') return paginate(runs.filter((run) => run.status === 'running'), page, size)
          if (q === 'pending') return paginate([], page, size)
          if (q === 'failed') return paginate([], page, size)
          return paginate(runs, page, size)
        }

        const runDetailsMatch = pathname.match(/^\/api\/v1\/watchlists\/runs\/(\d+)\/details$/)
        if (runDetailsMatch && method === 'GET') {
          const runId = Number(runDetailsMatch[1])
          const run = runs.find((item) => item.id === runId)
          if (!run) return null
          return {
            id: run.id,
            job_id: run.job_id,
            status: run.status,
            started_at: run.started_at,
            finished_at: run.finished_at,
            stats: run.stats || {
              items_found: 0,
              items_ingested: 0,
              items_filtered: 0,
              items_errored: 0
            },
            filter_tallies: null,
            log_text: 'Run started',
            log_path: null,
            truncated: false,
            filtered_sample: null,
            error_msg: null
          }
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    const dialog = page.getByRole('dialog', { name: 'Add initial collection' })
    await dialog.waitFor({ state: 'visible', timeout: 1_000 }).catch(() => {})
    if (!(await isVisibleLocator(dialog))) {
      const guidedSetupButton = page.getByTestId('watchlists-overview-cta-guided-setup')
      await expect(guidedSetupButton).toBeVisible()
      await guidedSetupButton.click()
    }

    await expect(dialog).toBeVisible()
    await dialog.getByPlaceholder('e.g., Daily Tech Feed').fill('Guided Feed')
    await dialog.getByPlaceholder('https://example.com/feed.xml').fill('https://example.com/guided.xml')
    await dialog.getByRole('button', { name: 'Next' }).click()

    await expect(dialog.getByPlaceholder('e.g., Morning Brief')).toBeVisible()
    await dialog.getByPlaceholder('e.g., Morning Brief').fill('Guided Monitor')
    await dialog.getByRole('button', { name: 'Next' }).click()
    const createCollectionButton = dialog.getByRole('button', { name: /Create collection/ })
    await expect(createCollectionButton).toBeVisible()
    await createCollectionButton.click()

    await expectWatchlistsDestination(page, 'Activity')

    const quickSetupState = await page.evaluate(() => (window as any).__watchlistsQuickSetup)
    expect(quickSetupState.sourceBody?.name).toBe('Guided Feed')
    expect(quickSetupState.sourceBody?.url).toBe('https://example.com/guided.xml')
    expect(quickSetupState.jobBody?.name).toBe('Guided Monitor')
    expect(quickSetupState.jobBody?.scope?.sources).toEqual([501])
    expect(quickSetupState.run?.job_id).toBe(601)

    await context.close()
  })

  test('creates a monitor without opening advanced sections', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      ;(window as any).__watchlistsCreatedJobBody = null

      const now = () => new Date().toISOString()
      const sources = [
        {
          id: 1,
          name: 'Tech Daily',
          url: 'https://example.com/rss.xml',
          source_type: 'rss',
          active: true,
          tags: ['tech'],
          status: 'healthy',
          created_at: now(),
          updated_at: now(),
          last_scraped_at: now()
        }
      ]
      const jobs: Array<Record<string, unknown>> = []

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const body = payload?.body || null
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        if (pathname === '/api/v1/watchlists/groups' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/tags' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate(jobs, page, size)
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'POST') {
          ;(window as any).__watchlistsCreatedJobBody = body
          const created = {
            id: 44,
            name: String(body?.name || 'Untitled'),
            description: body?.description || null,
            active: body?.active !== false,
            scope: body?.scope || {},
            schedule_expr: body?.schedule_expr || null,
            timezone: body?.timezone || 'UTC',
            job_filters: body?.job_filters || null,
            output_prefs: body?.output_prefs || null,
            created_at: now(),
            updated_at: now(),
            last_run_at: null,
            next_run_at: null
          }
          jobs.unshift(created)
          return created
        }

        if (pathname === '/api/v1/watchlists/items' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/outputs/templates' && method === 'GET') {
          return { items: [], total: 0 }
        }

        if (pathname === '/api/v1/watchlists/templates' && method === 'GET') {
          return { items: [] }
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})
    console.log('[E2E_STEP] monitor-create: page ready')

    await navigateWatchlistsDestination(page, 'Monitors')
    console.log('[E2E_STEP] monitor-create: monitors tab opened')
    await page.getByRole('button', { name: 'Add Monitor' }).click()
    console.log('[E2E_STEP] monitor-create: add monitor clicked')

    const dialog = page.getByRole('dialog', { name: 'Add Monitor' })
    await expect(dialog).toBeVisible()
    console.log('[E2E_STEP] monitor-create: dialog visible')
    await dialog.getByPlaceholder('e.g., Daily Tech News').fill('Simple Monitor')
    console.log('[E2E_STEP] monitor-create: name filled')

    const scopeHeader = dialog.locator('.ant-collapse-header').filter({ hasText: 'Feeds to Include' }).first()
    await expect(scopeHeader).toBeVisible({ timeout: 10_000 })
    console.log('[E2E_STEP] monitor-create: scope header visible')
    const scopeExpanded = await scopeHeader.getAttribute('aria-expanded', { timeout: 1_000 })
    if (scopeExpanded !== 'true') {
      await scopeHeader.click()
      console.log('[E2E_STEP] monitor-create: scope header expanded')
    }

    await expect(dialog.getByRole('tab', { name: 'Feeds' })).toBeVisible({ timeout: 10_000 })
    console.log('[E2E_STEP] monitor-create: feeds tab visible')
    const scopeSelect = dialog.getByRole('combobox').first()
    await scopeSelect.click()
    console.log('[E2E_STEP] monitor-create: scope select opened')
    const techDailyOption = page
      .locator('.ant-select-dropdown .ant-select-item-option')
      .filter({ hasText: 'Tech Daily' })
      .first()
    await expect(techDailyOption).toBeVisible({ timeout: 10_000 })
    await techDailyOption.click()
    console.log('[E2E_STEP] monitor-create: source selected')
    await expect(dialog.getByTestId('job-form-summary-scope')).not.toContainText('No feeds selected')
    console.log('[E2E_STEP] monitor-create: summary updated')

    await dialog.getByRole('button', { name: 'Create' }).click()
    console.log('[E2E_STEP] monitor-create: create clicked')
    await expect(dialog).not.toBeVisible()
    console.log('[E2E_STEP] monitor-create: dialog closed')
    await expect(page.locator('.ant-table-tbody')).toContainText('Simple Monitor')
    console.log('[E2E_STEP] monitor-create: monitor row visible')

    const createdPayload = await page.evaluate(() => (window as any).__watchlistsCreatedJobBody)
    expect(createdPayload).toBeTruthy()
    expect(createdPayload.name).toBe('Simple Monitor')
    expect(createdPayload.scope?.sources).toEqual([1])
    expect(createdPayload.schedule_expr ?? null).toBeNull()
    expect(
      createdPayload.job_filters == null ||
        !Array.isArray(createdPayload.job_filters?.filters) ||
        createdPayload.job_filters.filters.length === 0
    ).toBe(true)

    await context.close()
  })

  test('articles batch triage controls handle 50+ items efficiently', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig,
      seedLocalStorage: {
        'watchlists:items:page-size': '50',
      },
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      ;(window as any).__watchlistsItemReviewUpdates = 0

      const now = () => new Date().toISOString()
      const sources = [
        {
          id: 1,
          name: 'Triage Feed',
          url: 'https://example.com/triage.xml',
          source_type: 'rss',
          active: true,
          tags: ['triage'],
          status: 'healthy',
          created_at: now(),
          updated_at: now(),
          last_scraped_at: now()
        }
      ]

      const items = Array.from({ length: 55 }, (_, index) => {
        const id = 1000 + index
        return {
          id,
          run_id: 77,
          job_id: 66,
          source_id: 1,
          url: `https://example.com/article-${id}`,
          title: `Batch item ${id}`,
          summary: `Summary for item ${id}`,
          published_at: now(),
          tags: ['triage'],
          status: index % 5 === 0 ? 'filtered' : 'ingested',
          reviewed: false,
          created_at: now()
        }
      })

      const jobs = [
        {
          id: 66,
          name: 'Batch Monitor',
          description: 'Bulk review load',
          active: true,
          scope: { sources: [1], groups: [], tags: ['triage'] },
          schedule_expr: null,
          timezone: 'UTC',
          job_filters: { filters: [] },
          created_at: now(),
          updated_at: now(),
          last_run_at: now(),
          next_run_at: null
        }
      ]

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const applyItemFilters = (params) => {
        let filtered = [...items]
        if (params.get('source_id')) {
          const sourceIdRaw = params.get('source_id')
          const sourceId =
            sourceIdRaw && sourceIdRaw.trim() !== '' ? Number(sourceIdRaw) : Number.NaN
          filtered = filtered.filter((item) => item.source_id === sourceId)
        }
        if (params.get('status')) {
          const status = String(params.get('status'))
          filtered = filtered.filter((item) => item.status === status)
        }
        if (params.get('reviewed') === 'true') {
          filtered = filtered.filter((item) => item.reviewed)
        } else if (params.get('reviewed') === 'false') {
          filtered = filtered.filter((item) => !item.reviewed)
        }
        const query = String(params.get('q') || '').trim().toLowerCase()
        if (query) {
          filtered = filtered.filter((item) =>
            String(item.title || '').toLowerCase().includes(query)
          )
        }
        return filtered
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const body = payload?.body || null
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate(jobs, page, size)
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/items/smart-counts' && method === 'GET') {
          if (params.get('has_alert') === 'true') {
            return {
              all: 0,
              today: 0,
              today_unread: 0,
              unread: 0,
              reviewed: 0,
              queued: 0
            }
          }
          const filtered = applyItemFilters(params)
          const unread = filtered.filter((item) => !item.reviewed).length
          const reviewed = filtered.filter((item) => item.reviewed).length
          return {
            all: filtered.length,
            today: filtered.length,
            today_unread: unread,
            unread,
            reviewed,
            queued: filtered.filter((item) => Boolean(item.queued_for_briefing)).length
          }
        }

        if (pathname === '/api/v1/watchlists/items' && method === 'GET') {
          const filtered = applyItemFilters(params)
          return paginate(filtered, page, size)
        }

        if (pathname === '/api/v1/watchlists/items/batch-update' && method === 'POST') {
          const scopeParams = new URLSearchParams()
          if (body?.scope && typeof body.scope === 'object') {
            Object.entries(body.scope).forEach(([key, value]) => {
              if (value === undefined || value === null || value === '') return
              scopeParams.set(key, String(value))
            })
          }

          const targetIds = Array.isArray(body?.item_ids)
            ? body.item_ids.map((id) => Number(id)).filter((id) => Number.isFinite(id))
            : applyItemFilters(scopeParams).map((item) => item.id)
          const changedIds: number[] = []
          const unchangedIds: number[] = []
          const failedIds: number[] = []

          targetIds.forEach((itemId) => {
            const index = items.findIndex((item) => item.id === itemId)
            if (index < 0) {
              failedIds.push(itemId)
              return
            }
            if (typeof body?.reviewed === 'boolean' && items[index].reviewed !== body.reviewed) {
              items[index].reviewed = body.reviewed
              changedIds.push(itemId)
            } else {
              unchangedIds.push(itemId)
            }
          })

          ;(window as any).__watchlistsItemReviewUpdates += changedIds.length
          return {
            matched: targetIds.length,
            changed: changedIds.length,
            unchanged: unchangedIds.length,
            failed: failedIds.length,
            matched_ids: targetIds,
            changed_ids: changedIds,
            unchanged_ids: unchangedIds,
            failed_ids: failedIds,
            capped: false,
            exhausted: true,
            limit: targetIds.length
          }
        }

        const itemPatchMatch = pathname.match(/^\/api\/v1\/watchlists\/items\/(\d+)$/)
        if (itemPatchMatch && method === 'PATCH') {
          const itemId = Number(itemPatchMatch[1])
          const index = items.findIndex((item) => item.id === itemId)
          if (index < 0) return null
          if (typeof body?.reviewed === 'boolean') {
            items[index].reviewed = body.reviewed
          }
          ;(window as any).__watchlistsItemReviewUpdates += 1
          return { ...items[index] }
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await navigateWatchlistsDestination(page, 'Articles')
    await expect(page.locator('button[data-testid^="watchlists-item-row-"]')).toHaveCount(50)

    await page.getByTestId('watchlists-items-mark-page').click()
    const firstConfirm = page.locator('.ant-modal-confirm').last()
    await firstConfirm.getByRole('button', { name: 'Mark as reviewed' }).click()

    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsItemReviewUpdates))
      .toBe(50)

    await expect(page.getByTestId('watchlists-items-mark-page')).toBeDisabled()

    await page.getByTestId('watchlists-items-mark-all-filtered').click()
    const secondConfirm = page.locator('.ant-modal-confirm').last()
    await secondConfirm.getByRole('button', { name: 'Mark as reviewed' }).click()

    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsItemReviewUpdates))
      .toBe(55)

    await context.close()
  })

  test('articles keyboard shortcuts support mouse-free triage flow', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      ;(window as any).__watchlistsKeyboardReviewUpdates = 0
      ;(window as any).__watchlistsKeyboardItemsFetches = 0
      ;(window as any).__watchlistsOpenedUrls = []

      window.open = ((url?: string | URL | undefined) => {
        if (typeof url === 'string') {
          ;(window as any).__watchlistsOpenedUrls.push(url)
        } else if (url instanceof URL) {
          ;(window as any).__watchlistsOpenedUrls.push(url.toString())
        }
        return null
      }) as typeof window.open

      const now = () => new Date().toISOString()
      const sources = [
        {
          id: 1,
          name: 'Keyboard Feed',
          url: 'https://example.com/keyboard.xml',
          source_type: 'rss',
          active: true,
          tags: ['keyboard'],
          status: 'healthy',
          created_at: now(),
          updated_at: now(),
          last_scraped_at: now()
        }
      ]

      const items = [
        {
          id: 2101,
          run_id: 88,
          job_id: 77,
          source_id: 1,
          url: 'https://example.com/keyboard-1',
          title: 'Keyboard item one',
          summary: 'Summary one',
          published_at: '2026-04-01T10:03:00.000Z',
          tags: ['keyboard'],
          status: 'ingested',
          reviewed: false,
          created_at: '2026-04-01T10:03:00.000Z'
        },
        {
          id: 2102,
          run_id: 88,
          job_id: 77,
          source_id: 1,
          url: 'https://example.com/keyboard-2',
          title: 'Keyboard item two',
          summary: 'Summary two',
          published_at: '2026-04-01T10:02:00.000Z',
          tags: ['keyboard'],
          status: 'ingested',
          reviewed: false,
          created_at: '2026-04-01T10:02:00.000Z'
        },
        {
          id: 2103,
          run_id: 88,
          job_id: 77,
          source_id: 1,
          url: 'https://example.com/keyboard-3',
          title: 'Keyboard item three',
          summary: 'Summary three',
          published_at: '2026-04-01T10:01:00.000Z',
          tags: ['keyboard'],
          status: 'filtered',
          reviewed: true,
          created_at: '2026-04-01T10:01:00.000Z'
        }
      ]

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const applyItemFilters = (params) => {
        let filtered = [...items]
        if (params.get('reviewed') === 'true') {
          filtered = filtered.filter((item) => item.reviewed)
        } else if (params.get('reviewed') === 'false') {
          filtered = filtered.filter((item) => !item.reviewed)
        }
        if (params.get('status')) {
          const status = String(params.get('status'))
          filtered = filtered.filter((item) => item.status === status)
        }
        const query = String(params.get('q') || '').trim().toLowerCase()
        if (query) {
          filtered = filtered.filter((item) =>
            String(item.title || '').toLowerCase().includes(query)
          )
        }
        return filtered
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const body = payload?.body || null
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        if (pathname === '/api/v1/watchlists/groups' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/tags' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/items' && method === 'GET') {
          ;(window as any).__watchlistsKeyboardItemsFetches += 1
          return paginate(applyItemFilters(params), page, size)
        }

        const itemPatchMatch = pathname.match(/^\/api\/v1\/watchlists\/items\/(\d+)$/)
        if (itemPatchMatch && method === 'PATCH') {
          const itemId = Number(itemPatchMatch[1])
          const index = items.findIndex((item) => item.id === itemId)
          if (index < 0) return null
          if (typeof body?.reviewed === 'boolean') {
            items[index].reviewed = body.reviewed
          }
          ;(window as any).__watchlistsKeyboardReviewUpdates += 1
          return { ...items[index] }
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          return paginate([], page, size)
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await navigateWatchlistsDestination(page, 'Articles')
    await expect(page.getByTestId('watchlists-item-reader')).toContainText('Keyboard item one')

    await page.keyboard.press('j')
    await expect(page.getByTestId('watchlists-item-reader')).toContainText('Keyboard item two')

    await page.keyboard.press('k')
    await expect(page.getByTestId('watchlists-item-reader')).toContainText('Keyboard item one')

    await page.getByTestId('watchlists-item-reader').click()
    await page.keyboard.press(' ')
    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsKeyboardReviewUpdates))
      .toBeGreaterThanOrEqual(1)

    const openedUrlsBeforeShortcut = await page.evaluate(
      () => ((window as any).__watchlistsOpenedUrls || []) as string[]
    )
    await page.keyboard.press('o')
    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsOpenedUrls.length))
      .toBe(openedUrlsBeforeShortcut.length + 1)
    await expect
      .poll(async () =>
        page.evaluate(() => {
          const opened = ((window as any).__watchlistsOpenedUrls ?? []) as string[]
          return opened[opened.length - 1] ?? null
        })
      )
      .toBe('https://example.com/keyboard-1')

    const fetchCountBeforeRefresh = await page.evaluate(
      () => (window as any).__watchlistsKeyboardItemsFetches
    )
    await page.keyboard.press('r')
    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsKeyboardItemsFetches))
      .toBeGreaterThan(fetchCountBeforeRefresh)

    await page.keyboard.press('n')
    await expect(page.getByRole('dialog', { name: 'Add Source' })).toBeVisible()

    await context.close()
  })

  test('feed delete warns when active monitors depend on the feed', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      ;(window as any).__watchlistsDeleteCalls = 0

      const now = () => new Date().toISOString()
      const sources = [
        {
          id: 1,
          name: 'Critical Feed',
          url: 'https://example.com/critical.xml',
          source_type: 'rss',
          active: true,
          tags: ['critical'],
          created_at: now(),
          updated_at: now(),
          last_scraped_at: now()
        }
      ]

      const jobs = [
        {
          id: 11,
          name: 'Critical Monitor',
          description: 'Tracks critical updates',
          active: true,
          scope: { sources: [1], groups: [], tags: [] },
          schedule_expr: '0 9 * * *',
          timezone: 'UTC',
          job_filters: { filters: [] },
          created_at: now(),
          updated_at: now(),
          last_run_at: now(),
          next_run_at: now()
        }
      ]

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate(jobs, page, size)
        }

        if (pathname === '/api/v1/watchlists/tags' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/groups' && method === 'GET') {
          return paginate([], page, size)
        }

        const sourceSeenMatch = pathname.match(/^\/api\/v1\/watchlists\/sources\/(\d+)\/seen(?:\?.*)?$/)
        if (sourceSeenMatch && method === 'GET') {
          return {
            source_id: Number(sourceSeenMatch[1]),
            user_id: 1,
            seen_count: 0,
            latest_seen_at: null,
            consec_not_modified: 0,
            defer_until: null,
            recent_keys: []
          }
        }

        const sourceDeleteMatch = pathname.match(/^\/api\/v1\/watchlists\/sources\/(\d+)$/)
        if (sourceDeleteMatch && method === 'DELETE') {
          const sourceId = Number(sourceDeleteMatch[1])
          const index = sources.findIndex((item) => item.id === sourceId)
          if (index >= 0) {
            sources.splice(index, 1)
          }
          ;(window as any).__watchlistsDeleteCalls += 1
          return { ok: true }
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await navigateWatchlistsDestination(page, 'Feeds')
    await expect(page.getByText('Critical Feed')).toBeVisible()

    await selectRowsAndAssertCount(page, 1)
    await page.getByRole('button', { name: /^Delete$/ }).first().click()

    const confirm = page.locator('.ant-modal-confirm').last()
    await expect(confirm).toContainText(/active monitor/i)
    await expect(confirm).toContainText('Critical Monitor')
    await confirm.getByRole('button', { name: 'Cancel' }).click()

    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsDeleteCalls))
      .toBe(0)

    await context.close()
  })

  test('feeds bulk disable shows impact summary before commit', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      ;(window as any).__watchlistsBulkDisableCalls = 0

      const now = () => new Date().toISOString()
      const sources = [
        {
          id: 1,
          name: 'Breaking Feed',
          url: 'https://example.com/breaking.xml',
          source_type: 'rss',
          active: true,
          tags: ['news'],
          created_at: now(),
          updated_at: now(),
          last_scraped_at: now()
        },
        {
          id: 2,
          name: 'Already Paused Feed',
          url: 'https://example.com/paused.xml',
          source_type: 'site',
          active: false,
          tags: ['news'],
          created_at: now(),
          updated_at: now(),
          last_scraped_at: null
        }
      ]

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const body = payload?.body || {}
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        const sourceUpdateMatch = pathname.match(/^\/api\/v1\/watchlists\/sources\/(\d+)$/)
        if (sourceUpdateMatch && method === 'PATCH') {
          const sourceId = Number(sourceUpdateMatch[1])
          const source = sources.find((item) => item.id === sourceId)
          if (!source) return null
          source.active = Boolean(body?.active)
          source.updated_at = now()
          ;(window as any).__watchlistsBulkDisableCalls += 1
          return source
        }

        if (pathname === '/api/v1/watchlists/tags' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/groups' && method === 'GET') {
          return paginate([], page, size)
        }

        const sourceSeenMatch = pathname.match(/^\/api\/v1\/watchlists\/sources\/(\d+)\/seen-stats$/)
        if (sourceSeenMatch && method === 'GET') {
          return {
            keys_total: 0,
            keys_limit: 0,
            seen_recent: 0,
            top_keys: [],
            consec_not_modified: 0,
            defer_until: null,
            old_seen_count: 0,
            old_last_seen: null,
            old_cleanup_cutoff: null
          }
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await navigateWatchlistsDestination(page, 'Feeds')
    await expect(page.getByText('Breaking Feed')).toBeVisible()
    await expect(page.getByText('Already Paused Feed')).toBeVisible()

    await selectRowsAndAssertCount(page, 2)
    await page.getByRole('button', { name: 'Disable' }).click()

    const firstConfirm = page.locator('.ant-modal-confirm').last()
    await expect(firstConfirm).toContainText('2 selected (1 active, 1 inactive). 1 will change state.')
    await firstConfirm.getByRole('button', { name: 'Cancel' }).click()
    await expect(firstConfirm).toBeHidden()

    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsBulkDisableCalls))
      .toBe(0)

    await page.getByRole('button', { name: 'Disable' }).first().click()
    const secondConfirm = page.locator('.ant-modal-confirm').last()
    await secondConfirm.getByRole('button', { name: 'Disable' }).click()

    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsBulkDisableCalls))
      .toBe(2)

    await context.close()
  })

  test('feeds OPML import supports retry failed only recovery flow', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      ;(window as any).__watchlistsImportUploadTexts = []

      const now = () => new Date().toISOString()
      const sources: Array<Record<string, any>> = []

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const decodeUploadText = (uploadPayload: any): string => {
        try {
          const data = uploadPayload?.file?.data
          if (!data) return ''
          let bytes: Uint8Array
          if (data instanceof ArrayBuffer) {
            bytes = new Uint8Array(data)
          } else if (ArrayBuffer.isView(data)) {
            bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
          } else if (Array.isArray(data)) {
            bytes = Uint8Array.from(data)
          } else {
            return ''
          }
          return new TextDecoder().decode(bytes)
        } catch {
          return ''
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        if (pathname === '/api/v1/watchlists/tags' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/groups' && method === 'GET') {
          return paginate([], page, size)
        }

        const sourceSeenMatch = pathname.match(/^\/api\/v1\/watchlists\/sources\/(\d+)\/seen(?:\?.*)?$/)
        if (sourceSeenMatch && method === 'GET') {
          return {
            source_id: Number(sourceSeenMatch[1]),
            user_id: 1,
            seen_count: 0,
            latest_seen_at: null,
            consec_not_modified: 0,
            defer_until: null,
            recent_keys: []
          }
        }

        return null
      }

      const handleUpload = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'POST').toUpperCase()
        if (path !== '/api/v1/watchlists/sources/import' || method !== 'POST') {
          return null
        }

        const opmlText = decodeUploadText(payload)
        ;(window as any).__watchlistsImportUploadTexts.push(opmlText)

        if ((window as any).__watchlistsImportUploadTexts.length === 1) {
          return {
            items: [
              {
                status: 'created',
                name: 'Good Feed',
                url: 'https://good.example.com/rss.xml'
              },
              {
                status: 'error',
                name: 'Failed Feed',
                url: 'https://failed.example.com/rss.xml',
                error: 'timeout while fetching'
              }
            ]
          }
        }

        return {
          items: [
            {
              status: 'created',
              name: 'Failed Feed',
              url: 'https://failed.example.com/rss.xml'
            }
          ]
        }
      }

      ;(window as any).__watchlistsBindBridge(handleRequest, handleUpload)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await navigateWatchlistsDestination(page, 'Feeds')
    await page.getByRole('button', { name: 'Import OPML' }).first().click()
    const importDialog = page.getByRole('dialog', { name: 'Import OPML' })
    await expect(importDialog).toBeVisible()

    const opml = `<?xml version="1.0"?>
<opml version="2.0">
  <body>
    <outline text="Good Feed" xmlUrl="https://good.example.com/rss.xml"/>
    <outline text="Failed Feed" xmlUrl="https://failed.example.com/rss.xml"/>
  </body>
</opml>`

    await importDialog.locator('input[type="file"]').setInputFiles({
      name: 'feeds.opml',
      mimeType: 'text/xml',
      buffer: Buffer.from(opml, 'utf-8')
    })

    await expect(importDialog.getByText('Preflight Summary')).toBeVisible()
    await importDialog.getByRole('button', { name: 'Import 2 feeds' }).click()

    const retryButton = importDialog.getByRole('button', { name: 'Retry failed only' })
    await expect(retryButton).toBeVisible()
    await retryButton.click()

    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsImportUploadTexts.length))
      .toBe(2)

    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsImportUploadTexts[1] || ''))
      .toContain('https://failed.example.com/rss.xml')
    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsImportUploadTexts[1] || ''))
      .not.toContain('https://good.example.com/rss.xml')

    await context.close()
  })

  test('watchlists help links and guided-tour resume are discoverable', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      localStorage.setItem(
        'watchlists:guided-tour:v1',
        JSON.stringify({ status: 'in_progress', step: 2 })
      )

      const now = () => new Date().toISOString()
      const sources = [
        {
          id: 1,
          name: 'Docs Feed',
          url: 'https://example.com/docs.xml',
          source_type: 'rss',
          active: true,
          tags: ['docs'],
          status: 'healthy',
          created_at: now(),
          updated_at: now(),
          last_scraped_at: now()
        }
      ]
      const jobs = [
        {
          id: 11,
          name: 'Docs Monitor',
          description: 'Tracks docs updates',
          active: true,
          scope: { sources: [1], groups: [], tags: ['docs'] },
          schedule_expr: '0 9 * * *',
          timezone: 'UTC',
          job_filters: { filters: [] },
          created_at: now(),
          updated_at: now(),
          last_run_at: now(),
          next_run_at: now()
        }
      ]
      const runs = [
        {
          id: 101,
          job_id: 11,
          status: 'completed',
          started_at: now(),
          finished_at: now(),
          error_msg: null,
          stats: {
            items_found: 1,
            items_ingested: 1,
            items_filtered: 0,
            items_errored: 0
          }
        }
      ]

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate(jobs, page, size)
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          const q = params.get('q')
          const filtered = q ? runs.filter((run) => run.status === q) : runs
          return paginate(filtered, page, size)
        }

        if (pathname === '/api/v1/watchlists/items' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/tags' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/groups' && method === 'GET') {
          return paginate([], page, size)
        }

        const sourceSeenMatch = pathname.match(/^\/api\/v1\/watchlists\/sources\/(\d+)\/seen(?:\?.*)?$/)
        if (sourceSeenMatch && method === 'GET') {
          return {
            source_id: Number(sourceSeenMatch[1]),
            user_id: 1,
            seen_count: 0,
            latest_seen_at: null,
            consec_not_modified: 0,
            defer_until: null,
            recent_keys: []
          }
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const page = await context.newPage()
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await expect(page.getByTestId('watchlists-main-docs-link')).toHaveAttribute(
      'href',
      'https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md'
    )
    await expect(page.getByTestId('watchlists-beta-docs-link')).toHaveAttribute(
      'href',
      'https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md'
    )
    await expect(page.getByTestId('watchlists-beta-report-link')).toHaveAttribute(
      'href',
      'https://github.com/rmusser01/tldw_server/issues/new'
    )
    await expect(page.getByTestId('watchlists-context-docs-link')).toHaveAttribute(
      'href',
      'https://github.com/rmusser01/tldw_server/blob/main/Docs/Product/Watchlists/Watchlist_PRD.md'
    )

    await expect(page.getByTestId('watchlists-resume-guide')).toBeVisible()
    await page.getByTestId('watchlists-resume-guide').click()
    await expect(page.getByText('Watchlists guided tour')).toBeVisible()
    await expect(page.getByText('Step 3 of 5')).toBeVisible()
    await expect(page.getByTestId('watchlists-context-docs-link')).toHaveAttribute(
      'href',
      'https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md#runs'
    )

    await page.getByRole('button', { name: 'Skip' }).click()
    await navigateWatchlistsDestination(page, 'Feeds')
    await expect(page.getByTestId('watchlists-context-docs-link')).toHaveAttribute(
      'href',
      'https://github.com/rmusser01/tldw_server/blob/main/Docs/API/Watchlists_Filters_OPML.md'
    )

    await context.close()
  })

  test('strict demo readiness route renders Activity and Reports without crashing on output errors', async () => {
    test.setTimeout(120_000)
    const extPath = path.resolve('.output/chrome-mv3')
    const { context, page: basePage, optionsUrl } = await launchWithExtensionOrSkip(test, extPath, {
      seedConfig: seededWatchlistsConfig
    })
    await installWatchlistsRuntimeBridge(context)

    await context.addInitScript(() => {
      ;(window as any).__watchlistsStubbed = true
      ;(window as any).__watchlistsOutputCreateCalls = 0

      const now = () => new Date().toISOString()
      const sources = [
        {
          id: 1,
          name: 'Demo Feed',
          url: 'https://example.com/demo.xml',
          source_type: 'rss',
          active: true,
          tags: ['demo'],
          status: 'healthy',
          created_at: now(),
          updated_at: now(),
          last_scraped_at: now()
        }
      ]
      const jobs = [
        {
          id: 11,
          name: 'Demo Briefing',
          description: 'Demo monitor',
          active: true,
          scope: { sources: [1], groups: [], tags: ['demo'] },
          schedule_expr: '0 9 * * *',
          timezone: 'UTC',
          job_filters: { filters: [] },
          output_prefs: {
            template_name: 'briefing_markdown',
            template: { default_name: 'briefing_markdown' },
            generate_audio: true
          },
          created_at: now(),
          updated_at: now(),
          last_run_at: now(),
          next_run_at: now()
        }
      ]
      const runs = [
        {
          id: 101,
          job_id: 11,
          status: 'completed',
          started_at: now(),
          finished_at: now(),
          error_msg: null,
          stats: {
            items_found: 2,
            items_ingested: 2,
            items_filtered: 0,
            items_errored: 0
          }
        }
      ]
      const outputs = [
        {
          id: 201,
          run_id: 101,
          job_id: 11,
          type: 'briefing',
          format: 'md',
          title: 'Demo report with failed audio',
          version: 1,
          expired: false,
          metadata: {
            template_name: 'briefing_markdown',
            audio_briefing_requested: true,
            audio_briefing_status: 'failed',
            audio_briefing_error: 'TTS provider timeout'
          },
          created_at: now(),
          expires_at: null
        }
      ]

      const paginate = (list, page, size) => {
        const current = page || 1
        const limit = size || list.length || 1
        const start = (current - 1) * limit
        const end = start + limit
        return {
          items: list.slice(start, end),
          total: list.length,
          page: current,
          size: limit,
          has_more: end < list.length
        }
      }

      const handleRequest = (payload) => {
        const path = payload?.path || ''
        const method = String(payload?.method || 'GET').toUpperCase()
        const body = payload?.body || null
        const [pathname, queryString] = path.split('?')
        const params = new URLSearchParams(queryString || '')
        const page = Number(params.get('page') || 1)
        const size = Number(params.get('size') || 20)

        if (pathname === '/api/v1/watchlists/sources' && method === 'GET') {
          return paginate(sources, page, size)
        }

        if (pathname === '/api/v1/watchlists/jobs' && method === 'GET') {
          return paginate(jobs, page, size)
        }

        if (pathname === '/api/v1/watchlists/runs' && method === 'GET') {
          const q = params.get('q')
          const filtered = q ? runs.filter((run) => run.status === q) : runs
          return paginate(filtered, page, size)
        }

        const runDetailsMatch = pathname.match(/^\/api\/v1\/watchlists\/runs\/(\d+)\/details$/)
        if (runDetailsMatch && method === 'GET') {
          return {
            ...runs[0],
            filter_tallies: { include: 2 },
            log_text: 'Completed successfully',
            log_path: null,
            truncated: false,
            filtered_sample: null
          }
        }

        if (pathname === '/api/v1/watchlists/items' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/tags' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/groups' && method === 'GET') {
          return paginate([], page, size)
        }

        if (pathname === '/api/v1/watchlists/templates' && method === 'GET') {
          return { items: [{ name: 'briefing_markdown', format: 'md', updated_at: now() }] }
        }

        if (pathname === '/api/v1/watchlists/settings' && method === 'GET') {
          return {
            default_output_ttl_seconds: 86400,
            temporary_output_ttl_seconds: 3600
          }
        }

        if (pathname === '/api/v1/watchlists/outputs' && method === 'GET') {
          return paginate(outputs, page, size)
        }

        if (pathname === '/api/v1/watchlists/outputs' && method === 'POST') {
          ;(window as any).__watchlistsOutputCreateCalls += 1
          if (body?.template_name !== 'briefing_markdown') {
            throw new Error('unexpected_template_name')
          }
          throw new Error('template_not_found: briefing_markdown')
        }

        return null
      }

      ;(window as any).__watchlistsBindBridge(handleRequest)
    })

    const pageErrors: string[] = []
    const page = await context.newPage()
    page.on('pageerror', (error) => {
      pageErrors.push(error.message)
    })
    await page.goto(optionsUrl + '?e2e=1&view=all#/watchlists', { waitUntil: 'domcontentloaded' })
    await page.waitForFunction(() => (window as any).__watchlistsStubbed === true, undefined, {
      timeout: 5_000
    })
    await basePage.close().catch(() => {})

    await expect(page.getByRole('heading', { name: 'Watchlists' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Activity' })).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Reports' })).toBeVisible()

    await page.getByRole('tab', { name: 'Activity' }).click()
    await expect(
      page.locator('.ant-tabs-tabpane-active').getByText('Completed', { exact: true }).first()
    ).toBeVisible()

    await page.getByRole('tab', { name: 'Reports' }).click()
    await expect(
      page.locator('.ant-tabs-tabpane-active').getByText('Demo report with failed audio')
    ).toBeVisible()
    await page.getByRole('button', { name: 'Regenerate' }).first().click()
    await page.getByRole('button', { name: 'Regenerate' }).last().click()

    await expect
      .poll(async () => page.evaluate(() => (window as any).__watchlistsOutputCreateCalls))
      .toBe(1)
    await expect(page.getByTestId('watchlists-outputs-live-region')).toContainText(
      'Failed to regenerate Demo report with failed audio.'
    )
    expect(pageErrors).toEqual([])

    await context.close()
  })
})
