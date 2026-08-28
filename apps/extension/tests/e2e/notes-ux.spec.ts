import { test, expect, type BrowserContext, type Worker } from '@playwright/test'
import { launchWithBuiltExtension } from './utils/extension-build'
import {
  waitForConnectionStore,
  forceConnected
} from './utils/connection'

const NOTE_ID = 'extension-source-note'
const NOW = '2026-08-28T12:00:00Z'
const SOURCE_FINGERPRINT = `sha256:${'a'.repeat(64)}`
const TARGET_FINGERPRINT = `sha256:${'b'.repeat(64)}`
const CAPABILITY_REVISION = `sha256:${'c'.repeat(64)}`
const ENDPOINT_REVISION = `sha256:${'d'.repeat(64)}`
const PROVIDER = 'Deterministic Extension Provider With A Long Disclosure Name'
const MODEL = 'extension-grounding-model-with-a-long-version-label'
const TARGET_TITLE =
  'Grounded extension parity target with a deliberately long title that wraps cleanly'

const installNotesGraphMocks = async (
  context: BrowserContext,
  authorized: boolean
): Promise<Worker> => {
  const worker =
    context.serviceWorkers()[0] ??
    (await context.waitForEvent('serviceworker'))
  const installed = await worker.evaluate(
    ({ authorized, constants }) => {
      try {
        let accepted = false
        const suggestionRequests: Array<{
          method: string
          path: string
          commandUuid: string | null
        }> = []
        const apiRequests: string[] = []
        let graphRequests = 0
        ;(globalThis as any).__notesGraphSuggestionRequests = suggestionRequests
        ;(globalThis as any).__notesGraphApiRequests = apiRequests
        ;(globalThis as any).__notesGraphRequestCount = () => graphRequests

        const note = {
          id: constants.noteId,
          title: 'Extension source note',
          content: 'Durable synchronization and grounded graph review evidence.',
          version: 2,
          keywords: [],
          created_at: constants.now,
          updated_at: constants.now
        }
        const node = (
          id: string,
          label: string,
          type: 'note' | 'tag' | 'source' = 'note'
        ) => ({
          id,
          type,
          label,
          created_at: type === 'note' ? constants.now : null,
          deleted: false,
          degree: type === 'note' ? 1 : null,
          tag_count: type === 'note' ? 0 : null,
          primary_source_id: null
        })
        const graph = () => ({
          nodes: [
            node(`note:${constants.noteId}`, note.title),
            node('note:extension-target', constants.targetTitle),
            node('source:extension-web', 'Extension web source', 'source')
          ],
          edges: [
            {
              id: 'source-edge',
              source: `note:${constants.noteId}`,
              target: 'source:extension-web',
              type: 'source_membership',
              directed: false,
              weight: 1,
              label: null
            },
            ...(accepted
              ? [
                  {
                    id: 'accepted-edge',
                    source: `note:${constants.noteId}`,
                    target: 'note:extension-target',
                    type: 'manual',
                    directed: false,
                    weight: 1,
                    label: null
                  }
                ]
              : [])
          ],
          truncated: false,
          truncated_by: [],
          has_more: false,
          cursor: null,
          limits: { max_nodes: 120, max_edges: 480, max_degree: 300 },
          radius_cap_applied: false,
          active_note_count: 9,
          all_notes_note_cap: 8,
          all_notes_eligible: false,
          suggestions_authorized: authorized
        })
        const capability = {
          provider: constants.provider,
          model: constants.model,
          endpoint_origin_revision: constants.endpointRevision,
          data_boundary: 'remote',
          disclosure_external: true,
          outbound_data_categories: [
            'selected_note_title',
            'selected_note_excerpts',
            'candidate_note_titles',
            'candidate_note_excerpts',
            'existing_tag_labels'
          ],
          generation_available: true,
          unavailable_reason: null,
          limits: {
            max_candidates: 30,
            max_relationships: 5,
            max_tags: 5,
            max_new_tags: 2,
            max_tag_catalog: 100,
            max_estimated_input_tokens: 24000,
            max_output_tokens: 2000,
            provider_timeout_seconds: 120,
            response_candidates: 1
          },
          allowed_actions: [
            'generate',
            'cancel',
            'accept',
            'reject',
            'reset_rejections'
          ],
          revision: constants.capabilityRevision
        }
        const suggestion = {
          id: 'extension-related-suggestion',
          run_id: 'extension-run',
          kind: 'related_note',
          state: 'pending',
          revision: 1,
          source_note_id: constants.noteId,
          source_fingerprint: constants.sourceFingerprint,
          target_note_id: 'extension-target',
          target_fingerprint: constants.targetFingerprint,
          target_title: constants.targetTitle,
          normalized_tag: null,
          display_tag: null,
          existing_tag: false,
          match_strength: 'strong',
          rationale:
            'Both notes ground explicit review and durable synchronization in inspectable evidence.',
          evidence: [
            {
              side: 'source',
              note_id: constants.noteId,
              field: 'content',
              start_offset: 0,
              end_offset: 38,
              text: 'Durable synchronization and grounded graph review evidence.'
            },
            {
              side: 'target',
              note_id: 'extension-target',
              field: 'content',
              start_offset: 0,
              end_offset: 35,
              text: 'Inspect evidence before accepting a graph relationship.'
            }
          ],
          updated_at: constants.now
        }
        const ok = (data: unknown, headers?: Record<string, string>) =>
          new Response(JSON.stringify(data), {
            status: 200,
            headers: {
              'content-type': 'application/json',
              ...headers
            }
          })

        const originalFetch = globalThis.fetch.bind(globalThis)
        const handler = async (
          input: RequestInfo | URL,
          init?: RequestInit
        ): Promise<Response> => {
          const request = input instanceof Request ? input : null
          const url = new URL(request?.url ?? String(input))
          const path = url.pathname
          const method = String(init?.method ?? request?.method ?? 'GET').toUpperCase()
          if (path.startsWith('/api/v1/')) apiRequests.push(`${method} ${path}`)

          if (path === '/api/v1/notes/graph' && method === 'GET') {
            graphRequests += 1
            return ok(graph())
          }
          if (path.includes('/graph/suggestions')) {
            const headers = new Headers(init?.headers ?? request?.headers)
            suggestionRequests.push({
              method,
              path,
              commandUuid:
                headers.get('Idempotency-Key') || headers.get('idempotency-key') || null
            })
            if (path.endsWith('/capabilities')) {
              return ok(capability, {
                etag: `"${constants.capabilityRevision}"`
              })
            }
            if (path.endsWith('/runs') && method === 'GET') {
              return ok({ items: [], next_cursor: null })
            }
            if (path.endsWith('/suggestions') && method === 'GET') {
              return ok({
                items: accepted ? [] : [suggestion],
                next_cursor: null,
                current_source_fingerprint: constants.sourceFingerprint,
                rejection_set_revision: 0,
                rejection_count: 0
              })
            }
            if (path.endsWith('/extension-related-suggestion/accept')) {
              accepted = true
              return ok({
                resource_id: 'extension-related-suggestion',
                state: 'accepted',
                revision: 2,
                cleared_count: null
              })
            }
          }
          if (
            path.startsWith('/api/v1/notes/title-settings') ||
            path.startsWith('/api/v1/admin/notes/title-settings')
          ) {
            return ok({ llm_enabled: false, default_strategy: 'heuristic' })
          }
          if (path.startsWith('/api/v1/notes/search')) {
            return ok({ notes: [note], total: 1 })
          }
          if (path === '/api/v1/notes' || path === '/api/v1/notes/') {
            return ok({
              items: [note],
              pagination: { total_items: 1 }
            })
          }
          if (path === `/api/v1/notes/${constants.noteId}`) {
            return ok({ ...note, links: [] })
          }
          if (path.includes('/neighbors')) return ok({ nodes: [], edges: [] })
          if (path.startsWith('/api/v1/notes/keywords')) {
            return ok({ keywords: [], total: 0 })
          }
          if (path.startsWith('/api/v1/notes/collections')) {
            return ok({ collections: [], total: 0 })
          }
          if (path.startsWith('/api/v1/notes/moodboards')) {
            return ok({ moodboards: [], total: 0 })
          }
          if (path.startsWith('/api/v1/notes/trash')) {
            return ok({ notes: [], total: 0 })
          }
          if (path === '/api/v1/health' || path === '/api/v1/health/live') {
            return ok({ status: 'ok' })
          }
          if (path === '/api/v1/auth/me') {
            return ok({
              id: 1,
              username: 'extension-e2e',
              role: 'user',
              is_active: true
            })
          }
          if (path.startsWith('/api/v1/')) {
            return new Response(JSON.stringify({ detail: 'Not found in fixture' }), {
              status: 404,
              headers: { 'content-type': 'application/json' }
            })
          }
          return originalFetch(input, init)
        }
        globalThis.fetch = handler as typeof fetch
        ;(globalThis as any).__notesGraphMockInstalled =
          globalThis.fetch === handler
        return globalThis.fetch === handler
      } catch {
        return false
      }
    },
    {
      authorized,
      constants: {
        noteId: NOTE_ID,
        now: NOW,
        sourceFingerprint: SOURCE_FINGERPRINT,
        targetFingerprint: TARGET_FINGERPRINT,
        capabilityRevision: CAPABILITY_REVISION,
        endpointRevision: ENDPOINT_REVISION,
        provider: PROVIDER,
        model: MODEL,
        targetTitle: TARGET_TITLE
      }
    }
  )
  if (!installed) throw new Error('Notes graph runtime mock did not install')
  return worker
}

const connectedLaunchOptions = {
  allowOffline: true,
  seedConfig: {
    __tldw_first_run_complete: true,
    tldwConfig: {
      serverUrl: 'http://dummy-tldw',
      authMode: 'single-user',
      apiKey: 'test-key'
    }
  }
} as const

test.describe('Notes workspace UX', () => {
  test('shows offline empty state and disables editor when not connected', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension()

    await page.goto(optionsUrl + '#/notes')
    await page.waitForLoadState('networkidle')

    const headline = page.getByText(/Connect to use Notes|Explore Notes in demo mode/i)
    await expect(headline).toBeVisible()

    const editorPanel = page.locator('div[aria-disabled="true"]').last()
    await expect(editorPanel).toBeVisible()

    const textarea = page.getByPlaceholder('Write your note here...')
    await expect(textarea).toHaveAttribute('readonly', '')

    await expect(
      page.getByRole('button', { name: /Copy note content/i })
    ).toHaveCount(1)
    await expect(
      page.getByRole('button', { name: /Export note as Markdown/i })
    ).toHaveCount(1)
    await expect(
      page.getByRole('button', { name: /Delete note/i })
    ).toHaveCount(1)

    const settingsCta = page.getByRole('button', {
      name: /Set up server|Open tldw server settings/i
    })
    await settingsCta.click()
    await expect(page).toHaveURL(/#\/settings\/tldw/i)
    await expect(
      page.getByRole('heading', { name: /tldw Server Configuration/i })
    ).toBeVisible()

    await context.close()
  })

  test('asks before discarding unsaved editor changes', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension()

    await page.goto(optionsUrl, { waitUntil: 'networkidle' })
    await waitForConnectionStore(page, 'notes-connected')
    await forceConnected(page, { serverUrl: 'http://dummy-tldw' }, 'notes-connected')

    await page.goto(optionsUrl + '#/notes')
    await page.waitForLoadState('networkidle')

    const textarea = page.getByPlaceholder('Write your note here...')
    await textarea.fill('Unsaved note content')
    await expect(page.getByText(/Unsaved changes/i)).toBeVisible()

    const newNoteButton = page.getByRole('button', { name: /New note/i })
    await expect(newNoteButton).toBeEnabled()
    await newNoteButton.click()

    const discardDialog = page.getByRole('dialog', { name: /Discard changes\?/i })
    await expect(discardDialog).toBeVisible()

    const cancelButton = discardDialog.getByRole('button', { name: /Cancel/i })
    await cancelButton.click()
    await expect(discardDialog).toBeHidden()
    await expect(textarea).toHaveValue('Unsaved note content')

    await newNoteButton.click()
    const discardDialogAgain = page.getByRole('dialog', { name: /Discard changes\?/i })
    await expect(discardDialogAgain).toBeVisible()
    const discardButton = discardDialogAgain.getByRole('button', { name: /Discard/i })
    await discardButton.click()
    await expect(textarea).toHaveValue('')

    await context.close()
  })

  test('uses the shared grounded graph review workflow at desktop and 320px', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension(
      connectedLaunchOptions
    )
    await page.goto(optionsUrl + '#/settings/tldw', { waitUntil: 'domcontentloaded' })
    await page.reload({ waitUntil: 'domcontentloaded' })
    const worker = await installNotesGraphMocks(context, true)
    await page.evaluate(() => {
      window.location.hash = '/notes'
    })
    await expect(page).toHaveURL(optionsUrl + '#/notes')
    await page
      .getByText('Skip tour', { exact: true })
      .click({ timeout: 5_000 })
      .catch(() => {})
    await waitForConnectionStore(page, 'notes-graph-connected')
    await forceConnected(
      page,
      { serverUrl: 'http://dummy-tldw' },
      'notes-graph-connected'
    )
    await expect
      .poll(() =>
        worker.evaluate(() => (globalThis as any).__notesGraphApiRequests ?? [])
      )
      .toContain('GET /api/v1/notes/')
    await page.getByRole('button', { name: 'Open note Extension source note' }).click()

    await expect(page.getByTestId('notes-view-mode-graph')).toBeVisible()
    await page.getByTestId('notes-view-mode-graph').click()
    await expect(page.getByTestId('notes-graph-canvas')).toBeVisible()
    await expect(page.getByRole('button', { name: 'All notes' })).toBeDisabled()
    await expect(page.getByTestId('notes-graph-all-disabled-reason')).toContainText(
      'up to 8 active notes'
    )

    await page.getByRole('tab', { name: 'Suggestions' }).click()
    await expect(page.getByText(PROVIDER, { exact: true })).toBeVisible()
    await expect(page.getByText(MODEL, { exact: true })).toBeVisible()
    await expect(page.getByText('External', { exact: true })).toBeVisible()
    await expect(page.getByText('Candidate note excerpts', { exact: true })).toBeVisible()
    await expect(page.getByText(TARGET_TITLE, { exact: true })).toBeVisible()

    const pixels = await page.getByTestId('notes-graph-canvas').evaluate((root) => {
      let painted = 0
      for (const canvas of Array.from(root.querySelectorAll('canvas'))) {
        const context = canvas.getContext('2d')
        if (!context) continue
        const data = context.getImageData(0, 0, canvas.width, canvas.height).data
        for (let index = 3; index < data.length; index += 32) {
          if (data[index] > 0) painted += 1
        }
      }
      return painted
    })
    expect(pixels).toBeGreaterThan(20)

    const graphCallsBeforeAccept = await worker.evaluate(
      () => (globalThis as any).__notesGraphRequestCount?.() ?? 0
    )
    await page.getByRole('button', { name: `Accept ${TARGET_TITLE}` }).click()
    await expect(
      page.locator('[data-suggestion-review-row="extension-related-suggestion"]')
    ).toHaveCount(0)
    await expect
      .poll(() =>
        worker.evaluate(() => (globalThis as any).__notesGraphRequestCount?.() ?? 0)
      )
      .toBeGreaterThan(graphCallsBeforeAccept)
    await page.getByRole('button', { name: 'Relationships', exact: true }).click()
    await expect(
      page.getByTestId('notes-graph-relationships-view').getByText(TARGET_TITLE)
    ).toBeVisible()

    await page.setViewportSize({ width: 320, height: 900 })
    const geometry = await page.getByTestId('notes-graph-workspace').evaluate((root) => {
      const rect = root.getBoundingClientRect()
      return {
        left: rect.left,
        right: rect.right,
        viewport: innerWidth,
        overflow: document.documentElement.scrollWidth - innerWidth
      }
    })
    expect(geometry.left).toBeGreaterThanOrEqual(-1)
    expect(geometry.right).toBeLessThanOrEqual(geometry.viewport + 1)
    expect(geometry.overflow).toBeLessThanOrEqual(1)
    await context.close()
  })

  test('makes no nested suggestion requests for a read-only graph', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension(
      connectedLaunchOptions
    )
    await page.goto(optionsUrl + '#/settings/tldw', { waitUntil: 'domcontentloaded' })
    await page.reload({ waitUntil: 'domcontentloaded' })
    const worker = await installNotesGraphMocks(context, false)
    await page.evaluate(() => {
      window.location.hash = '/notes'
    })
    await expect(page).toHaveURL(optionsUrl + '#/notes')
    await page
      .getByText('Skip tour', { exact: true })
      .click({ timeout: 5_000 })
      .catch(() => {})
    await waitForConnectionStore(page, 'notes-graph-read-only-connected')
    await forceConnected(
      page,
      { serverUrl: 'http://dummy-tldw' },
      'notes-graph-read-only-connected'
    )
    await expect
      .poll(() =>
        worker.evaluate(() => (globalThis as any).__notesGraphApiRequests ?? [])
      )
      .toContain('GET /api/v1/notes/')
    await page.getByRole('button', { name: 'Open note Extension source note' }).click()
    await page.getByTestId('notes-view-mode-graph').click()
    await expect(page.getByTestId('notes-graph-canvas')).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Suggestions' })).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Generate' })).toHaveCount(0)
    await page.waitForTimeout(500)
    const suggestionRequests = await worker.evaluate(
      () => (globalThis as any).__notesGraphSuggestionRequests ?? []
    )
    expect(suggestionRequests).toEqual([])
    await context.close()
  })
})
