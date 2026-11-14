import {
  type BrowserContext,
  type Page,
  type Worker,
  expect,
  test
} from '@playwright/test'

import { forceConnected, waitForConnectionStore } from './utils/connection'
import { launchWithBuiltExtension } from './utils/extension-build'

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
const SUGGESTION_BASE = `/api/v1/notes/${NOTE_ID}/graph/suggestions`
const UUID =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

type SuggestionRequest = {
  method: string
  path: string
  commandUuid: string | null
}

const installNotesGraphMocks = async (
  context: BrowserContext,
  authorized: boolean
): Promise<Worker> => {
  const worker =
    context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'))
  const installed = await worker.evaluate(
    ({ authorized, constants }) => {
      try {
        let accepted = false
        let semanticConverted = false
        const suggestionRequests: SuggestionRequest[] = []
        const apiRequests: string[] = []
        const semanticGraphUrls: string[] = []
        let graphRequests = 0
        ;(globalThis as any).__notesGraphSuggestionRequests = suggestionRequests
        ;(globalThis as any).__notesGraphApiRequests = apiRequests
        ;(globalThis as any).__notesGraphRequestCount = () => graphRequests
        ;(globalThis as any).__notesSemanticGraphUrls = semanticGraphUrls
        ;(globalThis as any).__notesSemanticManualLinkBody = null

        const note = {
          id: constants.noteId,
          title: 'Extension source note',
          content:
            'Durable synchronization and grounded graph review evidence.',
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
        const semanticEvidence = {
          similarity: 0.8765,
          qualitative_band: 'high',
          source_note_id: `note:${constants.noteId}`,
          target_note_id: 'note:extension-target',
          source_content_version: 2,
          target_content_version: 3,
          generation_id: 'extension-semantic-generation',
          semantic_index_revision: 4,
          configuration_revision: 1,
          normalization_version: 'normalize-v1',
          chunker_version: 'chunk-v1',
          provider_label: 'Extension local embedding provider',
          model_label: 'extension-semantic-model-with-a-long-version-label',
          model_revision: 'model-r1',
          excerpt_pairs: [
            {
              source: {
                field: 'content',
                start_code_point: 0,
                end_code_point:
                  'Extension source passage for semantic evidence.'.length,
                text: 'Extension source passage for semantic evidence.'
              },
              target: {
                field: 'content',
                start_code_point: 0,
                end_code_point:
                  'Extension target passage for semantic evidence.'.length,
                text: 'Extension target passage for semantic evidence.'
              }
            }
          ]
        }
        const semanticGraphStatus = {
          available: true,
          state: 'ready',
          detail_reason: null,
          generation_id: 'extension-semantic-generation',
          semantic_index_revision: 4,
          configuration_revision: 1,
          active_notes: 2,
          indexed_notes: 2,
          dirty_notes: 0,
          excluded_notes: 0,
          failed_notes: 0,
          effective_top_k: 10,
          effective_threshold: 0.75,
          max_top_k: 50,
          max_admission_nodes: 50,
          max_admission_edges: 50,
          max_evidence_pairs: 3,
          max_excerpt_code_points: 480,
          max_edge_evidence_code_points: 2880,
          max_response_evidence_bytes: 262144,
          truncated_by: []
        }
        const semanticCapabilities = {
          active_note_count: 2,
          estimated_chunk_count: 6,
          estimated_run_count: 1,
          provider_label: 'Extension local embedding provider',
          model: 'extension-semantic-model-with-a-long-version-label',
          endpoint_display: 'http://127.0.0.1:8099',
          execution_boundary: 'local',
          storage_boundary: 'local',
          storage_label: 'Extension local vector store',
          outbound_data_categories: ['note_content_chunks', 'note_title'],
          capability_revision: constants.capabilityRevision,
          indexing_available: true,
          unavailable_reason: null,
          metric: 'cosine',
          resolved_dimensions: 384,
          dimension_probe_required: false,
          renewal_requires_delete: false,
          manage_authorized: authorized
        }
        const semanticStatus = {
          state: 'ready',
          detail_reason: null,
          desired_state: 'enabled',
          configuration_revision: 1,
          semantic_index_revision: 4,
          active_generation_id: 'extension-semantic-generation',
          active_generation_usable: true,
          indexed_notes: 2,
          excluded_notes: 0,
          failed_notes: 0,
          pending_notes: 0,
          published_chunks: 6,
          cleanup_pending: false,
          active_run: null
        }
        const graph = (includeSemantic: boolean) => ({
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
            ...(includeSemantic
              ? [
                  {
                    id: 'extension-semantic-edge',
                    source: `note:${constants.noteId}`,
                    target: 'note:extension-target',
                    type: 'semantic',
                    directed: false,
                    weight: 0.8765,
                    label: null,
                    evidence: semanticEvidence
                  }
                ]
              : []),
            ...(semanticConverted
              ? [
                  {
                    id: 'semantic-converted-edge',
                    source: `note:${constants.noteId}`,
                    target: 'note:extension-target',
                    type: 'manual',
                    directed: false,
                    weight: 1,
                    label: null
                  }
                ]
              : []),
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
          suggestions_authorized: authorized,
          manual_link_authorized: authorized,
          ...(includeSemantic ? { semantic_status: semanticGraphStatus } : {})
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
        const staticPrefixResponses: Array<[string, unknown]> = [
          ['/api/v1/notes/keywords', { keywords: [], total: 0 }],
          ['/api/v1/notes/collections', { collections: [], total: 0 }],
          ['/api/v1/notes/moodboards', { moodboards: [], total: 0 }],
          ['/api/v1/notes/trash', { notes: [], total: 0 }]
        ]

        const originalFetch = globalThis.fetch.bind(globalThis)
        const handler = async (
          input: RequestInfo | URL,
          init?: RequestInit
        ): Promise<Response> => {
          const request = input instanceof Request ? input : null
          const url = new URL(request?.url ?? String(input))
          const path = url.pathname
          const method = String(
            init?.method ?? request?.method ?? 'GET'
          ).toUpperCase()
          if (path.startsWith('/api/v1/')) apiRequests.push(`${method} ${path}`)

          if (
            path === '/api/v1/notes/graph/semantic-index/capabilities' &&
            method === 'GET'
          ) {
            return ok(semanticCapabilities)
          }
          if (
            path === '/api/v1/notes/graph/semantic-index' &&
            method === 'GET'
          ) {
            return ok(semanticStatus)
          }
          if (path === '/api/v1/notes/graph' && method === 'GET') {
            graphRequests += 1
            semanticGraphUrls.push(url.toString())
            const includeSemantic =
              url.searchParams
                .get('edge_types')
                ?.split(',')
                .includes('semantic') ?? false
            return ok(graph(includeSemantic))
          }
          if (
            path === `/api/v1/notes/${constants.noteId}/links` &&
            method === 'POST'
          ) {
            semanticConverted = true
            const bodyText =
              typeof init?.body === 'string'
                ? init.body
                : request
                  ? await request.clone().text()
                  : ''
            ;(globalThis as any).__notesSemanticManualLinkBody = bodyText
              ? JSON.parse(bodyText)
              : null
            return ok({
              status: 'created',
              edge: {
                edge_id: 'semantic-converted-edge',
                from_note_id: constants.noteId,
                to_note_id: 'extension-target'
              }
            })
          }
          if (path.includes('/graph/suggestions')) {
            const headers = new Headers(init?.headers ?? request?.headers)
            suggestionRequests.push({
              method,
              path,
              commandUuid:
                headers.get('Idempotency-Key') ||
                headers.get('idempotency-key') ||
                null
            })
            if (path.endsWith('/capabilities') && method === 'GET') {
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
            if (
              path.endsWith('/extension-related-suggestion/accept') &&
              method === 'POST'
            ) {
              accepted = true
              return ok({
                resource_id: 'extension-related-suggestion',
                state: 'accepted',
                revision: 2,
                cleared_count: null
              })
            }
            throw new Error(`Unhandled suggestion request: ${method} ${path}`)
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
          const staticResponse = staticPrefixResponses.find(([prefix]) =>
            path.startsWith(prefix)
          )
          if (staticResponse) return ok(staticResponse[1])
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
            return new Response(
              JSON.stringify({ detail: 'Not found in fixture' }),
              {
                status: 404,
                headers: { 'content-type': 'application/json' }
              }
            )
          }
          return originalFetch(input, init)
        }
        globalThis.fetch = handler as typeof fetch
        ;(globalThis as any).__notesGraphMockInstalled =
          globalThis.fetch === handler
        return globalThis.fetch === handler
      } catch (error) {
        throw new Error('Notes graph service-worker fixture setup failed', {
          cause: error
        })
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

const dismissTourIfVisible = async (page: Page) => {
  const skipTour = page.getByText('Skip tour', { exact: true })
  if ((await skipTour.count()) > 0 && (await skipTour.isVisible())) {
    await skipTour.click()
  }
}

const closeMobileNotesList = async (page: Page) => {
  const backdrop = page.getByTestId('notes-mobile-sidebar-backdrop')
  if ((await backdrop.count()) > 0 && (await backdrop.isVisible())) {
    await backdrop.click()
  }
  const list = page.getByTestId('notes-list-region')
  await expect(list).toHaveClass(/-translate-x-full/)
  await expect
    .poll(async () => {
      const bounds = await list.boundingBox()
      return bounds ? bounds.x + bounds.width : 0
    })
    .toBeLessThanOrEqual(1)
  await expect(backdrop).toHaveCount(0)
}

const assertExactSuggestionRequests = async (
  worker: Worker,
  expected: Record<string, number>
) => {
  const calls = await worker.evaluate<SuggestionRequest[]>(
    () => (globalThis as any).__notesGraphSuggestionRequests ?? []
  )
  const multiset = Object.fromEntries(
    calls.reduce<Array<[string, number]>>((entries, call) => {
      const key = `${call.method} ${call.path}`
      const existing = entries.find(([candidate]) => candidate === key)
      if (existing) existing[1] += 1
      else entries.push([key, 1])
      return entries
    }, [])
  )
  expect(multiset).toEqual(expected)
  const commands = calls.filter((call) => call.method !== 'GET')
  expect(
    commands.every(
      (call) => Boolean(call.commandUuid) && UUID.test(call.commandUuid ?? '')
    )
  ).toBe(true)
  expect(new Set(commands.map((call) => call.commandUuid)).size).toBe(
    commands.length
  )
}

test.describe('Notes workspace UX', () => {
  test('shows offline empty state and disables editor when not connected', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension()

    await page.goto(optionsUrl + '#/notes')
    await page.waitForLoadState('networkidle')

    const headline = page.getByText(
      /Connect to use Notes|Explore Notes in demo mode/i
    )
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
    await forceConnected(
      page,
      { serverUrl: 'http://dummy-tldw' },
      'notes-connected'
    )

    await page.goto(optionsUrl + '#/notes')
    await page.waitForLoadState('networkidle')

    const textarea = page.getByPlaceholder('Write your note here...')
    await textarea.fill('Unsaved note content')
    await expect(page.getByText(/Unsaved changes/i)).toBeVisible()

    const newNoteButton = page.getByRole('button', { name: /New note/i })
    await expect(newNoteButton).toBeEnabled()
    await newNoteButton.click()

    const discardDialog = page.getByRole('dialog', {
      name: /Discard changes\?/i
    })
    await expect(discardDialog).toBeVisible()

    const cancelButton = discardDialog.getByRole('button', { name: /Cancel/i })
    await cancelButton.click()
    await expect(discardDialog).toBeHidden()
    await expect(textarea).toHaveValue('Unsaved note content')

    await newNoteButton.click()
    const discardDialogAgain = page.getByRole('dialog', {
      name: /Discard changes\?/i
    })
    await expect(discardDialogAgain).toBeVisible()
    const discardButton = discardDialogAgain.getByRole('button', {
      name: /Discard/i
    })
    await discardButton.click()
    await expect(textarea).toHaveValue('')

    await context.close()
  })

  test('uses the shared grounded graph review workflow at desktop and 320px', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension(
      connectedLaunchOptions
    )
    await page.goto(optionsUrl + '#/settings/tldw', {
      waitUntil: 'domcontentloaded'
    })
    await page.reload({ waitUntil: 'domcontentloaded' })
    const worker = await installNotesGraphMocks(context, true)
    await page.evaluate(() => {
      window.location.hash = '/notes'
    })
    await expect(page).toHaveURL(optionsUrl + '#/notes')
    await dismissTourIfVisible(page)
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
    await page
      .getByRole('button', { name: 'Open note Extension source note' })
      .click()

    await expect(page.getByTestId('notes-view-mode-graph')).toBeVisible()
    await page.getByTestId('notes-view-mode-graph').click()
    await expect(page.getByTestId('notes-graph-canvas')).toBeVisible()
    await expect(page.getByRole('button', { name: 'All notes' })).toBeDisabled()
    await expect(
      page.getByTestId('notes-graph-all-disabled-reason')
    ).toContainText('up to 8 active notes')

    await page.getByRole('tab', { name: 'Suggestions' }).click()
    await expect(page.getByText(PROVIDER, { exact: true })).toBeVisible()
    await expect(page.getByText(MODEL, { exact: true })).toBeVisible()
    await expect(page.getByText('External', { exact: true })).toBeVisible()
    await expect(
      page.getByText('Candidate note excerpts', { exact: true })
    ).toBeVisible()
    await expect(page.getByText(TARGET_TITLE, { exact: true })).toBeVisible()

    const pixels = await page
      .getByTestId('notes-graph-canvas')
      .evaluate((root) => {
        let painted = 0
        for (const canvas of Array.from(root.querySelectorAll('canvas'))) {
          const context = canvas.getContext('2d')
          if (!context) continue
          const data = context.getImageData(
            0,
            0,
            canvas.width,
            canvas.height
          ).data
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
      page.locator(
        '[data-suggestion-review-row="extension-related-suggestion"]'
      )
    ).toHaveCount(0)
    await expect
      .poll(() =>
        worker.evaluate(
          () => (globalThis as any).__notesGraphRequestCount?.() ?? 0
        )
      )
      .toBeGreaterThan(graphCallsBeforeAccept)
    await page
      .getByRole('button', { name: 'Relationships', exact: true })
      .click()
    await expect(
      page.getByTestId('notes-graph-relationships-view').getByText(TARGET_TITLE)
    ).toBeVisible()

    await page.setViewportSize({ width: 320, height: 900 })
    await closeMobileNotesList(page)
    const geometry = await page
      .getByTestId('notes-graph-workspace')
      .evaluate((root) => {
        const rect = root.getBoundingClientRect()
        return {
          left: rect.left,
          right: rect.right,
          viewport: innerWidth,
          overflow: document.documentElement.scrollWidth - innerWidth,
          visibleOverlays: Array.from(
            document.querySelectorAll(
              '[role="dialog"], [data-testid="notes-mobile-sidebar-backdrop"], .ant-drawer-mask'
            )
          ).filter((element) => {
            const bounds = element.getBoundingClientRect()
            const style = getComputedStyle(element)
            return (
              bounds.width > 0 &&
              bounds.height > 0 &&
              style.visibility !== 'hidden' &&
              style.display !== 'none' &&
              Number(style.opacity || '1') > 0
            )
          }).length
        }
      })
    expect(geometry.left).toBeGreaterThanOrEqual(-1)
    expect(geometry.right).toBeLessThanOrEqual(geometry.viewport + 1)
    expect(geometry.overflow).toBeLessThanOrEqual(1)
    expect(geometry.visibleOverlays).toBe(0)
    await assertExactSuggestionRequests(worker, {
      [`GET ${SUGGESTION_BASE}`]: 1,
      [`GET ${SUGGESTION_BASE}/capabilities`]: 1,
      [`GET ${SUGGESTION_BASE}/runs`]: 1,
      [`POST ${SUGGESTION_BASE}/extension-related-suggestion/accept`]: 1
    })
    await context.close()
  })

  test('uses semantic graph evidence and canonical conversion in Chromium at desktop and 320px', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension(
      connectedLaunchOptions
    )
    await page.goto(optionsUrl + '#/settings/tldw', {
      waitUntil: 'domcontentloaded'
    })
    await page.reload({ waitUntil: 'domcontentloaded' })
    const worker = await installNotesGraphMocks(context, true)
    await page.evaluate(() => {
      window.location.hash = '/notes'
    })
    await expect(page).toHaveURL(optionsUrl + '#/notes')
    await dismissTourIfVisible(page)
    await waitForConnectionStore(page, 'notes-semantic-connected')
    await forceConnected(
      page,
      { serverUrl: 'http://dummy-tldw' },
      'notes-semantic-connected'
    )
    await page
      .getByRole('button', { name: 'Open note Extension source note' })
      .click()
    await page.getByTestId('notes-view-mode-graph').click()

    const similar = page.getByRole('checkbox', {
      name: 'Similar content',
      exact: true
    })
    await expect(similar).toBeVisible()
    await expect(similar).not.toBeChecked()
    await similar.check()
    await expect
      .poll(async () => {
        const urls = await worker.evaluate<string[]>(
          () => (globalThis as any).__notesSemanticGraphUrls ?? []
        )
        return urls.at(-1) ?? ''
      })
      .toContain('semantic_top_k=10')

    await page
      .getByRole('button', { name: 'Relationships', exact: true })
      .click()
    const relationships = page.getByTestId('notes-graph-relationships-view')
    const evidenceDisclosure = relationships.getByTestId(
      'notes-graph-semantic-evidence-toggle'
    )
    await expect(evidenceDisclosure).toContainText('Passage similarity: 0.8765')
    await expect(
      relationships.getByText('Extension source passage for semantic evidence.')
    ).not.toBeVisible()
    await evidenceDisclosure.click()
    await expect(
      relationships.getByText('Extension source passage for semantic evidence.')
    ).toBeVisible()
    await evidenceDisclosure.click()
    await relationships
      .getByRole('button', { name: 'Similar content', exact: true })
      .click()
    const inspector = page.getByTestId('notes-graph-inspector-region')
    await expect(
      inspector.getByText('Extension target passage for semantic evidence.')
    ).toBeVisible()
    await inspector
      .getByRole('button', { name: 'Create manual link', exact: true })
      .click()

    await expect
      .poll(() =>
        worker.evaluate(
          () => (globalThis as any).__notesSemanticManualLinkBody ?? null
        )
      )
      .toEqual({
        to_note_id: 'extension-target',
        directed: false,
        weight: 1,
        idempotency_key: expect.any(String),
        semantic_conversion: {
          generation_id: 'extension-semantic-generation'
        }
      })
    await expect(
      page.getByRole('button', { name: 'Create manual link', exact: true })
    ).toHaveCount(0)

    await page.setViewportSize({ width: 320, height: 900 })
    await closeMobileNotesList(page)
    const geometry = await page
      .getByTestId('notes-graph-workspace')
      .evaluate((root) => {
        const rect = root.getBoundingClientRect()
        return {
          left: rect.left,
          right: rect.right,
          viewport: innerWidth,
          overflow: document.documentElement.scrollWidth - innerWidth,
          nestedCards: root.querySelectorAll(
            '[data-ui="card"] [data-ui="card"]'
          ).length
        }
      })
    expect(geometry.left).toBeGreaterThanOrEqual(-1)
    expect(geometry.right).toBeLessThanOrEqual(geometry.viewport + 1)
    expect(geometry.overflow).toBeLessThanOrEqual(1)
    expect(geometry.nestedCards).toBe(0)
    const apiRequests = await worker.evaluate<string[]>(
      () => (globalThis as any).__notesGraphApiRequests ?? []
    )
    expect(
      apiRequests.some((request) => request.includes('/api/v1/jobs'))
    ).toBe(false)
    await context.close()
  })

  test('makes no nested suggestion requests for a read-only graph', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension(
      connectedLaunchOptions
    )
    await page.goto(optionsUrl + '#/settings/tldw', {
      waitUntil: 'domcontentloaded'
    })
    await page.reload({ waitUntil: 'domcontentloaded' })
    const worker = await installNotesGraphMocks(context, false)
    await page.evaluate(() => {
      window.location.hash = '/notes'
    })
    await expect(page).toHaveURL(optionsUrl + '#/notes')
    await dismissTourIfVisible(page)
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
    await page
      .getByRole('button', { name: 'Open note Extension source note' })
      .click()
    await page.getByTestId('notes-view-mode-graph').click()
    await expect(page.getByTestId('notes-graph-canvas')).toBeVisible()
    await expect(page.getByRole('tab', { name: 'Suggestions' })).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Generate' })).toHaveCount(0)
    await page.waitForTimeout(500)
    await assertExactSuggestionRequests(worker, {})
    await context.close()
  })
})
