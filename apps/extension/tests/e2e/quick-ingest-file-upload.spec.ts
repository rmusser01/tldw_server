import { test, expect, type Page } from '@playwright/test'
import http from 'node:http'
import { AddressInfo } from 'node:net'
import { launchWithBuiltExtension } from './utils/extension-build'
import { forceConnected, waitForConnectionStore } from './utils/connection'

const ACCEPTED_FILE_FIXTURES = [
  { name: 'sample.pdf', mimeType: 'application/pdf', body: '%PDF-1.4\n' },
  { name: 'sample.txt', mimeType: 'text/plain', body: 'plain text' },
  { name: 'sample.rtf', mimeType: 'application/rtf', body: '{\\rtf1 text}' },
  {
    name: 'sample.docx',
    mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    body: 'PK\u0003\u0004docx'
  },
  { name: 'sample.md', mimeType: 'text/markdown', body: '# Markdown' },
  { name: 'sample.markdown', mimeType: 'text/markdown', body: '# Markdown' },
  { name: 'sample.html', mimeType: 'text/html', body: '<p>HTML</p>' },
  { name: 'sample.htm', mimeType: 'text/html', body: '<p>HTML</p>' },
  {
    name: 'sample.xhtml',
    mimeType: 'application/xhtml+xml',
    body: '<html xmlns="http://www.w3.org/1999/xhtml"><body>XHTML</body></html>'
  },
  { name: 'sample.xml', mimeType: 'application/xml', body: '<root />' },
  { name: 'sample.json', mimeType: 'application/json', body: '{"ok":true}' },
  { name: 'sample.epub', mimeType: 'application/epub+zip', body: 'PK\u0003\u0004epub' },
  { name: 'sample.mp3', mimeType: 'audio/mpeg', body: 'audio' },
  { name: 'sample.wav', mimeType: 'audio/wav', body: 'audio' },
  { name: 'sample.m4a', mimeType: 'audio/mp4', body: 'audio' },
  { name: 'sample.flac', mimeType: 'audio/flac', body: 'audio' },
  { name: 'sample.aac', mimeType: 'audio/aac', body: 'audio' },
  { name: 'sample.ogg', mimeType: 'application/ogg', body: 'ogg' },
  { name: 'sample.mp4', mimeType: 'video/mp4', body: 'video' },
  { name: 'sample.webm', mimeType: 'video/webm', body: 'video' },
  { name: 'sample.mkv', mimeType: '', body: 'video' },
  { name: 'sample.mov', mimeType: 'video/quicktime', body: 'video' },
  { name: 'sample.avi', mimeType: 'video/avi', body: 'video' }
]

test.describe('Quick ingest file upload', () => {
  let server: http.Server
  let baseUrl = ''
  let ingestJobSubmitCount = 0
  let ingestJobSubmitBytes = 0
  const ingestJobSubmitFileNames: string[] = []
  const ingestJobStates = new Map<number, { polls: number; resultId: string }>()

  const readBody = (req: http.IncomingMessage) =>
    new Promise<Buffer>((resolve) => {
      const chunks: Buffer[] = []
      req.on('data', (chunk) => {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
      })
      req.on('end', () => resolve(Buffer.concat(chunks)))
      req.on('error', () => resolve(Buffer.concat(chunks)))
    })

  const readBodyBytes = async (req: http.IncomingMessage) =>
    (await readBody(req)).byteLength

  const patchQuickIngestRuntime = async (page: Page, apiBaseUrl: string) =>
    page.evaluate(async ({ resolvedApiBaseUrl }) => {
      try {
        const runtime =
          (globalThis as any)?.browser?.runtime ||
          (globalThis as any)?.chrome?.runtime
        const onMessage = runtime?.onMessage
        const originalSendMessage =
          typeof runtime?.sendMessage === 'function'
            ? runtime.sendMessage.bind(runtime)
            : null
        if (!runtime || !onMessage || !originalSendMessage) {
          return false
        }

        const originalAddListener =
          typeof onMessage.addListener === 'function'
            ? onMessage.addListener.bind(onMessage)
            : null
        const originalRemoveListener =
          typeof onMessage.removeListener === 'function'
            ? onMessage.removeListener.bind(onMessage)
            : null
        const listeners = new Set<
          (message: any, sender?: any, sendResponse?: any) => void
        >()

        const emit = (message: any) => {
          for (const listener of [...listeners]) {
            try {
              listener(message, {}, () => undefined)
            } catch {
              // best-effort test emitter
            }
          }
        }

        const wait = (ms: number) =>
          new Promise<void>((resolve) => {
            setTimeout(resolve, ms)
          })

        const pollJobUntilCompleted = async (jobId: number) => {
          for (let attempt = 0; attempt < 8; attempt += 1) {
            const response = await fetch(
              `${resolvedApiBaseUrl}/api/v1/media/ingest/jobs/${jobId}`,
              {
                method: 'GET'
              }
            )
            if (!response.ok) {
              throw new Error(`Polling failed (${response.status})`)
            }
            const payload = await response.json().catch(() => ({}))
            const status = String(payload?.status || '').toLowerCase()
            if (
              status === 'completed' ||
              status === 'complete' ||
              status === 'succeeded' ||
              status === 'success'
            ) {
              return payload
            }
            if (status === 'failed' || status === 'error') {
              throw new Error(String(payload?.error || 'Ingest job failed'))
            }
            await wait(150)
          }
          throw new Error('Timed out waiting for ingest job completion')
        }

        onMessage.addListener = (listener: any) => {
          listeners.add(listener)
        }
        onMessage.removeListener = (listener: any) => {
          listeners.delete(listener)
        }

        runtime.sendMessage = async (message: any) => {
          if (message?.type === 'tldw:quick-ingest/start') {
            const payload = message?.payload || {}
            const sessionId = `qi-e2e-upload-${Date.now()}`
            const files = Array.isArray(payload?.files) ? payload.files : []

            queueMicrotask(async () => {
              const results: Array<Record<string, any>> = []
              for (let index = 0; index < files.length; index += 1) {
                const file = files[index] || {}
                const sourceId = String(file?.id || `file-${index + 1}`)
                const fileName = String(file?.name || `file-${index + 1}.bin`)
                const byteLength = Array.isArray(file?.data)
                  ? file.data.length
                  : file?.data instanceof Uint8Array
                    ? file.data.byteLength
                    : file?.data instanceof ArrayBuffer
                      ? file.data.byteLength
                      : 0
                try {
                  const submitResponse = await fetch(
                    `${resolvedApiBaseUrl}/api/v1/media/ingest/jobs`,
                    {
                      method: 'POST',
                      headers: { 'content-type': 'application/json' },
                      body: JSON.stringify({
                        media_type: 'document',
                        source_id: sourceId,
                        file_name: fileName,
                        size_bytes: byteLength
                      })
                    }
                  )
                  if (!submitResponse.ok) {
                    throw new Error(`Submit failed (${submitResponse.status})`)
                  }
                  const submitPayload = await submitResponse
                    .json()
                    .catch(() => ({}))
                  const jobId = Number(submitPayload?.jobs?.[0]?.id)
                  if (!Number.isFinite(jobId)) {
                    throw new Error('Submit response missing job id')
                  }

                  const jobResult = await pollJobUntilCompleted(jobId)
                  const result = {
                    id: sourceId,
                    status: 'ok',
                    fileName,
                    type: 'document',
                    data: {
                      id: jobResult?.result?.id || `media-${sourceId}`
                    }
                  }
                  results.push(result)
                  emit({
                    type: 'tldw:quick-ingest/progress',
                    payload: {
                      sessionId,
                      result
                    }
                  })
                } catch (error: any) {
                  const result = {
                    id: sourceId,
                    status: 'error',
                    fileName,
                    type: 'file',
                    error: error?.message || 'Upload failed'
                  }
                  results.push(result)
                  emit({
                    type: 'tldw:quick-ingest/progress',
                    payload: {
                      sessionId,
                      result
                    }
                  })
                }
              }

              emit({
                type: 'tldw:quick-ingest/completed',
                payload: {
                  sessionId,
                  results
                }
              })
            })

            return { ok: true, sessionId }
          }

          if (message?.type === 'tldw:quick-ingest/cancel') {
            emit({
              type: 'tldw:quick-ingest/cancelled',
              payload: {
                sessionId: String(message?.payload?.sessionId || ''),
                reason: String(message?.payload?.reason || 'Cancelled by user.')
              }
            })
            return { ok: true }
          }

          return originalSendMessage(message)
        }

        ;(window as any).__restoreQuickIngestUploadPatch = () => {
          runtime.sendMessage = originalSendMessage
          if (originalAddListener) {
            onMessage.addListener = originalAddListener
          }
          if (originalRemoveListener) {
            onMessage.removeListener = originalRemoveListener
          }
          listeners.clear()
        }

        return true
      } catch {
        return false
      }
    }, { resolvedApiBaseUrl: apiBaseUrl })

  const openQuickIngestModal = async (page: Page, optionsUrl: string) => {
    await page.goto(optionsUrl + '#/chat', { waitUntil: 'domcontentloaded' })
    await waitForConnectionStore(page, 'quick-ingest-file-upload')
    await forceConnected(page, {}, 'quick-ingest-file-upload')

    const trigger = page
      .getByTestId('open-quick-ingest')
      .or(page.getByRole('button', { name: /open quick ingest|quick ingest|add content/i }))
      .first()
    if (await trigger.isVisible().catch(() => false)) {
      await trigger.click()
    } else {
      await page.evaluate(async () => {
        const hasModal = () =>
          Boolean(document.querySelector('.quick-ingest-modal .ant-modal-content'))
        for (let attempt = 0; attempt < 20 && !hasModal(); attempt += 1) {
          window.dispatchEvent(new CustomEvent('tldw:open-quick-ingest'))
          await new Promise((resolve) => window.setTimeout(resolve, 150))
        }
      })
    }

    const modal = page.getByRole('dialog', { name: /quick ingest/i }).first()
    await expect(modal).toBeVisible({ timeout: 10_000 })
    return modal
  }

  test.beforeAll(async () => {
    server = http.createServer(async (req, res) => {
      const url = req.url || ''
      const method = (req.method || 'GET').toUpperCase()

      const json = (code: number, body: Record<string, any>) => {
        res.writeHead(code, {
          'content-type': 'application/json',
          'access-control-allow-origin': '*',
          'access-control-allow-credentials': 'true'
        })
        res.end(JSON.stringify(body))
      }

      if (method === 'OPTIONS') {
        res.writeHead(204, {
          'access-control-allow-origin': '*',
          'access-control-allow-credentials': 'true',
          'access-control-allow-headers': 'content-type, x-api-key, authorization'
        })
        return res.end()
      }

      if (url === '/api/v1/health' && method === 'GET') {
        return json(200, { status: 'ok' })
      }
      if (url === '/openapi.json' && method === 'GET') {
        return json(200, {
          openapi: '3.1.0',
          info: { title: 'tldw mock', version: 'e2e' },
          paths: {
            '/api/v1/chat/completions': {},
            '/api/v1/rag/search': {},
            '/api/v1/rag/search/stream': {},
            '/api/v1/media/ingest/jobs': {},
            '/api/v1/media/process-videos': {},
            '/api/v1/media/process-audios': {},
            '/api/v1/media/process-pdfs': {},
            '/api/v1/media/process-ebooks': {},
            '/api/v1/media/process-documents': {},
            '/api/v1/media/process-web-scraping': {},
            '/api/v1/reading/save': {},
            '/api/v1/reading/items': {},
            '/api/v1/audio/transcriptions': {},
            '/api/v1/audio/speech': {},
            '/api/v1/llm/models': {},
            '/api/v1/llm/models/metadata': {},
            '/api/v1/llm/providers': {},
            '/api/v1/notes/': {},
            '/api/v1/notes/search/': {},
            '/api/v1/flashcards': {},
            '/api/v1/flashcards/decks': {},
            '/api/v1/characters/world-books': {},
            '/api/v1/chat/dictionaries': {}
          }
        })
      }
      if (url === '/api/v1/rag/health' && method === 'GET') {
        return json(200, {
          status: 'ok',
          components: {
            search_index: {
              status: 'healthy',
              message: '',
              fts_table_count: 1
            }
          }
        })
      }
      if (url === '/api/v1/media/ingest/jobs' && method === 'POST') {
        ingestJobSubmitCount += 1
        const body = await readBody(req)
        ingestJobSubmitBytes += body.byteLength
        try {
          const payload = JSON.parse(body.toString('utf8') || '{}')
          if (typeof payload?.file_name === 'string') {
            ingestJobSubmitFileNames.push(payload.file_name)
          }
        } catch {
          // ignore malformed test payloads
        }
        const jobId = 7000 + ingestJobSubmitCount
        ingestJobStates.set(jobId, {
          polls: 0,
          resultId: `media-${ingestJobSubmitCount}`
        })
        return json(200, {
          batch_id: `batch-${ingestJobSubmitCount}`,
          jobs: [{ id: jobId, status: 'queued' }]
        })
      }
      if (/^\/api\/v1\/media\/ingest\/jobs\/\d+$/.test(url) && method === 'GET') {
        const jobId = Number(url.split('/').pop())
        const tracked = ingestJobStates.get(jobId)
        if (!tracked) return json(404, { error: 'job not found' })
        tracked.polls += 1
        if (tracked.polls > 1) {
          return json(200, {
            id: jobId,
            status: 'completed',
            result: { id: tracked.resultId }
          })
        }
        return json(200, {
          id: jobId,
          status: 'processing'
        })
      }
      if (url.startsWith('/api/v1/media/process-') && method === 'POST') {
        await readBodyBytes(req)
        return json(200, { id: 'processed-1' })
      }
      if (url === '/api/v1/media/process-web-scraping' && method === 'POST') {
        await readBodyBytes(req)
        return json(200, { id: 'scrape-1' })
      }

      res.writeHead(404)
      res.end('not found')
    })

    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
    const addr = server.address() as AddressInfo
    baseUrl = `http://127.0.0.1:${addr.port}`
  })

  test.afterAll(async () => {
    await new Promise<void>((resolve) => server.close(() => resolve()))
  })

  test('uploads a text file and shows success summary', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        serverUrl: baseUrl,
        authMode: 'single-user',
        apiKey: 'test-key'
      }
    })

    try {
      const patched = await patchQuickIngestRuntime(page, baseUrl)
      if (!patched) {
        test.skip(true, 'Unable to patch runtime messaging in extension page context.')
        return
      }

      const modal = await openQuickIngestModal(page, optionsUrl)

      await page.setInputFiles('[data-testid="qi-file-input"]', {
        name: 'sample.txt',
        mimeType: 'text/plain',
        buffer: Buffer.from('hello from playwright')
      })

      await expect(modal.getByText('sample.txt')).toBeVisible({ timeout: 10_000 })

      const runButton = modal.getByRole('button', {
        name: /Run quick ingest|Ingest|Process/i
      }).first()
      await expect(runButton).toBeEnabled()
      await runButton.click()

      await expect(
        modal.getByText(/Quick ingest completed (successfully|with some errors)/i)
      ).toBeVisible({ timeout: 30_000 })

      expect(ingestJobSubmitCount).toBeGreaterThan(0)
      expect(ingestJobSubmitBytes).toBeGreaterThan(0)
    } finally {
      try {
        await page.evaluate(() => {
          try {
            const restore = (window as any).__restoreQuickIngestUploadPatch
            if (typeof restore === 'function') {
              restore()
            }
            delete (window as any).__restoreQuickIngestUploadPatch
          } catch {
            // ignore cleanup failures if page context is gone
          }
        })
      } catch {
        // ignore cleanup failures if page already closed
      }
      await context.close()
    }
  })

  test('accepts every advertised upload extension and excludes legacy doc', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        serverUrl: baseUrl,
        authMode: 'single-user',
        apiKey: 'test-key'
      }
    })

    try {
      const patched = await patchQuickIngestRuntime(page, baseUrl)
      if (!patched) {
        test.skip(true, 'Unable to patch runtime messaging in extension page context.')
        return
      }

      const modal = await openQuickIngestModal(page, optionsUrl)
      const fileInput = page.locator('[data-testid="qi-file-input"]')
      const accept = (await fileInput.getAttribute('accept')) || ''
      const acceptEntries = accept
        .split(',')
        .map((entry) => entry.trim().toLowerCase())
        .filter(Boolean)

      for (const fixture of ACCEPTED_FILE_FIXTURES) {
        const extension = `.${fixture.name.split('.').pop()?.toLowerCase()}`
        expect(acceptEntries).toContain(extension)
      }
      expect(acceptEntries).not.toContain('.doc')
      expect(acceptEntries).not.toContain('application/msword')

      await page.setInputFiles('[data-testid="qi-file-input"]', {
        name: 'legacy.doc',
        mimeType: 'application/msword',
        buffer: Buffer.from('legacy doc')
      })
      await expect(modal.getByText('legacy.doc')).toHaveCount(0)

      const uploadRunId = `accepted-${Date.now()}`
      const uploadFiles = ACCEPTED_FILE_FIXTURES.map((fixture, index) => {
        const extension = fixture.name.slice(fixture.name.lastIndexOf('.'))
        return {
          name: `${uploadRunId}-${index + 1}${extension}`,
          mimeType: fixture.mimeType,
          buffer: Buffer.from(fixture.body)
        }
      })
      const expectedNames = uploadFiles.map((file) => file.name)

      await page.setInputFiles('[data-testid="qi-file-input"]', uploadFiles)

      for (const fileName of expectedNames) {
        await expect(modal.getByText(fileName)).toBeVisible()
      }

      const runButton = modal.getByRole('button', {
        name: /Run quick ingest|Ingest|Process/i
      }).first()
      await expect(runButton).toBeEnabled()
      await runButton.click()

      await expect(
        modal.getByText(/Quick ingest completed (successfully|with some errors)/i)
      ).toBeVisible({ timeout: 30_000 })

      await expect
        .poll(
          () =>
            ingestJobSubmitFileNames.filter((fileName) =>
              expectedNames.includes(fileName)
            ).length,
          { timeout: 10_000 }
        )
        .toBe(expectedNames.length)
    } finally {
      try {
        await page.evaluate(() => {
          try {
            const restore = (window as any).__restoreQuickIngestUploadPatch
            if (typeof restore === 'function') {
              restore()
            }
            delete (window as any).__restoreQuickIngestUploadPatch
          } catch {
            // ignore cleanup failures if page context is gone
          }
        })
      } catch {
        // ignore cleanup failures if page already closed
      }
      await context.close()
    }
  })

  test('exposes dropzone a11y attributes', async () => {
    const { context, page, optionsUrl } = await launchWithBuiltExtension({
      seedConfig: {
        serverUrl: baseUrl,
        authMode: 'single-user',
        apiKey: 'test-key'
      }
    })

    try {
      await openQuickIngestModal(page, optionsUrl)

      const dropzone = page.getByTestId('qi-file-dropzone')
      await expect(dropzone).toHaveAttribute('role', 'button')
      await expect(dropzone).toHaveAttribute('tabindex', '0')
      await expect(dropzone).toHaveAttribute('aria-label', /File upload zone/i)
      await expect(
        dropzone.locator('[aria-live="polite"]')
      ).toHaveCount(1)
    } finally {
      await context.close()
    }
  })
})
