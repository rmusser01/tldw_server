import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// W4 Stage 1 (WP-36/37/38) quick-ingest HTTP batching pins. These follow the
// W2 background harness pattern from background.quick-ingest-authority.test.ts:
// hoisted fake storage/browser, real background module, global fetch stub.

const harness = vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (value: unknown) => value
  })
  return {
    saved: {} as Record<string, unknown>,
    changes: new Set<(changes: Record<string, { oldValue?: unknown; newValue?: unknown }>, area: string) => void>(),
    values: new Map<string, unknown>(),
    sessionValues: new Map<string, unknown>(),
    listeners: new Set<
      (
        message: unknown,
        sender: unknown,
        reply: (value: unknown) => void
      ) => unknown
    >(),
    sent: [] as any[]
  }
})
vi.mock("@/utils/safe-storage", async importOriginal => {
  const { safeStorageSerde } = await importOriginal<typeof import("@/utils/safe-storage")>()
  return {
  safeStorageSerde,
  createSafeStorage: ({ area = "local" } = {}) => {
    const values = area === "session" ? harness.sessionValues : harness.values
    return {
      get: async (key: string) => values.get(key),
      set: async (key: string, value: unknown) => {
        const oldValue = values.get(key)
        values.set(key, value)
        for (const listener of harness.changes) listener({ [key]: {
          oldValue: safeStorageSerde.serializer(oldValue),
          newValue: safeStorageSerde.serializer(value)
        } }, area)
      },
      remove: async (key: string) => {
        const oldValue = values.get(key)
        values.delete(key)
        if (oldValue !== undefined) {
          for (const listener of harness.changes) listener({ [key]: {
            oldValue: safeStorageSerde.serializer(oldValue), newValue: undefined
          } }, area)
        }
      }
    }
  }
  }
})
vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "warm",
  initBackground: async () => {}
}))
vi.mock("@/entries/shared/notification-subscription", () => ({
  startNotificationSubscription: async () => {}
}))
vi.mock("wxt/browser", () => {
  const event = () => ({ addListener: vi.fn() })
  return {
    browser: {
      runtime: {
        id: "extension-id",
        getURL: (path: string) => `chrome-extension://extension-id${path}`,
        sendMessage: async (message: unknown) => {
          harness.sent.push(message)
        },
        onConnect: event(),
        onStartup: event(),
        onMessage: {
          addListener: (
            listener: Parameters<typeof harness.listeners.add>[0]
          ) => harness.listeners.add(listener)
        }
      },
      storage: {
        local: { get: async () => ({}), set: async () => {} },
        session: { get: async () => harness.saved, set: async (items: Record<string, unknown>) => { Object.assign(harness.saved, items) } },
        onChanged: { addListener: (listener: Parameters<typeof harness.changes.add>[0]) => harness.changes.add(listener) }
      },
      alarms: {
        get: async () => undefined,
        clear: async () => true,
        create: async () => {},
        onAlarm: event()
      },
      tabs: {
        query: async () => [],
        create: vi.fn(),
        sendMessage: async () => {}
      },
      action: { onClicked: event() },
      contextMenus: { create: vi.fn(), removeAll: vi.fn(), onClicked: event() },
      i18n: { getMessage: (key: string) => key }
    }
  }
})
const send = (message: unknown): Promise<unknown> =>
  new Promise((resolve) => {
    const listener = [...harness.listeners][0]
    if (!listener) throw new Error("Missing background listener")
    listener(message, { id: "extension-id" }, resolve)
  })
const json = (value: unknown, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" }
  })

import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
const config = { serverUrl: "https://api.example.test/base", authMode: "single-user" as const, authSource: "manual" as const, apiKey: "worker-key", credentialSource: "manual" as const, apiKeyPersistence: "device" as const, apiKeyServerOrigin: "https://api.example.test" }
const requestScope = { config: { serverUrl: config.serverUrl, authMode: config.authMode, authSource: config.authSource, expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope("single-user", "worker-key")! }, userId: null }
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done }); return { promise, resolve } }

const urlA = "https://source.test/talk-a.mp4"
const urlB = "https://source.test/talk-b.mp4"
const urlC = "https://source.test/paper-c.pdf"

const sentOfType = (type: string) =>
  harness.sent.filter((message) => message?.type === type)

beforeEach(async () => {
  vi.resetModules(); harness.listeners.clear(); harness.changes.clear(); harness.values.clear(); harness.sessionValues.clear(); harness.sent.length = 0; harness.saved = {}
  harness.values.set("tldwConfig", config)
  vi.stubGlobal("window", undefined)
  vi.stubGlobal("chrome", { storage: (await import("wxt/browser")).browser.storage })
  const background = (await import("@/entries/background")).default
  background.main()
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.unstubAllEnvs(); vi.useRealTimers() })

// Shared fetch router for URL-entry batches: derives the submit response from
// the urls actually present in the submitted form (one job per url, its source
// echoed back), tracks batches by id, and completes every job on poll.
const createUrlBatchRouter = () => {
  const batches = new Map<string, Array<{ id: number; source: string }>>()
  let submitCount = 0
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (url.endsWith("/api/v1/media/ingest/jobs") && init?.method === "POST") {
      const urls = (init.body as FormData).getAll("urls") as string[]
      submitCount += 1
      const batchId = `batch-${submitCount}`
      const jobs = urls.map((source, index) => ({ id: submitCount * 100 + index, source }))
      batches.set(batchId, jobs)
      return json({ batch_id: batchId, jobs: jobs.map((job) => ({ ...job, source_kind: "url" })) })
    }
    const batchMatch = url.match(/batch_id=([^&]+)/)
    if (batchMatch && url.includes("/api/v1/media/ingest/jobs?")) {
      const jobs = batches.get(batchMatch[1]) ?? []
      return json({
        jobs: jobs.map((job) => ({ ...job, status: "completed", result: { media_id: job.id } }))
      })
    }
    throw new Error(`Unexpected request: ${init?.method} ${url}`)
  })
  return { fetcher, batches }
}

describe("quick ingest submission batching (WP-36)", () => {
  it("submits same-config URL entries as one ingest-jobs request with k urls", async () => {
    const { fetcher } = createUrlBatchRouter()
    const fetchCalls = fetcher.mock.calls
    vi.stubGlobal("fetch", fetcher)

    const ack = (await send({
      type: "tldw:quick-ingest/start",
      payload: {
        requestScope,
        entries: [
          { id: "e1", url: urlA, type: "video" },
          { id: "e2", url: urlB, type: "video" },
          { id: "e3", url: urlC, type: "video" }
        ],
        files: [],
        storeRemote: true,
        processOnly: false
      }
    })) as { sessionId: string }

    await vi.waitFor(() =>
      expect(sentOfType("tldw:quick-ingest/completed")).toHaveLength(1)
    )

    const submitCalls = fetchCalls.filter(([url, init]) =>
      String(url).endsWith("/api/v1/media/ingest/jobs") && init?.method === "POST"
    )
    // One submission request for the three same-config entries (HEAD issued
    // one POST per URL), carrying all three urls.
    expect(submitCalls).toHaveLength(1)
    const submitBody = submitCalls[0][1]?.body as FormData
    expect(submitBody.getAll("urls")).toEqual([urlA, urlB, urlC])
    expect(submitBody.getAll("media_type")).toEqual(["video"])

    const completed = sentOfType("tldw:quick-ingest/completed")[0]
    expect(completed.payload.sessionId).toBe(ack.sessionId)
    // Each returned job maps back to its own entry, in entry order (the
    // router gives job 100→urlA, 101→urlB, 102→urlC, media_id = job id).
    expect(completed.payload.results).toEqual([
      expect.objectContaining({ id: "e1", status: "ok", url: urlA, data: { media_id: 100 } }),
      expect.objectContaining({ id: "e2", status: "ok", url: urlB, data: { media_id: 101 } }),
      expect.objectContaining({ id: "e3", status: "ok", url: urlC, data: { media_id: 102 } })
    ])
  })

  it("batches same-config file uploads into one multi-file ingest-jobs request", async () => {
    const batches = new Map<string, number[]>()
    let submitCount = 0
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith("/api/v1/media/ingest/jobs") && init?.method === "POST") {
        const files = (init.body as FormData).getAll("files")
        submitCount += 1
        const batchId = `file-batch-${submitCount}`
        const jobs = files.map((_, index) => 200 + submitCount * 10 + index)
        batches.set(batchId, jobs)
        return json({ batch_id: batchId, jobs: jobs.map((id) => ({ id })) })
      }
      const batchMatch = url.match(/batch_id=([^&]+)/)
      if (batchMatch && url.includes("/api/v1/media/ingest/jobs?")) {
        const jobs = batches.get(batchMatch[1]) ?? []
        return json({
          jobs: jobs.map((id) => ({ id, status: "completed", result: { media_id: id } }))
        })
      }
      throw new Error(`Unexpected request: ${init?.method} ${url}`)
    })
    vi.stubGlobal("fetch", fetcher)

    await send({
      type: "tldw:quick-ingest/start",
      payload: {
        requestScope,
        entries: [],
        files: [
          { id: "f1", name: "one.pdf", type: "application/pdf", data: { data: [104, 105, 33] } },
          { id: "f2", name: "two.pdf", type: "application/pdf", data: { data: [104, 105, 34] } }
        ],
        storeRemote: true,
        processOnly: false
      }
    })

    await vi.waitFor(() =>
      expect(sentOfType("tldw:quick-ingest/completed")).toHaveLength(1)
    )
    const submitCalls = fetcher.mock.calls.filter(([url, init]) =>
      String(url).endsWith("/api/v1/media/ingest/jobs") && init?.method === "POST"
    )
    expect(submitCalls).toHaveLength(1)
    expect((submitCalls[0][1]?.body as FormData).getAll("files")).toHaveLength(2)
    const completed = sentOfType("tldw:quick-ingest/completed")[0]
    expect(completed.payload.results).toEqual([
      expect.objectContaining({ id: "f1", status: "ok", fileName: "one.pdf" }),
      expect.objectContaining({ id: "f2", status: "ok", fileName: "two.pdf" })
    ])
  })
})

describe("quick ingest progress event preservation (W2 interplay)", () => {
  it("emits per-item progress events in the same shape and order as sequential HEAD", async () => {
    const { fetcher } = createUrlBatchRouter()
    vi.stubGlobal("fetch", fetcher)

    const ack = (await send({
      type: "tldw:quick-ingest/start",
      payload: {
        requestScope,
        entries: [
          { id: "e1", url: urlA, type: "video" },
          { id: "e2", url: urlB, type: "video" },
          { id: "e3", url: urlC, type: "video" }
        ],
        files: [],
        storeRemote: true,
        processOnly: false
      }
    })) as { sessionId: string }

    await vi.waitFor(() =>
      expect(sentOfType("tldw:quick-ingest/completed")).toHaveLength(1)
    )

    const progress = sentOfType("tldw:quick-ingest/progress")
    // N=3 batch emits exactly 3 per-item events, input-ordered, with the same
    // payload shape the wizard consumed under sequential submissions.
    expect(progress).toHaveLength(3)
    expect(progress.map((message) => message.payload.processedCount)).toEqual([1, 2, 3])
    expect(progress.map((message) => message.payload.result.id)).toEqual(["e1", "e2", "e3"])
    expect(progress.map((message) => message.payload.result.status)).toEqual(["ok", "ok", "ok"])
    for (const message of progress) {
      expect(message.payload).toMatchObject({
        sessionId: ack.sessionId,
        totalCount: 3
      })
    }
    // Terminal message strictly follows the last per-item event.
    const lastProgressIndex = harness.sent.lastIndexOf(progress[2])
    const completedIndex = harness.sent.indexOf(sentOfType("tldw:quick-ingest/completed")[0])
    expect(completedIndex).toBeGreaterThan(lastProgressIndex)
  })
})

describe("ingest session poll round concurrency (WP-37)", () => {
  it("issues every per-job status GET of a round concurrently", async () => {
    const job11 = deferred<Response>()
    const job12 = deferred<Response>()
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith("/api/v1/media/ingest/jobs/11")) return job11.promise
      if (url.endsWith("/api/v1/media/ingest/jobs/12")) return job12.promise
      throw new Error(`Unexpected request: ${url}`)
    })
    vi.stubGlobal("fetch", fetcher)

    // Rehydrate a persisted ingest session (worker restart) holding two job
    // ids; resume drives pollIngestJobsForSession for both.
    harness.saved = {
      "tldw:backgroundSessionStateV1": {
        ingestSessions: {
          "funnel-resume": {
            funnelId: "funnel-resume",
            url: "https://source.test/watch?v=1",
            normalizedUrl: "https://source.test/watch?v=1",
            status: "queued",
            jobIds: [11, 12],
            createdAt: Date.now(),
            retryCount: 0,
            awaitingAuth: false
          }
        },
        pendingAuthReplay: [],
        quickIngestSessions: [],
        quickIngestBatches: []
      }
    }
    harness.listeners.clear()
    const background = (await import("@/entries/background")).default
    background.main()

    // Both per-job GETs must be inflight simultaneously: neither response has
    // resolved, so a sequential implementation would still be waiting on 11.
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2))
    job11.resolve(json({ status: "completed", result: { media_id: 7 } }))
    job12.resolve(json({ status: "running" }))
    await vi.waitFor(() =>
      expect(
        harness.sent.some(
          (message) =>
            message?.type === "media-ingest-status" &&
            message.payload?.funnelId === "funnel-resume" &&
            message.payload?.status === "completed"
        )
      ).toBe(true)
    )
  })
})

describe("terminal PATCH and cancel fan-out concurrency (WP-38)", () => {
  it("fans terminal conference item PATCHes out concurrently", async () => {
    const terminalPatches = { 501: deferred<Response>(), 502: deferred<Response>() }
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === "https://api.example.test/base/api/v1/media/collections" && init?.method === "POST") {
        return json({ id: 5 })
      }
      if (url.endsWith("/api/v1/media/collections/5/items") && init?.method === "POST") {
        const body = String(init.body)
        return json({ id: body.includes(urlA) ? 501 : 502 })
      }
      if (init?.method === "POST" && url.endsWith("/api/v1/media/ingest/jobs")) {
        const urls = (init.body as FormData).getAll("urls")
        return urls[0] === urlA
          ? json({ batch_id: "b1", jobs: [{ id: 601, source: urlA }] })
          : json({ batch_id: "b2", jobs: [{ id: 602, source: urlB }] })
      }
      if (url.includes("batch_id=b1")) {
        return json({ jobs: [{ id: 601, status: "completed", result: { media_id: 61 } }] })
      }
      if (url.includes("batch_id=b2")) {
        return json({ jobs: [{ id: 602, status: "completed", result: { media_id: 62 } }] })
      }
      const patchMatch = url.match(/\/api\/v1\/media\/collections\/5\/items\/(\d+)$/)
      if (patchMatch && init?.method === "PATCH") {
        const itemId = Number(patchMatch[1])
        const body = JSON.parse(String(init.body))
        // "processing" patches settle immediately; terminal patches are held.
        if (body.status === "processing") return json({ id: itemId })
        return terminalPatches[itemId as 501 | 502].promise
      }
      throw new Error(`Unexpected request: ${init?.method} ${url}`)
    })
    vi.stubGlobal("fetch", fetcher)

    await send({
      type: "tldw:quick-ingest/start",
      payload: {
        requestScope,
        entries: [
          { id: "e1", url: urlA, type: "video" },
          { id: "e2", url: urlB, type: "video" }
        ],
        files: [],
        storeRemote: true,
        processOnly: false,
        conferenceBatchMetadata: {
          collectionName: "Conference talks",
          conferenceName: "Conference",
          eventYear: "2026",
          sharedTags: []
        }
      }
    })

    // Both terminal PATCHes are issued while neither has settled: a
    // sequential per-result loop would block on the first one.
    await vi.waitFor(() => {
      expect(fetcher.mock.calls.filter(([url, init]) =>
        String(url).match(/\/api\/v1\/media\/collections\/5\/items\/\d+$/) &&
        init?.method === "PATCH" &&
        JSON.parse(String(init.body)).status === "completed"
      )).toHaveLength(2)
    })
    terminalPatches[501].resolve(json({ id: 501 }))
    terminalPatches[502].resolve(json({ id: 502 }))

    const completed = await vi.waitFor(() => {
      const [message] = sentOfType("tldw:quick-ingest/completed")
      expect(message).toBeTruthy()
      return message
    })
    expect(completed.payload.results.map((result: any) => result.id)).toEqual(["e1", "e2"])
  })

  it("fans batch cancel POSTs out concurrently on user cancel", async () => {
    const cancel1 = deferred<Response>()
    const cancel2 = deferred<Response>()
    const poll1 = deferred<Response>()
    const poll2 = deferred<Response>()
    const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.includes("/ingest/jobs/cancel") && url.includes("batch_id=b1")) return cancel1.promise
      if (url.includes("/ingest/jobs/cancel") && url.includes("batch_id=b2")) return cancel2.promise
      if (init?.method === "POST" && url.endsWith("/api/v1/media/ingest/jobs")) {
        const urls = (init.body as FormData).getAll("urls")
        return urls[0] === urlA
          ? json({ batch_id: "b1", jobs: [{ id: 701, source: urlA }] })
          : json({ batch_id: "b2", jobs: [{ id: 702, source: urlB }] })
      }
      if (url.includes("/api/v1/media/ingest/jobs?") && url.includes("batch_id=b1")) return poll1.promise
      if (url.includes("/api/v1/media/ingest/jobs?") && url.includes("batch_id=b2")) return poll2.promise
      throw new Error(`Unexpected request: ${init?.method} ${url}`)
    })
    vi.stubGlobal("fetch", fetcher)

    const ack = (await send({
      type: "tldw:quick-ingest/start",
      payload: {
        requestScope,
        // Distinct keywords keep the entries heterogeneous: two separate
        // submissions, two distinct server batch ids to cancel.
        entries: [
          { id: "e1", url: urlA, type: "video", keywords: "alpha" },
          { id: "e2", url: urlB, type: "video", keywords: "beta" }
        ],
        files: [],
        storeRemote: true,
        processOnly: false
      }
    })) as { sessionId: string }

    // Wait for the first poll round to be inflight for both batches.
    await vi.waitFor(() => {
      expect(fetcher.mock.calls.filter(([url]) => String(url).includes("/api/v1/media/ingest/jobs?") && String(url).includes("batch_id=b1")).length).toBe(1)
      expect(fetcher.mock.calls.filter(([url]) => String(url).includes("/api/v1/media/ingest/jobs?") && String(url).includes("batch_id=b2")).length).toBe(1)
    })

    expect(await send({
      type: "tldw:quick-ingest/cancel",
      payload: { sessionId: ack.sessionId, requestScope }
    })).toMatchObject({ ok: true })

    // Settle the in-flight poll round so the cancelled poll wakes up (it
    // sleeps one poll interval before re-checking the cancel flag).
    poll1.resolve(json({ jobs: [{ id: 701, status: "running" }] }))
    poll2.resolve(json({ jobs: [{ id: 702, status: "running" }] }))

    // Both batch cancels are issued while neither has settled: a sequential
    // cancel loop would block on the first one.
    await vi.waitFor(
      () => {
        expect(
          fetcher.mock.calls.filter(([url]) => String(url).includes("/ingest/jobs/cancel"))
        ).toHaveLength(2)
      },
      { timeout: 8000 }
    )
    cancel1.resolve(json({ cancelled: 1 }))
    cancel2.resolve(json({ cancelled: 1 }))
  })
})
