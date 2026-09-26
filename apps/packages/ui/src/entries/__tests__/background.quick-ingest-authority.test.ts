import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const harness = vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (value: unknown) => value
  })
  return {
    saved: {} as Record<string, unknown>,
    changes: new Set<(changes: Record<string, { oldValue?: unknown; newValue?: unknown }>, area: string) => void>(),
    values: new Map<string, unknown>(),
    listeners: new Set<
      (
        message: unknown,
        sender: unknown,
        reply: (value: unknown) => void
      ) => unknown
    >(),
    sent: [] as unknown[]
  }
})
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  },
  createSafeStorage: () => ({
    get: async (key: string) => harness.values.get(key),
    set: async (key: string, value: unknown) => {
      harness.values.set(key, value)
    },
    remove: async (key: string) => {
      harness.values.delete(key)
    }
  })
}))
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
          return send(message)
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
const config = { serverUrl: "https://api.example.test/base", authMode: "single-user" as const, authSource: "manual" as const, apiKey: "worker-key", credentialSource: "manual", apiKeyPersistence: "device", apiKeyServerOrigin: "https://api.example.test" }
const requestScope = { config: { serverUrl: config.serverUrl, authMode: config.authMode, authSource: config.authSource, expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope("single-user", "worker-key")! }, userId: null }
const batch = { requestScope, entries: [{ id: "private", url: "https://source.test/doc.pdf", type: "pdf" }], files: [], storeRemote: true, processOnly: false }
const switchConfig = (next: typeof config) => {
  const previous = harness.values.get("tldwConfig")
  harness.values.set("tldwConfig", next)
  for (const listener of harness.changes) listener({ tldwConfig: { oldValue: previous, newValue: next } }, "local")
}
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done }); return { promise, resolve } }

beforeEach(async () => {
  vi.resetModules(); harness.listeners.clear(); harness.changes.clear(); harness.values.clear(); harness.sent.length = 0; harness.saved = {}
  harness.values.set("tldwConfig", config)
  vi.stubGlobal("window", undefined)
  vi.stubGlobal("chrome", { storage: (await import("wxt/browser")).browser.storage })
  const background = (await import("@/entries/background")).default
  background.main()
})
afterEach(() => vi.unstubAllGlobals())

describe("native Quick Ingest worker ownership", () => {
  it("keeps processing after the initiating UI message returns and no UI listener remains", async () => {
    const uploaded = deferred<Response>()
    const fetcher = vi.fn(async (url: string) => url.endsWith("/ingest/jobs") ? uploaded.promise : json({ status: "completed", result: { media_id: 7 } }))
    vi.stubGlobal("fetch", fetcher)
    const ack = await send({ type: "tldw:quick-ingest/start", payload: batch }) as { sessionId: string }
    expect(ack.sessionId).toBeTruthy()
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    // The UI is already gone: only the worker's runtime handler remains.
    uploaded.resolve(json({ batch_id: "owned-batch", jobs: [{ id: 7 }] }))
    await vi.waitFor(() => expect(harness.sent).toContainEqual(expect.objectContaining({ type: "tldw:quick-ingest/completed", payload: expect.objectContaining({ sessionId: ack.sessionId }) })))
    expect(fetcher.mock.calls.map(call => call[0])).toEqual(["https://api.example.test/base/api/v1/media/ingest/jobs", "https://api.example.test/base/api/v1/media/ingest/jobs/7"])
  })

  it("rejects a legacy unowned start and a foreign colliding cancellation before dispatch", async () => {
    const fetcher = vi.fn(); vi.stubGlobal("fetch", fetcher)
    expect(await send({ type: "tldw:quick-ingest/start", payload: { ...batch, requestScope: undefined } })).toMatchObject({ ok: false })
    expect(await send({ type: "tldw:quick-ingest/cancel", payload: { sessionId: "qi-old", requestScope: { ...requestScope, config: { ...requestScope.config, serverUrl: "https://foreign.test" } } } })).toMatchObject({ ok: false })
    expect(fetcher).not.toHaveBeenCalled()
  })

  it.each([false, true])("resumes saved worker jobs only for their exact stored authority (foreign=%s)", async (foreign) => {
    const fetcher = vi.fn(async () => json({ status: "completed", result: { media_id: 7 } })); vi.stubGlobal("fetch", fetcher)
    // A fresh worker hydrates only metadata, never raw credentials.
    harness.listeners.clear(); harness.changes.clear()
    harness.saved = { "tldw:backgroundSessionStateV1": {
      ingestSessions: {}, pendingAuthReplay: [],
      quickIngestSessions: [{ sessionId: "qi-restored", cancelled: false, requestScope }],
      quickIngestBatches: [{ sessionId: "qi-restored", requestScope, totalCount: 1, processedCount: 0, ingestTimeoutMs: 60000,
        remoteJobs: [{ jobId: 7, batchId: "saved-batch", meta: { id: "private", type: "pdf", fileName: "Owned.pdf" } }], collectedResults: [], plannedConferenceItems: [] }]
    } }
    if (foreign) harness.values.set("tldwConfig", { ...config, serverUrl: "https://foreign.test", apiKeyServerOrigin: "https://foreign.test" })
    const background = (await import("@/entries/background")).default
    background.main()
    if (foreign) {
      await new Promise(resolve => setTimeout(resolve, 30))
      expect(fetcher).not.toHaveBeenCalled()
      expect(harness.sent.filter((message) => typeof message === "object" && message !== null && "type" in message && typeof message.type === "string" && message.type.startsWith("tldw:quick-ingest/"))).toEqual([])
    } else {
      await vi.waitFor(() => expect(harness.sent).toContainEqual(expect.objectContaining({ type: "tldw:quick-ingest/completed", payload: expect.objectContaining({ sessionId: "qi-restored" }) })))
      expect(fetcher).toHaveBeenCalledTimes(1)
      expect(fetcher.mock.calls[0][0]).toBe("https://api.example.test/base/api/v1/media/ingest/jobs/7")
    }
  })

  it("drops an upload completion across A to B to A without polling or server-job cancellation", async () => {
    const uploaded = deferred<Response>()
    const fetcher = vi.fn(async () => uploaded.promise); vi.stubGlobal("fetch", fetcher)
    await send({ type: "tldw:quick-ingest/start", payload: batch })
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    switchConfig({ ...config, serverUrl: "https://other.test", apiKeyServerOrigin: "https://other.test" }); switchConfig(config)
    uploaded.resolve(json({ batch_id: "foreign-colliding-batch", jobs: [{ id: 7 }] }))
    await new Promise(resolve => setTimeout(resolve, 30))
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(harness.sent.filter((message) => typeof message === "object" && message !== null && "type" in message && typeof message.type === "string" && message.type.startsWith("tldw:quick-ingest/"))).toEqual([])
  })

  it('regression worker stale marker check preserves a valid rotated in-flight ingest', async () => {
    const {refreshSessionInvalidationKey,storeRefreshRotationIfCurrent}=await import('@/services/tldw/single-user-credential')
    const token=(revision:number)=>'test.'+btoa(JSON.stringify({sub:'1',revision}))+'.signature'
    const original={serverUrl:config.serverUrl,authMode:'multi-user' as const,authSource:'manual' as const,accessToken:token(1),refreshToken:'worker-source-refresh'}
    harness.values.set('tldwConfig',original)
    const ownScope={config:{serverUrl:original.serverUrl,authMode:original.authMode,authSource:original.authSource},userId:1}
    const uploaded=deferred<Response>()
    const fetcher=vi.fn(async(url:string)=>url.endsWith('/ingest/jobs')?uploaded.promise:json({status:'completed',result:{media_id:7}}))
    vi.stubGlobal('fetch',fetcher)
    const ack=await send({type:'tldw:quick-ingest/start',payload:{...batch,requestScope:ownScope}}) as {sessionId:string}
    await vi.waitFor(()=>expect(fetcher).toHaveBeenCalledTimes(1))
    const key=refreshSessionInvalidationKey(original)!
    harness.values.set(key,true)
    const pending=deferred<boolean>()
    let intercepted=false
    const rawGet=harness.values.get.bind(harness.values)
    const get=vi.spyOn(harness.values,'get').mockImplementation(keyName=>{
      if(keyName===key&&!intercepted){intercepted=true;return pending.promise}
      return rawGet(keyName)
    })
    try {
      for(const listener of harness.changes)listener({[key]:{newValue:true}},'local')
      await vi.waitFor(()=>expect(intercepted).toBe(true))
      const storage={get:async<T,>(key:string)=>harness.values.get(key) as T,set:async<T,>(key:string,value:T)=>{harness.values.set(key,value)},remove:async(key:string)=>{harness.values.delete(key)}}
      expect(await storeRefreshRotationIfCurrent(storage,original,original.refreshToken,{accessToken:token(2),refreshToken:'worker-rotated-refresh'})).toBe(true)
      for(const listener of harness.changes)listener({tldwRefreshRotation:{newValue:harness.values.get('tldwRefreshRotation')}},'local')
      pending.resolve(true)
      await new Promise(resolve=>setTimeout(resolve,0))
      uploaded.resolve(json({batch_id:'owned-batch',jobs:[{id:7}]}))
      await vi.waitFor(()=>expect(harness.sent).toContainEqual(expect.objectContaining({type:'tldw:quick-ingest/completed',payload:expect.objectContaining({sessionId:ack.sessionId})})))
      expect(fetcher).toHaveBeenCalledTimes(2)
    } finally {pending.resolve(true);uploaded.resolve(json({batch_id:'owned-batch',jobs:[{id:7}]}));get.mockRestore()}
  })

  it("abandons current terminal worker credentials without cancelling another owner's jobs", async () => {
    const { refreshSessionInvalidationKey } = await import("@/services/tldw/single-user-credential")
    const original = { serverUrl: config.serverUrl, authMode: "multi-user" as const, authSource: "manual" as const, accessToken: "test." + btoa(JSON.stringify({ sub: "1" })) + ".signature", refreshToken: "terminal-refresh" }
    harness.values.set("tldwConfig", original)
    const ownScope = { config: { serverUrl: original.serverUrl, authMode: original.authMode, authSource: original.authSource }, userId: 1 }
    const uploaded = deferred<Response>()
    const fetcher = vi.fn(async (_url: string, _init?: RequestInit) => uploaded.promise)
    vi.stubGlobal("fetch", fetcher)
    await send({ type: "tldw:quick-ingest/start", payload: { ...batch, requestScope: ownScope } })
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    const key = refreshSessionInvalidationKey(original)!
    harness.values.set(key, true)
    for (const listener of harness.changes) listener({ [key]: { newValue: true } }, "local")
    await vi.waitFor(() => expect(fetcher.mock.calls[0][1]?.signal?.aborted).toBe(true))
    uploaded.resolve(json({ batch_id: "old-owner", jobs: [{ id: 7 }] }))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(fetcher).toHaveBeenCalledTimes(1)
  })
})
