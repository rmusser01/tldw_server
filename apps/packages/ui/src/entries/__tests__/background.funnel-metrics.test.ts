import { beforeEach, describe, expect, it, vi } from "vitest"

const runtimeMessageListeners = vi.hoisted(() => ({
  listeners: new Set<(message: any, sender: any, sendResponse: any) => void>()
}))

const localArea = vi.hoisted(() => ({
  store: new Map<string, unknown>(),
  gets: [] as string[],
  sets: [] as Array<{ key: string; value: unknown }>
}))

vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (options: unknown) => options
  })
  return {}
})

vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  },
  createSafeStorage: (options?: { area?: string }) => ({
    get: async (key: string) => {
      if (options?.area !== "local") return undefined
      localArea.gets.push(key)
      return localArea.store.has(key) ? localArea.store.get(key) : undefined
    },
    set: async (key: string, value: unknown) => {
      if (options?.area !== "local") return
      localArea.sets.push({ key, value })
      localArea.store.set(key, value)
    },
    remove: async () => undefined,
    watch: () => undefined
  })
}))

vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "tldw:model-warm",
  initBackground: vi.fn(async () => ({ modelWarmAlarmName: "tldw:model-warm" }))
}))

vi.mock("@/entries/shared/notification-subscription", () => ({
  startNotificationSubscription: vi.fn(async () => {})
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      id: "extension-id",
      getURL: (path: string) => `chrome-extension://extension-id${path}`,
      sendMessage: vi.fn(async () => ({ handled: true })),
      onMessage: {
        addListener: (
          listener: (message: any, sender: any, sendResponse: any) => void
        ) => {
          runtimeMessageListeners.listeners.add(listener)
        },
        removeListener: (
          listener: (message: any, sender: any, sendResponse: any) => void
        ) => {
          runtimeMessageListeners.listeners.delete(listener)
        }
      },
      onStartup: {
        addListener: vi.fn()
      },
      onConnect: {
        addListener: vi.fn(),
        removeListener: vi.fn()
      }
    },
    storage: {
      local: {
        get: vi.fn(async () => ({})),
        set: vi.fn(async () => {})
      },
      session: {
        get: vi.fn(async () => ({})),
        set: vi.fn(async () => {})
      },
      onChanged: {
        addListener: vi.fn(),
        removeListener: vi.fn()
      }
    },
    alarms: {
      clear: vi.fn(async () => true),
      create: vi.fn(async () => {}),
      onAlarm: {
        addListener: vi.fn()
      }
    },
    tabs: {
      create: vi.fn()
    },
    action: {
      onClicked: {
        addListener: vi.fn()
      }
    },
    contextMenus: {
      create: vi.fn(),
      removeAll: vi.fn(),
      remove: vi.fn(),
      onClicked: {
        addListener: vi.fn()
      }
    },
    commands: {
      onCommand: {
        addListener: vi.fn()
      }
    },
    i18n: {
      getMessage: (key: string) => key
    }
  }
}))

import background from "@/entries/background"

const FUNNEL_METRICS_KEY = "tldw:ingestFunnelMetrics"

const settle = async () => {
  await new Promise<void>((resolve) => setTimeout(resolve, 0))
  await new Promise<void>((resolve) => setTimeout(resolve, 0))
}

const sendFunnelEvent = (
  event: string,
  funnelId: string,
  metadata: Record<string, unknown> = {}
) => {
  for (const listener of runtimeMessageListeners.listeners) {
    listener(
      {
        type: "tldw:media-ingest/funnel-event",
        payload: { event, funnelId, metadata }
      },
      { id: "extension-id" },
      () => undefined
    )
  }
}

const metricWrites = () =>
  localArea.sets.filter((entry) => entry.key === FUNNEL_METRICS_KEY)

const metricReads = () => localArea.gets.filter((key) => key === FUNNEL_METRICS_KEY)

const persistedMetrics = () =>
  localArea.store.get(FUNNEL_METRICS_KEY) as Array<Record<string, unknown>>

describe("background ingest funnel metrics writer", () => {
  beforeEach(() => {
    runtimeMessageListeners.listeners.clear()
    localArea.store.clear()
    localArea.gets.length = 0
    localArea.sets.length = 0
    background.main()
  })

  it("persists a funnel event and acknowledges the runtime message", async () => {
    sendFunnelEvent("context_click", "ingest-1", { url: "https://example.com" })
    await vi.waitFor(() => expect(metricWrites()).toHaveLength(1))

    expect(persistedMetrics()).toEqual([
      expect.objectContaining({
        event: "context_click",
        funnelId: "ingest-1",
        metadata: { url: "https://example.com" },
        timestamp: expect.any(String)
      })
    ])
  })

  it("coalesces a burst of funnel events into a single storage write", async () => {
    for (let index = 0; index < 5; index += 1) {
      sendFunnelEvent("job_queued", `ingest-${index}`)
    }
    await vi.waitFor(() => expect(metricWrites().length).toBeGreaterThanOrEqual(1))
    await settle()

    // WP-61: one drain window, one read-modify-write — NOT one per event.
    expect(metricWrites()).toHaveLength(1)
    expect(metricReads().length).toBeLessThanOrEqual(1)

    const persisted = persistedMetrics()
    expect(persisted).toHaveLength(5)
    expect(persisted.map((entry) => entry.funnelId)).toEqual([
      "ingest-0",
      "ingest-1",
      "ingest-2",
      "ingest-3",
      "ingest-4"
    ])
    for (const entry of persisted) {
      expect(entry).toMatchObject({
        event: "job_queued",
        timestamp: expect.any(String)
      })
    }
  })

  it("merges new events with previously persisted history on the first flush", async () => {
    localArea.store.set(FUNNEL_METRICS_KEY, [
      {
        event: "context_click",
        funnelId: "ingest-old",
        timestamp: "2026-11-01T00:00:00.000Z",
        metadata: {}
      }
    ])

    sendFunnelEvent("job_queued", "ingest-new")
    await vi.waitFor(() => expect(metricWrites()).toHaveLength(1))

    const persisted = persistedMetrics()
    expect(persisted).toHaveLength(2)
    expect(persisted.map((entry) => entry.funnelId)).toEqual([
      "ingest-old",
      "ingest-new"
    ])
  })

  it("bounds the persisted funnel metrics at 200 events", async () => {
    for (let index = 0; index < 205; index += 1) {
      sendFunnelEvent("job_queued", `ingest-${index}`)
    }
    await vi.waitFor(() =>
      expect(persistedMetrics()?.length).toBe(200)
    )
    await settle()

    const persisted = persistedMetrics()
    expect(persisted).toHaveLength(200)
    // The oldest events are dropped, the newest 200 survive in arrival order.
    expect(persisted[0]).toMatchObject({ funnelId: "ingest-5" })
    expect(persisted[199]).toMatchObject({ funnelId: "ingest-204" })
  })
})
