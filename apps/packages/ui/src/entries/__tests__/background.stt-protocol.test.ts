import { beforeEach, describe, expect, it, vi } from "vitest"

const connectListeners = new Set<(port: any) => void>()
const runtimeMessageListeners = new Set<(message: unknown) => void>()
const storageChangeListeners = new Set<(changes: unknown, areaName: string) => void>()
const alarmListeners = new Set<(alarm: { name: string }) => void>()
const startupListeners = new Set<() => void>()

const storageState = vi.hoisted(() => ({
  tldwConfig: {
    serverUrl: "http://localhost:8000",
    authMode: "single-user",
    apiKey: "test-key"
  }
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
  createSafeStorage: () => ({
    get: async (key: string) =>
      key === "tldwConfig" ? storageState.tldwConfig : undefined,
    set: vi.fn()
  })
}))

vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "model-warm",
  initBackground: vi.fn(async () => {})
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
      onConnect: {
        addListener: (listener: (port: any) => void) => {
          connectListeners.add(listener)
        },
        removeListener: (listener: (port: any) => void) => {
          connectListeners.delete(listener)
        }
      },
      onMessage: {
        addListener: (listener: (message: unknown) => void) => {
          runtimeMessageListeners.add(listener)
        },
        removeListener: (listener: (message: unknown) => void) => {
          runtimeMessageListeners.delete(listener)
        }
      },
      onStartup: {
        addListener: (listener: () => void) => {
          startupListeners.add(listener)
        }
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
        addListener: (
          listener: (changes: unknown, areaName: string) => void
        ) => {
          storageChangeListeners.add(listener)
        }
      }
    },
    alarms: {
      clear: vi.fn(async () => true),
      create: vi.fn(async () => {}),
      onAlarm: {
        addListener: (listener: (alarm: { name: string }) => void) => {
          alarmListeners.add(listener)
        }
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

class MockWebSocket {
  static OPEN = 1
  static CLOSED = 3
  static instances: MockWebSocket[] = []

  readyState = 0
  binaryType = "blob"
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null

  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }

  send(payload: string) {
    this.sent.push(payload)
  }

  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.()
  }
}

type RuntimePortHarness = {
  messages: unknown[]
  postMessage: (message: unknown) => void
  disconnect: () => void
}

const connectRuntimePort = (name: string): RuntimePortHarness => {
  const inboundListeners = new Set<(message: unknown) => void>()
  const disconnectListeners = new Set<() => void>()
  const messages: unknown[] = []
  const port = {
    name,
    sender: { id: "extension-id" },
    postMessage: (message: unknown) => {
      messages.push(message)
    },
    onMessage: {
      addListener: (listener: (message: unknown) => void) => {
        inboundListeners.add(listener)
      },
      removeListener: (listener: (message: unknown) => void) => {
        inboundListeners.delete(listener)
      }
    },
    onDisconnect: {
      addListener: (listener: () => void) => {
        disconnectListeners.add(listener)
      }
    },
    disconnect: () => {
      for (const listener of disconnectListeners) listener()
    }
  }

  for (const listener of connectListeners) listener(port)

  return {
    messages,
    postMessage: (message: unknown) => {
      for (const listener of inboundListeners) listener(message)
    },
    disconnect: port.disconnect
  }
}

import background from "@/entries/background"

describe("background STT audio protocol", () => {
  beforeEach(() => {
    connectListeners.clear()
    runtimeMessageListeners.clear()
    storageChangeListeners.clear()
    alarmListeners.clear()
    startupListeners.clear()
    MockWebSocket.instances = []
    ;(globalThis as any).WebSocket = MockWebSocket
    background.main()
  })

  it("sends captions config before reporting STT open and wraps audio as JSON", async () => {
    const port = connectRuntimePort("tldw:stt")

    port.postMessage({ action: "connect" })
    await Promise.resolve()

    const ws = MockWebSocket.instances[0]
    expect(ws.url).toBe("ws://localhost:8000/api/v1/audio/stream/transcribe")

    ws.open()

    expect(JSON.parse(ws.sent[0])).toMatchObject({ type: "auth" })
    expect(JSON.parse(ws.sent[1])).toMatchObject({
      type: "config",
      protocol_version: 1,
      mode: "captions",
      audio_format: "pcm16",
      sample_rate: 16000,
      channels: 1
    })
    expect(port.messages.at(-1)).toEqual({ event: "open" })

    port.postMessage({ action: "audio", data: new Uint8Array([0, 0]).buffer })

    expect(JSON.parse(ws.sent[2])).toMatchObject({
      type: "audio",
      data: "AAA="
    })
  })
})
