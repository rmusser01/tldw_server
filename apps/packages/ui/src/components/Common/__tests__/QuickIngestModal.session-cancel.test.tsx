import React from "react"
import { describe, expect, it, beforeEach, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import {
  createInitialQuickIngestLastRunSummary,
  useQuickIngestStore
} from "@/store/quick-ingest"

const mocks = vi.hoisted(() => ({
  startQuickIngestSession: vi.fn(),
  cancelQuickIngestSession: vi.fn(),
  submitQuickIngestBatch: vi.fn(),
  confirmDanger: vi.fn(),
  runtimeListeners: [] as Array<(message: any) => void>,
  tabsCreate: vi.fn(),
  checkOnce: vi.fn(),
  storageValues: new Map<string, unknown>(),
  storageSetters: new Map<string, (next: unknown) => void>()
}))

const translate = vi.hoisted(() => {
  return (
    key: string,
    defaultValueOrOptions?:
      | string
      | {
          defaultValue?: string
          [k: string]: unknown
        },
    interpolation?: Record<string, unknown>
  ) => {
    if (typeof defaultValueOrOptions === "string") {
      return defaultValueOrOptions.replace(/\{\{(\w+)\}\}/g, (_m, token) =>
        String(interpolation?.[token] ?? "")
      )
    }
    if (defaultValueOrOptions?.defaultValue) {
      return defaultValueOrOptions.defaultValue.replace(
        /\{\{(\w+)\}\}/g,
        (_m, token) => String(defaultValueOrOptions?.[token] ?? interpolation?.[token] ?? "")
      )
    }
    return key
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translate
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({
    data: [],
    isLoading: false
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, initialValue: unknown) => {
    const storageKey = String(key)
    if (!mocks.storageValues.has(storageKey)) {
      mocks.storageValues.set(storageKey, initialValue)
    }
    if (!mocks.storageSetters.has(storageKey)) {
      mocks.storageSetters.set(storageKey, (next: unknown) => {
        const prev = mocks.storageValues.get(storageKey)
        const resolved =
          typeof next === "function"
            ? (next as (prevValue: unknown) => unknown)(prev)
            : next
        mocks.storageValues.set(storageKey, resolved)
      })
    }
    return [
      mocks.storageValues.get(storageKey),
      mocks.storageSetters.get(storageKey)!,
      { isLoading: false }
    ]
  }
}))

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    useNavigate: () => vi.fn()
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    getOpenAPISpec: vi.fn().mockRejectedValue(new Error("no-spec")),
    getTranscriptionModels: vi.fn().mockResolvedValue([])
  }
}))

vi.mock("@/services/tldw", () => ({
  tldwModels: {
    getEmbeddingModels: vi.fn().mockResolvedValue([]),
    getProviderDisplayName: vi.fn().mockReturnValue("Provider")
  }
}))

vi.mock("@/services/tldw/quick-ingest-batch", () => ({
  startQuickIngestSession: (...args: unknown[]) => mocks.startQuickIngestSession(...args),
  cancelQuickIngestSession: (...args: unknown[]) => mocks.cancelQuickIngestSession(...args),
  submitQuickIngestBatch: (...args: unknown[]) => mocks.submitQuickIngestBatch(...args)
}))

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: vi.fn().mockResolvedValue([]),
  getEmbeddingModels: vi.fn().mockResolvedValue([]),
  defaultEmbeddingModelForRag: vi.fn().mockReturnValue("")
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    phase: "connected",
    isConnected: true,
    serverUrl: "http://localhost:8000"
  }),
  useConnectionActions: () => ({
    checkOnce: mocks.checkOnce
  })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mocks.confirmDanger
}))

vi.mock("@/components/Common/QuickIngestInspectorDrawer", () => ({
  QuickIngestInspectorDrawer: () => null
}))

vi.mock("@/db/dexie/drafts", () => ({
  DRAFT_STORAGE_CAP_BYTES: 5 * 1024 * 1024,
  storeDraftAsset: vi.fn().mockResolvedValue({ asset: null }),
  upsertContentDraft: vi.fn().mockResolvedValue(undefined),
  upsertDraftBatch: vi.fn().mockResolvedValue(undefined)
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    setSetting: vi.fn()
  }
})

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      onMessage: {
        addListener: (listener: (message: any) => void) => {
          mocks.runtimeListeners.push(listener)
        },
        removeListener: (listener: (message: any) => void) => {
          const idx = mocks.runtimeListeners.indexOf(listener)
          if (idx >= 0) {
            mocks.runtimeListeners.splice(idx, 1)
          }
        }
      },
      getURL: (path: string) => `chrome-extension://test${path}`
    },
    tabs: {
      create: (...args: unknown[]) => mocks.tabsCreate(...args)
    }
  }
}))

import { QuickIngestModal } from "@/components/Common/QuickIngestModal"

const emitRuntimeMessage = (message: any) => {
  for (const listener of [...mocks.runtimeListeners]) {
    listener(message)
  }
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

describe("QuickIngestModal session cancel flow", () => {
  beforeEach(() => {
    mocks.runtimeListeners.splice(0, mocks.runtimeListeners.length)
    mocks.startQuickIngestSession.mockReset()
    mocks.cancelQuickIngestSession.mockReset()
    mocks.submitQuickIngestBatch.mockReset()
    mocks.confirmDanger.mockReset()
    mocks.tabsCreate.mockReset()
    mocks.checkOnce.mockReset()
    mocks.startQuickIngestSession.mockResolvedValue({
      ok: true,
      sessionId: "qi-test-session"
    })
    mocks.cancelQuickIngestSession.mockResolvedValue({ ok: true })
    mocks.confirmDanger.mockResolvedValue(true)
    mocks.storageValues.clear()
    mocks.storageSetters.clear()
    mocks.storageValues.set("quickIngestQueuedRows", [
      {
        id: "row-1",
        url: "https://example.com/article",
        type: "auto"
      }
    ])
    mocks.storageValues.set("quickIngestQueuedFiles", [])
    useQuickIngestStore.setState((prev) => ({
      ...prev,
      queuedCount: 0,
      hadRecentFailure: false,
      lastRunSummary: createInitialQuickIngestLastRunSummary()
    }))
  })

  it("starts with session ack and ignores events from other session ids", async () => {
    const user = userEvent.setup()
    render(<QuickIngestModal open onClose={vi.fn()} />)

    await user.click(screen.getByTestId("quick-ingest-run"))

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    emitRuntimeMessage({
      type: "tldw:quick-ingest/completed",
      payload: {
        sessionId: "qi-other-session",
        results: [
          {
            id: "other-1",
            status: "ok",
            url: "https://other.example",
            type: "document"
          }
        ]
      }
    })
    await waitFor(() => {
      expect(screen.queryByTestId("quick-ingest-complete")).not.toBeInTheDocument()
    })

    emitRuntimeMessage({
      type: "tldw:quick-ingest/completed",
      payload: {
        sessionId: "qi-test-session",
        results: [
          {
            id: "session-result-1",
            status: "ok",
            url: "https://example.com/article",
            type: "document"
          }
        ]
      }
    })

    expect(await screen.findByTestId("quick-ingest-complete")).toBeInTheDocument()
  })

  it("constrains the modal body height so the full options surface remains scrollable", async () => {
    render(<QuickIngestModal open onClose={vi.fn()} />)

    const modalBody = document.querySelector(".ant-modal-body") as HTMLElement | null

    expect(modalBody).not.toBeNull()
    expect(modalBody?.style.maxHeight).toBe("calc(100vh - 160px)")
    expect(modalBody?.style.overflowY).toBe("auto")
  })

  it("requires confirmation before sending cancel and applies immediate cancelled terminal state", async () => {
    const onClose = vi.fn()
    const user = userEvent.setup()
    mocks.storageValues.set("quickIngestQueuedRows", [
      {
        id: "row-1",
        url: "https://example.com/cancel-me",
        type: "auto"
      }
    ])
    render(<QuickIngestModal open onClose={onClose} />)
    await user.click(screen.getByTestId("quick-ingest-run"))

    await waitFor(() => {
      expect(mocks.startQuickIngestSession).toHaveBeenCalledTimes(1)
    })

    mocks.confirmDanger.mockResolvedValueOnce(false)
    await user.click(await screen.findByTestId("quick-ingest-cancel"))
    await waitFor(() => {
      expect(mocks.confirmDanger).toHaveBeenCalledTimes(1)
    })
    expect(mocks.cancelQuickIngestSession).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()

    mocks.confirmDanger.mockResolvedValueOnce(true)
    await user.click(screen.getByTestId("quick-ingest-cancel"))
    await waitFor(() => {
      expect(mocks.cancelQuickIngestSession).toHaveBeenCalledWith({
        sessionId: "qi-test-session",
        reason: "user_cancelled"
      })
    })

    const summary = await screen.findByTestId("quick-ingest-complete")
    expect(summary.textContent?.toLowerCase()).toContain("cancelled")

    emitRuntimeMessage({
      type: "tldw:quick-ingest/completed",
      payload: {
        sessionId: "qi-test-session",
        results: [
          {
            id: "late-success",
            status: "ok",
            url: "https://example.com/cancel-me",
            type: "document"
          }
        ]
      }
    })

    await waitFor(() => {
      expect(summary.textContent?.toLowerCase()).toContain("cancelled")
    })
    expect(onClose).not.toHaveBeenCalled()
  })

  it("keeps cancelled terminal state when direct-session completion arrives after cancel", async () => {
    const user = userEvent.setup()
    const directRun = deferred<{
      ok: boolean
      results: Array<{
        id: string
        status: "ok"
        type: string
        url: string
      }>
    }>()

    mocks.startQuickIngestSession.mockResolvedValueOnce({
      ok: true,
      sessionId: "qi-direct-test"
    })
    mocks.submitQuickIngestBatch.mockReturnValueOnce(directRun.promise)

    render(<QuickIngestModal open onClose={vi.fn()} />)

    await user.click(screen.getByTestId("quick-ingest-run"))
    await waitFor(() => {
      expect(mocks.submitQuickIngestBatch).toHaveBeenCalledTimes(1)
    })
    expect(mocks.submitQuickIngestBatch).toHaveBeenCalledWith(
      expect.objectContaining({
        __quickIngestSessionId: "qi-direct-test",
      })
    )

    await user.click(await screen.findByTestId("quick-ingest-cancel"))
    await waitFor(() => {
      expect(mocks.cancelQuickIngestSession).toHaveBeenCalledWith({
        sessionId: "qi-direct-test",
        reason: "user_cancelled"
      })
    })

    const cancelledSummary = await screen.findByTestId("quick-ingest-complete")
    expect(cancelledSummary.textContent?.toLowerCase()).toContain("cancelled")

    directRun.resolve({
      ok: true,
      results: [
        {
          id: "late-direct-success",
          status: "ok",
          type: "document",
          url: "https://example.com/cancel-me"
        }
      ]
    })
    await new Promise((resolve) => setTimeout(resolve, 50))

    await waitFor(() => {
      expect(
        screen.getByTestId("quick-ingest-complete").textContent?.toLowerCase()
      ).toContain("cancelled")
    })
    expect(useQuickIngestStore.getState().lastRunSummary.status).toBe("cancelled")
  })
})
