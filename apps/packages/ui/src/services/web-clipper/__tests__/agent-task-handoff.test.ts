import { afterEach, describe, expect, it, vi } from "vitest"
import {
  WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY,
  clearPendingWebClipAgentTaskRequest,
  readPendingWebClipAgentTaskRequest,
  writePendingWebClipAgentTaskRequest,
  type PendingWebClipAgentTaskRequest
} from "@/services/web-clipper/agent-task-handoff"

const createRequest = (
  overrides: Partial<PendingWebClipAgentTaskRequest> = {}
): PendingWebClipAgentTaskRequest => ({
  id: "handoff-1",
  clipId: "clip-123",
  noteId: "note-123",
  workspaceId: "workspace-alpha",
  workspaceNoteId: 42,
  pageUrl: "https://example.com/story",
  pageTitle: "Example Story",
  extractPreview: "Alpha body copy",
  hasScreenshot: false,
  createdAt: new Date().toISOString(),
  ...overrides
})

describe("web clipper agent-task handoff storage", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    window.localStorage.clear()
    window.sessionStorage.clear()
  })

  it("falls back when chrome storage callbacks report runtime errors", async () => {
    const runtimeState: {
      lastError: { message: string } | null
    } = { lastError: null }

    vi.stubGlobal("chrome", {
      runtime: {
        get lastError() {
          return runtimeState.lastError
        }
      },
      storage: {
        session: {
          set: vi.fn((_items: Record<string, unknown>, callback?: () => void) => {
            runtimeState.lastError = { message: "quota exceeded" }
            callback?.()
            runtimeState.lastError = null
          }),
          get: vi.fn(
            (
              _key: string,
              callback?: (items: Record<string, unknown>) => void
            ) => {
              runtimeState.lastError = { message: "storage unavailable" }
              callback?.({})
              runtimeState.lastError = null
            }
          ),
          remove: vi.fn((_key: string, callback?: () => void) => {
            callback?.()
          })
        }
      }
    })

    await writePendingWebClipAgentTaskRequest(createRequest())

    const fallbackValue = window.sessionStorage.getItem(
      WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY
    )
    expect(fallbackValue).not.toBeNull()
    expect(
      window.localStorage.getItem(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY)
    ).toBeNull()

    await expect(readPendingWebClipAgentTaskRequest()).resolves.toMatchObject({
      clipId: "clip-123",
      workspaceId: "workspace-alpha",
      workspaceNoteId: 42
    })
  })

  it("suppresses stale extension handoffs when chrome storage remove fails", async () => {
    const runtimeState: {
      lastError: { message: string } | null
    } = { lastError: null }
    const storageState = new Map<string, unknown>()

    vi.stubGlobal("chrome", {
      runtime: {
        get lastError() {
          return runtimeState.lastError
        }
      },
      storage: {
        session: {
          set: vi.fn((items: Record<string, unknown>, callback?: () => void) => {
            for (const [key, value] of Object.entries(items)) {
              storageState.set(key, value)
            }
            callback?.()
          }),
          get: vi.fn(
            (
              key: string,
              callback?: (items: Record<string, unknown>) => void
            ) => {
              callback?.(
                storageState.has(key) ? { [key]: storageState.get(key) } : {}
              )
            }
          ),
          remove: vi.fn((_key: string, callback?: () => void) => {
            runtimeState.lastError = { message: "remove failed" }
            callback?.()
            runtimeState.lastError = null
          })
        }
      }
    })

    await writePendingWebClipAgentTaskRequest(createRequest())
    await expect(readPendingWebClipAgentTaskRequest()).resolves.toMatchObject({
      clipId: "clip-123"
    })

    await clearPendingWebClipAgentTaskRequest()

    expect(storageState.get(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY)).toBeNull()
    await expect(readPendingWebClipAgentTaskRequest()).resolves.toBeNull()
  })

  it("does not read browser fallback when extension storage has a tombstone", async () => {
    const runtimeState: {
      lastError: { message: string } | null
    } = { lastError: null }
    const storageState = new Map<string, unknown>([
      [WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY, null]
    ])

    vi.stubGlobal("chrome", {
      runtime: {
        get lastError() {
          return runtimeState.lastError
        }
      },
      storage: {
        session: {
          set: vi.fn((_items: Record<string, unknown>, callback?: () => void) => {
            runtimeState.lastError = { message: "set failed" }
            callback?.()
            runtimeState.lastError = null
          }),
          get: vi.fn(
            (
              key: string,
              callback?: (items: Record<string, unknown>) => void
            ) => {
              callback?.(
                storageState.has(key) ? { [key]: storageState.get(key) } : {}
              )
            }
          ),
          remove: vi.fn((_key: string, callback?: () => void) => {
            callback?.()
          })
        }
      }
    })

    window.sessionStorage.setItem(
      WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY,
      JSON.stringify(createRequest({ id: "stale-fallback" }))
    )
    await writePendingWebClipAgentTaskRequest(createRequest())

    await expect(readPendingWebClipAgentTaskRequest()).resolves.toBeNull()
  })

  it("expires stale browser fallback handoffs", async () => {
    const staleRequest = createRequest({
      createdAt: new Date(Date.now() - 11 * 60 * 1000).toISOString()
    })
    window.sessionStorage.setItem(
      WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY,
      JSON.stringify(staleRequest)
    )
    window.localStorage.setItem(
      WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY,
      JSON.stringify(staleRequest)
    )

    await expect(readPendingWebClipAgentTaskRequest()).resolves.toBeNull()
    expect(
      window.sessionStorage.getItem(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY)
    ).toBeNull()
    expect(
      window.localStorage.getItem(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY)
    ).toBeNull()
  })
})
