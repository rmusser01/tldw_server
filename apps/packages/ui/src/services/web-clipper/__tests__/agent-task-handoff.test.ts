import { afterEach, describe, expect, it, vi } from "vitest"
import {
  WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY,
  clearPendingWebClipAgentTaskRequest,
  readPendingWebClipAgentTaskRequest,
  writePendingWebClipAgentTaskRequest,
  type PendingWebClipAgentTaskRequest
} from "@/services/web-clipper/agent-task-handoff"

const createRequest = (): PendingWebClipAgentTaskRequest => ({
  id: "handoff-1",
  clipId: "clip-123",
  noteId: "note-123",
  workspaceId: "workspace-alpha",
  workspaceNoteId: 42,
  pageUrl: "https://example.com/story",
  pageTitle: "Example Story",
  extractPreview: "Alpha body copy",
  hasScreenshot: false,
  createdAt: "2026-07-05T00:00:00.000Z"
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

    const fallbackValue = window.localStorage.getItem(
      WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY
    )
    expect(fallbackValue).not.toBeNull()

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

  it("allows fallback handoffs after an extension tombstone", async () => {
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

    await writePendingWebClipAgentTaskRequest(createRequest())

    await expect(readPendingWebClipAgentTaskRequest()).resolves.toMatchObject({
      clipId: "clip-123",
      workspaceId: "workspace-alpha"
    })
  })
})
