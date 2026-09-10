import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY,
  buildPendingWebClipAgentTaskRequest,
  clearPendingWebClipAgentTaskRequest,
  readPendingWebClipAgentTaskRequest,
  writePendingWebClipAgentTaskRequest,
  type PendingWebClipAgentTaskRequest
} from "@/services/web-clipper/agent-task-handoff"

const FIXED_NOW = new Date("2026-01-01T00:00:00.000Z")
const STALE_CREATED_AT = "2025-12-31T23:49:00.000Z"

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
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(FIXED_NOW)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    window.localStorage.clear()
    window.sessionStorage.clear()
  })

  it("builds handoffs when crypto randomUUID is unavailable", () => {
    vi.stubGlobal("crypto", {})

    const request = buildPendingWebClipAgentTaskRequest({
      draft: {
        clipId: "clip-123",
        requestedType: "article",
        clipType: "article",
        pageUrl: "https://example.com/story",
        pageTitle: "Example Story",
        visibleBody: "Visible body",
        selectionText: "Selected text",
        captureMetadata: {
          clipType: "article",
          actualType: "article",
          fallbackPath: []
        },
        capturedAt: FIXED_NOW.toISOString()
      },
      response: {
        clip_id: "clip-123",
        status: "saved",
        note: { id: "note-123", title: "Example Story", version: 1 },
        workspace_placement: {
          workspace_id: "workspace-alpha",
          workspace_note_id: 42,
          source_note_id: "note-123"
        },
        attachments: [],
        warnings: [],
        note_id: "note-123",
        workspace_placement_saved: true,
        workspace_placement_count: 1
      }
    })

    expect(request?.id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
    )
  })

  it("fills missing stored handoff ids when crypto randomUUID is unavailable", async () => {
    vi.stubGlobal("crypto", {})
    const { id: _id, ...requestWithoutId } = createRequest()
    window.sessionStorage.setItem(
      WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY,
      JSON.stringify(requestWithoutId)
    )

    await expect(readPendingWebClipAgentTaskRequest()).resolves.toMatchObject({
      id: expect.stringMatching(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
      ),
      clipId: "clip-123"
    })
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

  it.each(["callback", "promise", "both"] as const)(
    "preserves storage receivers with %s APIs",
    async (mode) => {
      const request = createRequest()
      const area = {
        entries: new Map<string, unknown>(),
        get(key: string, callback?: (items: Record<string, unknown>) => void) {
          const result = { [key]: this.entries.get(key) }
          if (mode !== "promise") callback?.(result)
          if (mode !== "callback") return Promise.resolve(result)
        },
        set(items: Record<string, unknown>, callback?: () => void) {
          for (const [key, value] of Object.entries(items)) {
            this.entries.set(key, value)
          }
          if (mode !== "promise") callback?.()
          if (mode !== "callback") return Promise.resolve()
        },
        remove(key: string, callback?: () => void) {
          this.entries.delete(key)
          if (mode !== "promise") callback?.()
          if (mode !== "callback") return Promise.resolve()
        }
      }
      vi.stubGlobal("chrome", { storage: { session: area } })

      await writePendingWebClipAgentTaskRequest(request)
      expect(area.entries.get(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY)).toEqual(request)
      expect(window.sessionStorage.length).toBe(0)
      await expect(readPendingWebClipAgentTaskRequest()).resolves.toEqual(request)
      await clearPendingWebClipAgentTaskRequest()
      expect(area.entries.size).toBe(0)
      await expect(readPendingWebClipAgentTaskRequest()).resolves.toBeNull()
    }
  )

  it("uses browser fallback when extension storage methods are missing", async () => {
    vi.stubGlobal("chrome", { storage: { session: {} } })
    const request = createRequest()

    await writePendingWebClipAgentTaskRequest(request)
    expect(window.sessionStorage.getItem(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY)).toBe(JSON.stringify(request))
    await expect(readPendingWebClipAgentTaskRequest()).resolves.toEqual(request)
    await clearPendingWebClipAgentTaskRequest()
    await expect(readPendingWebClipAgentTaskRequest()).resolves.toBeNull()
  })

  it("keeps the callback result when a local storage API also resolves a Promise", async () => {
    const request = createRequest()
    vi.stubGlobal("chrome", {
      storage: {
        local: {
          get(key: string, callback?: (items: Record<string, unknown>) => void) {
            callback?.({ [key]: request })
            return Promise.resolve({ [key]: createRequest({ id: "later-result" }) })
          }
        }
      }
    })

    await expect(readPendingWebClipAgentTaskRequest()).resolves.toEqual(request)
  })

  it.each(["throw", "reject"] as const)(
    "uses browser fallback when extension storage methods %s",
    async (mode) => {
      const fail = () => {
        const error = new Error("storage unavailable")
        if (mode === "throw") throw error
        return Promise.reject(error)
      }
      vi.stubGlobal("chrome", {
        storage: { session: { get: fail, set: fail, remove: fail } }
      })
      const request = createRequest()
      const warning = vi.spyOn(console, "warn").mockImplementation(() => {})
      try {
        await writePendingWebClipAgentTaskRequest(request)
        await expect(readPendingWebClipAgentTaskRequest()).resolves.toEqual(request)
        await clearPendingWebClipAgentTaskRequest()
        expect(window.sessionStorage.getItem(WEB_CLIPPER_PENDING_AGENT_TASK_STORAGE_KEY)).toBeNull()
      } finally {
        warning.mockRestore()
      }
    }
  )

  it("expires stale browser fallback handoffs", async () => {
    const staleRequest = createRequest({
      createdAt: STALE_CREATED_AT
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
