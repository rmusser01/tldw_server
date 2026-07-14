import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter, Route, Routes } from "react-router-dom"

const runtimeListeners = new Set<
  (message: {
    from?: string
    type?: string
    text?: string
    payload?: unknown
  }) => void
>()

const mocks = vi.hoisted(() => ({
  ensureSidepanelOpen: vi.fn(),
  sendMessage: vi.fn(),
  notify: vi.fn()
}))

vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (options: unknown) => options
  })
  return {}
})

vi.mock("@/services/background-helpers", () => ({
  ensureSidepanelOpen: (...args: unknown[]) =>
    (mocks.ensureSidepanelOpen as (...args: unknown[]) => unknown)(...args),
  notify: (...args: unknown[]) =>
    (mocks.notify as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      sendMessage: (...args: unknown[]) =>
        (mocks.sendMessage as (...args: unknown[]) => unknown)(...args),
      onMessage: {
        addListener: (listener: (message: unknown) => void) => {
          runtimeListeners.add(listener as never)
        },
        removeListener: (listener: (message: unknown) => void) => {
          runtimeListeners.delete(listener as never)
        }
      }
    },
    i18n: {
      getMessage: (key: string) =>
        ({
          contextSaveToClipper: "Save to Clipper",
          contextSaveToClipperRestrictedPage:
            "This page is restricted, so the clipper cannot capture it."
        } as Record<string, string>)[key] || key
    }
  }
}))

vi.mock("~/hooks/useDarkmode", () => ({
  useDarkMode: () => ({ mode: "light" })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: {
      language: "en",
      resolvedLanguage: "en"
    }
  })
}))

vi.mock("@/components/Common/PageAssistLoader", () => ({
  PageAssistLoader: () =>
    React.createElement("div", { "data-testid": "route-loader" }, "Loading")
}))

vi.mock("@/hooks/useAutoButtonTitles", () => ({
  useAutoButtonTitles: () => {}
}))

vi.mock("@/i18n", () => ({
  ensureI18nNamespaces: vi.fn().mockResolvedValue(undefined)
}))

vi.mock("@/utils/ui-diagnostics", () => ({
  registerUiDiagnostics: vi.fn()
}))

vi.mock("@/store/layout-ui", () => ({
  useLayoutUiStore: (
    selector: (state: { setChatSidebarCollapsed: () => void }) => unknown
  ) => selector({ setChatSidebarCollapsed: () => {} })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: null,
    loading: false
  })
}))

vi.mock("@/config/platform", () => ({
  platformConfig: { target: "browser" }
}))

vi.mock("@/routes/route-capabilities", () => ({
  isRouteEnabledForCapabilities: () => true
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    setSetting: vi.fn().mockResolvedValue(undefined)
  }
})

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    React.createElement("div", { "data-testid": "option-layout" }, children)
  )
}))

import { RouteShell } from "@/routes/app-route"
import { buildClipDraft } from "@/services/web-clipper/draft-builder"
import { CLIPPER_CAPTURE_MESSAGE_TYPE } from "@/services/web-clipper/pending-draft"
import * as backgroundEntry from "@/entries/background"
import {
  buildUploadRuntimeResponse,
  cancelQuickIngestRunInBackground,
  createQuickIngestRunSessionSaver,
  delegatePendingQuickIngestRun,
  launchWebClipperFromContextMenu,
  pollQuickIngestRunInBackground,
  updateQuickIngestRunSessions,
} from "@/entries/background"
import type { RouteDefinition } from "@/routes/route-registry"

const ROUTES: Record<"options" | "sidepanel", RouteDefinition[]> = {
  options: [
    {
      kind: "options",
      path: "/",
      element: React.createElement("div", { "data-testid": "home-route" }, "Home")
    }
  ],
  sidepanel: [
    {
      kind: "sidepanel",
      path: "/chat",
      element: React.createElement(
        "div",
        { "data-testid": "sidepanel-chat" },
        "Chat"
      )
    },
    {
      kind: "sidepanel",
      path: "/clipper",
      element: React.createElement(
        "div",
        { "data-testid": "sidepanel-clipper" },
        "Clipper"
      )
    }
  ]
}

const renderRouteShell = (kind: "options" | "sidepanel", path: string) =>
  render(
    React.createElement(
      MemoryRouter,
      { initialEntries: [path] },
      React.createElement(
        Routes,
        null,
        React.createElement(Route, {
          path: "*",
          element: React.createElement(RouteShell, {
            kind,
            routes: ROUTES[kind]
          })
        })
      )
    )
  )

describe("background upload response metadata", () => {
  it("forwards only sanitized Retry-After timing to upload callers", () => {
    const response = buildUploadRuntimeResponse(
      new Response(null, {
        status: 429,
        headers: {
          "Retry-After": "3",
          "Set-Cookie": "secret=must-not-forward",
        },
      }),
      { detail: "rate limited" },
      "rate limited",
    )

    expect(response).toMatchObject({
      ok: false,
      status: 429,
      headers: { "retry-after": "3" },
      retryAfterMs: 3_000,
    })
    expect(response.headers).not.toHaveProperty("set-cookie")
  })
})

describe("background playlist run delegation", () => {
  it("preserves structured review-required recovery through the runtime adapter", () => {
    const reviewRequired = [
      {
        occurrenceId: "occ-runtime-adapter-review",
        reason: "duplicate_action_required",
        evidence: {
          kind: "library",
          existingMediaId: 42,
          duplicateOfOccurrenceId: null,
        },
        allowedActions: ["skip", "overwrite"],
      },
    ]
    const adapt = (backgroundEntry as any)
      .adaptQuickIngestBatchResultForRuntime

    expect(adapt).toBeTypeOf("function")
    if (typeof adapt !== "function") return
    expect(adapt({ ok: false, results: [], reviewRequired })).toEqual({
      results: [],
      reviewRequired,
    })
  })

  it("passes a pending v2 run to the shared client without legacy classification", async () => {
    const pendingRunRequest = {
      contractVersion: 2 as const,
      inputs: [
        {
          occurrenceId: "occ-playlist-2",
          sourceKind: "playlist_entry" as const,
          sourceUrl: "https://www.youtube.com/watch?v=video-2",
          selected: true,
          duplicatePolicy: "skip" as const,
        },
      ],
    }
    const payload = {
      pendingRunRequest,
      entries: [
        {
          id: "occ-playlist-2",
          url: "https://www.youtube.com/watch?v=video-2",
          type: "auto",
        },
      ],
      files: [],
      storeRemote: true,
      processOnly: false,
    }
    const setRunTracking = vi.fn().mockResolvedValue(undefined)
    const submit = vi.fn(async (received: any) => {
      await received.onTrackingMetadata({
        mode: "extension-runtime",
        runId: "run-playlist-2",
        submissionOccurrenceIds: ["occ-playlist-2"],
      })
      return { ok: true, accepted: true, runId: "run-playlist-2" }
    })
    const isCancelled = vi.fn().mockReturnValue(false)

    const result = await delegatePendingQuickIngestRun(
      payload,
      {
        sessionId: "session-playlist-2",
        isCancelled,
        registerAbortController: vi.fn(),
        setJobIds: vi.fn(),
        setRunTracking,
        emitProgress: vi.fn(),
      },
      submit,
    )

    expect(submit).toHaveBeenCalledTimes(1)
    expect(submit).toHaveBeenCalledWith(
      expect.objectContaining({
        pendingRunRequest,
        __quickIngestSessionId: "session-playlist-2",
        __quickIngestShouldStop: isCancelled,
      }),
    )
    expect(setRunTracking).toHaveBeenCalledWith(
      expect.objectContaining({
        runId: "run-playlist-2",
        submissionOccurrenceIds: ["occ-playlist-2"],
      }),
    )
    expect(result).toEqual({ ok: true, results: [] })
  })

  it("does not let stale terminal cleanup delete a replacement run", () => {
    const newer = {
      version: 1 as const,
      sessionId: "session-reused",
      runId: "run-new",
      occurrenceIds: ["occ-new"],
      jobIdToItemId: {},
      startedAt: 2000,
    }

    expect(
      updateQuickIngestRunSessions(
        [newer],
        null,
        "session-reused",
        "run-old",
      ),
    ).toEqual([newer])
    expect(
      updateQuickIngestRunSessions(
        [newer],
        null,
        "session-reused",
        "run-new",
      ),
    ).toEqual([])
  })

  it("uses direct run polling and run cancellation from the background adapter", async () => {
    const tracking = {
      mode: "extension-runtime" as const,
      runId: "run-adapter",
      submissionOccurrenceIds: ["occ-adapter"],
    }
    const reattach = vi.fn().mockResolvedValue({
      lifecycle: "processing",
      jobs: [],
      errorMessage: null,
    })
    const cancel = vi.fn().mockResolvedValue({ runId: "run-adapter" })

    await pollQuickIngestRunInBackground(
      tracking,
      { transportPreference: "poll" },
      reattach,
    )
    const cancelResponse = await cancelQuickIngestRunInBackground(
      tracking,
      "user_cancelled",
      cancel,
    )

    expect(reattach).toHaveBeenCalledWith(tracking, {
      transportPreference: "poll",
      requestOptions: { preferDirect: true },
    })
    expect(cancel).toHaveBeenCalledWith(
      expect.anything(),
      "run-adapter",
      { reason: "user_cancelled" },
      { preferDirect: true },
    )
    expect(cancelResponse).toEqual({ ok: true })
  })

  it("serializes tracking writes before cancellation cleanup", async () => {
    let stored: unknown[] = []
    let releaseFirstWrite!: () => void
    const firstWriteGate = new Promise<void>((resolve) => {
      releaseFirstWrite = resolve
    })
    const persist = vi
      .fn()
      .mockImplementationOnce(async (records: unknown[]) => {
        await firstWriteGate
        stored = records
      })
      .mockImplementation(async (records: unknown[]) => {
        stored = records
      })
    const save = createQuickIngestRunSessionSaver(
      async () => stored,
      persist,
    )
    const record = {
      version: 1 as const,
      sessionId: "session-serialized",
      runId: "run-serialized",
      occurrenceIds: ["occ-serialized"],
      jobIdToItemId: {},
      startedAt: 2001,
    }

    const trackingWrite = save(record)
    await vi.waitFor(() => {
      expect(persist).toHaveBeenCalledTimes(1)
    })
    const cleanupWrite = save(
      null,
      "session-serialized",
      "run-serialized",
    )

    expect(persist).toHaveBeenCalledTimes(1)
    releaseFirstWrite()
    await Promise.all([trackingWrite, cleanupWrite])

    expect(stored).toEqual([])
    expect(persist).toHaveBeenCalledTimes(2)
  })

  it("does not let a stale replay acknowledgement delete a newer run generation", async () => {
    let stored: unknown[] = [
      {
        version: 1,
        kind: "terminal",
        sessionId: "session-generation-race",
        runId: "run-generation-race",
        generation: "generation-new",
        requestFingerprint: "request-new",
        expiresAt: Date.now() + 60_000,
        event: {
          type: "tldw:quick-ingest/completed",
          payload: {
            sessionId: "session-generation-race",
            runId: "run-generation-race",
            results: [],
          },
        },
      },
    ]
    const save = createQuickIngestRunSessionSaver(
      async () => stored,
      async (records) => {
        stored = records
      }
    )

    await (save as any)(
      null,
      "session-generation-race",
      "run-generation-race",
      "generation-old"
    )

    expect(stored).toEqual([
      expect.objectContaining({
        sessionId: "session-generation-race",
        runId: "run-generation-race",
        generation: "generation-new",
      }),
    ])
  })

  it("makes a generation-mismatched conditional replacement a true no-op", () => {
    const current = {
      version: 1,
      kind: "terminal",
      sessionId: "session-cas-mismatch",
      runId: "run-cas-mismatch",
      generation: "generation-current",
      requestFingerprint: "legacy-current",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/completed",
        payload: {
          sessionId: "session-cas-mismatch",
          runId: "run-cas-mismatch",
          results: [],
        },
      },
    }
    const stale = {
      ...current,
      generation: "generation-stale",
      requestFingerprint: "legacy-stale",
    }

    expect(
      updateQuickIngestRunSessions(
        [current],
        stale as any,
        "session-cas-mismatch",
        "run-cas-mismatch",
        "generation-stale",
      ),
    ).toEqual([current])
  })

  it("does not let a same-generation terminal writer replace an existing tombstone", () => {
    const current = {
      version: 1,
      kind: "terminal",
      sessionId: "session-terminal-winner",
      runId: "run-terminal-winner",
      generation: "generation-terminal-winner",
      attemptToken: "attempt-terminal-winner",
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/completed",
        payload: {
          sessionId: "session-terminal-winner",
          runId: "run-terminal-winner",
          results: [],
        },
      },
    }
    const staleReplacement = {
      ...current,
      expiresAt: current.expiresAt + 1_000,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: current.sessionId,
          runId: current.runId,
          results: [],
          error: "stale terminal writer",
        },
      },
    }

    expect(
      updateQuickIngestRunSessions(
        [current],
        staleReplacement as any,
        current.sessionId,
        current.runId,
        current.generation,
      ),
    ).toEqual([current])
  })

  it("replaces a same-generation start marker instead of appending a duplicate terminal record", () => {
    const marker = {
      version: 1,
      kind: "start",
      sessionId: "session-marker-terminal-cas",
      generation: "generation-marker-terminal-cas",
      attemptToken: "attempt-marker-terminal-cas",
      occurrenceIds: ["occ-marker-terminal-cas"],
      startedAt: 1_000,
    }
    const terminal = {
      version: 1,
      kind: "terminal",
      sessionId: marker.sessionId,
      runId: "run-marker-terminal-cas",
      generation: marker.generation,
      attemptToken: marker.attemptToken,
      expiresAt: Date.now() + 60_000,
      event: {
        type: "tldw:quick-ingest/completed",
        payload: {
          sessionId: marker.sessionId,
          runId: "run-marker-terminal-cas",
          results: [],
        },
      },
    }

    expect(
      updateQuickIngestRunSessions(
        [marker],
        terminal as any,
        marker.sessionId,
        terminal.runId,
        marker.generation,
      ),
    ).toEqual([terminal])
  })

  it("replaces a same-generation start marker with the recovered active run", () => {
    const marker = {
      version: 1,
      kind: "start",
      sessionId: "session-marker-active-cas",
      generation: "generation-marker-active-cas",
      attemptToken: "attempt-marker-active-cas",
      occurrenceIds: ["occ-marker-active-cas"],
      startedAt: 1_000,
    }
    const active = {
      version: 1,
      kind: "run",
      sessionId: marker.sessionId,
      runId: "run-marker-active-cas",
      generation: marker.generation,
      attemptToken: marker.attemptToken,
      submissionState: "run_created",
      occurrenceIds: marker.occurrenceIds,
      jobIdToItemId: {},
      startedAt: 1_001,
    }

    expect(
      updateQuickIngestRunSessions(
        [marker],
        active as any,
        marker.sessionId,
        undefined,
        marker.generation,
      ),
    ).toEqual([active])
  })

  it("bounds storage to 64 unique sessions without evicting active cleanup authority", async () => {
    const active = Array.from({ length: 64 }, (_, index) => ({
      version: 1,
      kind: "start",
      sessionId: `session-active-cap-${index}`,
      generation: `generation-active-cap-${index}`,
      attemptToken: `attempt-active-cap-${index}`,
      occurrenceIds: [`occ-active-cap-${index}`],
      startedAt: index,
    }))
    const persist = vi.fn()
    const save = createQuickIngestRunSessionSaver(
      async () => active,
      persist,
    )

    const applied = await save({
      version: 1,
      kind: "start",
      sessionId: "session-active-cap-overflow",
      generation: "generation-active-cap-overflow",
      attemptToken: "attempt-active-cap-overflow",
      occurrenceIds: ["occ-active-cap-overflow"],
      startedAt: 65,
    } as any)

    expect(applied).toBe(false)
    expect(persist).not.toHaveBeenCalled()
    expect(active).toHaveLength(64)
  })

  it("evicts the oldest terminal tombstones deterministically for record and aggregate byte caps", () => {
    const makeTerminal = (index: number, errorSize = 16) => ({
      version: 1,
      kind: "terminal",
      sessionId: `session-terminal-cap-${index}`,
      runId: `run-terminal-cap-${index}`,
      generation: `generation-terminal-cap-${index}`,
      attemptToken: `attempt-terminal-cap-${index}`,
      expiresAt: 10_000 + index,
      event: {
        type: "tldw:quick-ingest/failed",
        payload: {
          sessionId: `session-terminal-cap-${index}`,
          runId: `run-terminal-cap-${index}`,
          results: Array.from({ length: 240 }, (_, resultIndex) => ({
            id: `occ-terminal-cap-${index}-${resultIndex}`,
            status: "error",
            error: "x".repeat(errorSize),
          })),
        },
      },
    })
    const countBounded = Array.from({ length: 64 }, (_, index) =>
      makeTerminal(index),
    )
    const countResult = updateQuickIngestRunSessions(
      countBounded,
      makeTerminal(64) as any,
    ) as Array<any>

    expect(countResult).toHaveLength(64)
    expect(countResult.some((record) => record.sessionId === "session-terminal-cap-0")).toBe(false)
    expect(countResult.some((record) => record.sessionId === "session-terminal-cap-64")).toBe(true)

    const byteBounded = Array.from({ length: 5 }, (_, index) =>
      makeTerminal(100 + index, 1_900),
    )
    const byteResult = updateQuickIngestRunSessions(
      byteBounded,
      makeTerminal(105, 1_900) as any,
    ) as Array<any>
    const terminalBytes = byteResult
      .filter((record) => record.kind === "terminal")
      .reduce((total, record) => total + new TextEncoder().encode(JSON.stringify(record)).byteLength, 0)

    expect(terminalBytes).toBeLessThanOrEqual(2 * 1_024 * 1_024)
    expect(byteResult.some((record) => record.sessionId === "session-terminal-cap-100")).toBe(false)
  })

  it("evicts review tombstones before active recovery authority at the session cap", () => {
    const active = Array.from({ length: 63 }, (_, index) => ({
      version: 1,
      kind: "start",
      sessionId: `session-review-cap-active-${index}`,
      generation: `generation-review-cap-active-${index}`,
      attemptToken: `attempt-review-cap-active-${index}`,
      occurrenceIds: [`occ-review-cap-active-${index}`],
      startedAt: index,
    }))
    const review = {
      version: 1,
      kind: "review",
      sessionId: "session-review-cap-evictable",
      generation: "generation-review-cap-evictable",
      attemptToken: "attempt-review-cap-evictable",
      expiresAt: 1,
      event: {
        type: "tldw:quick-ingest/review-required",
        payload: {
          sessionId: "session-review-cap-evictable",
          reviewRequired: [],
        },
      },
    }
    const incoming = {
      version: 1,
      kind: "start",
      sessionId: "session-review-cap-active-63",
      generation: "generation-review-cap-active-63",
      attemptToken: "attempt-review-cap-active-63",
      occurrenceIds: ["occ-review-cap-active-63"],
      startedAt: 63,
    }

    const result = updateQuickIngestRunSessions(
      [...active, review],
      incoming as any,
    ) as Array<any>

    expect(result).toHaveLength(64)
    expect(result.some((record) => record.sessionId === review.sessionId)).toBe(false)
    expect(result.some((record) => record.sessionId === incoming.sessionId)).toBe(true)
    expect(result.filter((record) => record.kind !== "review")).toHaveLength(64)
  })

  it("counts review tombstones toward the aggregate replay byte cap", () => {
    const makeReview = (index: number) => ({
      version: 1,
      kind: "review",
      sessionId: `session-review-byte-cap-${index}`,
      generation: `generation-review-byte-cap-${index}`,
      attemptToken: `attempt-review-byte-cap-${index}`,
      expiresAt: 10_000 + index,
      event: {
        type: "tldw:quick-ingest/review-required",
        payload: {
          sessionId: `session-review-byte-cap-${index}`,
          reviewRequired: Array.from({ length: 500 }, (_, itemIndex) => ({
            occurrenceId: `occ-${index}-${itemIndex}-${"x".repeat(220)}`,
            reason: "duplicate_action_required",
            evidence: {
              kind: "library",
              existingMediaId: itemIndex + 1,
              duplicateOfOccurrenceId: null,
            },
            allowedActions: ["skip", "overwrite"],
          })),
        },
      },
    })
    const existing = Array.from({ length: 11 }, (_, index) => makeReview(index))
    const incoming = makeReview(11)

    const result = updateQuickIngestRunSessions(
      existing,
      incoming as any,
    ) as Array<any>
    const replayBytes = result
      .filter((record) => record.kind === "terminal" || record.kind === "review")
      .reduce(
        (total, record) =>
          total + new TextEncoder().encode(JSON.stringify(record)).byteLength,
        0,
      )

    expect(replayBytes).toBeLessThanOrEqual(2 * 1_024 * 1_024)
    expect(result.some((record) => record.sessionId === "session-review-byte-cap-0")).toBe(false)
    expect(result.some((record) => record.sessionId === incoming.sessionId)).toBe(true)
  })

  it("preserves structured review-required recovery through the background delegate", async () => {
    const reviewRequired = [
      {
        occurrenceId: "occ-background-review",
        reason: "duplicate_action_required",
        evidence: {
          kind: "library",
          existingMediaId: 42,
          duplicateOfOccurrenceId: null,
        },
        allowedActions: ["skip", "overwrite"],
      },
    ]
    const result = await delegatePendingQuickIngestRun(
      { pendingRunRequest: { inputs: [] } },
      {
        sessionId: "session-background-review",
        isCancelled: vi.fn().mockReturnValue(false),
        registerAbortController: vi.fn(),
        setJobIds: vi.fn(),
        setRunTracking: vi.fn(),
        emitProgress: vi.fn(),
      },
      vi.fn().mockResolvedValue({
        ok: false,
        accepted: false,
        error: "Review the updated duplicate choices.",
        results: [],
        reviewRequired,
      }) as any,
    )

    expect(result).toEqual({ ok: false, results: [], reviewRequired })
  })
})

describe("web clipper background launcher", () => {
  beforeEach(() => {
    runtimeListeners.clear()
    window.sessionStorage.clear()
    vi.clearAllMocks()
    vi.useRealTimers()
    mocks.sendMessage.mockResolvedValue({ handled: true })

    Object.defineProperty(globalThis, "browser", {
      configurable: true,
      value: {
        runtime: {
          onMessage: {
            addListener: vi.fn((listener: (message: unknown) => void) => {
              runtimeListeners.add(listener)
            }),
            removeListener: vi.fn((listener: (message: unknown) => void) => {
              runtimeListeners.delete(listener)
            })
          }
        }
      }
    })
  })

  afterEach(() => {
    Reflect.deleteProperty(globalThis, "browser")
    Reflect.deleteProperty(globalThis, "defineBackground")
  })

  it("routes the clipper handoff into the dedicated clipper sidepanel route", async () => {
    renderRouteShell("sidepanel", "/chat")

    await waitFor(() => {
      expect(screen.getByTestId("sidepanel-chat")).toBeVisible()
    })

    const draft = buildClipDraft({
      requestedType: "article",
      pageUrl: "https://example.com/story",
      pageTitle: "Story",
      extracted: {
        articleText: "",
        fullPageText: "Fallback body"
      }
    })

    act(() => {
      for (const listener of runtimeListeners) {
        listener({
          from: "background",
          type: CLIPPER_CAPTURE_MESSAGE_TYPE,
          payload: draft
        })
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId("sidepanel-clipper")).toBeVisible()
    })

    const stored = window.sessionStorage.getItem(
      "tldw:web-clipper:pendingDraft"
    )
    expect(stored).not.toBeNull()
    expect(JSON.parse(String(stored))).toMatchObject({
      clipType: "article",
      pageUrl: "https://example.com/story",
      pageTitle: "Story",
      captureMetadata: {
        fallbackPath: ["article", "full_page"]
      }
    })
  })

  it("opens the sidepanel and sends a clipper message instead of the notes flow", async () => {
    await launchWebClipperFromContextMenu(
      {
        pageUrl: "https://example.com/story",
        selectionText: "Selected excerpt"
      },
      { id: 8, url: "https://example.com/story", title: "Story" }
    )

    expect(mocks.ensureSidepanelOpen).toHaveBeenCalledWith(8)
    expect(mocks.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        from: "background",
        type: CLIPPER_CAPTURE_MESSAGE_TYPE,
        payload: expect.objectContaining({
          clipType: "selection",
          captureMetadata: expect.objectContaining({
            fallbackPath: ["selection"]
          })
        })
      })
    )
    expect(mocks.sendMessage).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "save-to-notes" })
    )
  })

  it("retries clipper delivery until the sidepanel listener is ready", async () => {
    vi.useFakeTimers()
    mocks.sendMessage
      .mockRejectedValueOnce(
        new Error("Could not establish connection. Receiving end does not exist.")
      )
      .mockRejectedValueOnce(
        new Error("Could not establish connection. Receiving end does not exist.")
      )
      .mockResolvedValueOnce({ handled: true })

    const handoffPromise = launchWebClipperFromContextMenu(
      {
        pageUrl: "https://example.com/story",
        selectionText: "Selected excerpt"
      },
      { id: 9, url: "https://example.com/story", title: "Story" }
    )

    await Promise.resolve()

    expect(mocks.sendMessage).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(500)
    expect(mocks.sendMessage).toHaveBeenCalledTimes(2)

    await vi.advanceTimersByTimeAsync(500)
    await handoffPromise

    expect(mocks.sendMessage).toHaveBeenCalledTimes(3)
    expect(mocks.notify).not.toHaveBeenCalled()
  })

  it("fails restricted pages with a user-visible explanation instead of sending a silent message", async () => {
    await launchWebClipperFromContextMenu(
      {
        pageUrl: "chrome://extensions"
      },
      { id: 14, url: "chrome://extensions", title: "Extensions" }
    )

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.notify).toHaveBeenCalledWith(
      expect.stringContaining("Clipper"),
      expect.stringContaining("restricted")
    )
  })
})
