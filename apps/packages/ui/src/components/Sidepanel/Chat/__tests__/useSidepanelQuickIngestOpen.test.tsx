// @vitest-environment jsdom
import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { DEFAULT_PRESETS } from "@/components/Common/QuickIngest/presets"
import { useComposerEvents } from "@/hooks/useComposerEvents"
import {
  createEmptyQuickIngestSession,
  useQuickIngestSessionStore,
} from "@/store/quick-ingest-session"
import {
  consumePendingQuickIngestOpen,
  requestQuickIngestOpen,
  type QuickIngestOpenDetail,
} from "@/utils/quick-ingest-open"
import { useSidepanelQuickIngestOpen } from "../useSidepanelQuickIngestOpen"

describe("useSidepanelQuickIngestOpen", () => {
  beforeEach(() => {
    consumePendingQuickIngestOpen()
    useQuickIngestSessionStore.setState({
      session: null,
      triggerSummary: { count: 0, label: null, hadFailure: false },
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    consumePendingQuickIngestOpen()
  })

  it("waits for hydration and replays the exact open once without replacing the persisted Review draft", async () => {
    const persistedDraft = {
      ...createEmptyQuickIngestSession(),
      id: "persisted-sidepanel-review",
      visibility: "hidden" as const,
      currentStep: 3 as const,
      selectedPreset: "deep" as const,
      customBasePreset: "deep" as const,
      presetConfig: DEFAULT_PRESETS.deep,
      queueItems: [
        {
          id: "persisted-playlist-item",
          kind: "url",
          url: "https://www.youtube.com/watch?v=existing&list=PLpersisted",
          detectedType: "video" as const,
          icon: "Youtube",
          fileSize: 0,
          validation: { valid: true },
        },
      ],
      badge: { queueCount: 1, hasRecentFailure: false },
      createdAt: 1,
      updatedAt: 2,
    }
    const detail: QuickIngestOpenDetail = {
      source: "extension_active_tab",
      url: "https://www.youtube.com/playlist?list=PLincoming",
      sourceKind: "youtube_playlist",
      action: "playlist_preflight",
    }
    const persistApi = useQuickIngestSessionStore.persist
    let hydrated = false
    const finishHydrationListeners = new Set<(state: unknown) => void>()
    vi.spyOn(persistApi, "hasHydrated").mockImplementation(() => hydrated)
    vi.spyOn(persistApi, "rehydrate").mockImplementation(async () => undefined)
    vi.spyOn(persistApi, "onHydrate").mockImplementation(() => () => undefined)
    vi.spyOn(persistApi, "onFinishHydration").mockImplementation((listener) => {
      finishHydrationListeners.add(listener as (state: unknown) => void)
      return () => {
        finishHydrationListeners.delete(listener as (state: unknown) => void)
      }
    })

    const setIngestOpen = vi.fn()
    const setAutoProcessQueued = vi.fn()
    const focus = vi.fn()
    const focusTriggerRef = {
      current: { focus },
    } as unknown as React.RefObject<HTMLElement>

    renderHook(() => {
      const handleOpenQuickIngest = useSidepanelQuickIngestOpen({
        focusTriggerRef,
        setAutoProcessQueued,
        setIngestOpen,
      })
      useComposerEvents({ onOpenQuickIngest: handleOpenQuickIngest })
    })

    act(() => {
      requestQuickIngestOpen(detail, {
        autoProcessQueued: true,
        focusTrigger: false,
      })
    })

    expect(useQuickIngestSessionStore.getState().session).toBeNull()
    expect(setIngestOpen).not.toHaveBeenCalled()
    expect(persistApi.rehydrate).toHaveBeenCalledTimes(1)

    act(() => {
      useQuickIngestSessionStore.setState({ session: persistedDraft })
      hydrated = true
      for (const listener of finishHydrationListeners) {
        listener(useQuickIngestSessionStore.getState())
      }
    })

    await waitFor(() => expect(setIngestOpen).toHaveBeenCalledWith(true))

    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      id: "persisted-sidepanel-review",
      currentStep: 3,
      selectedPreset: "deep",
      customBasePreset: "deep",
      presetConfig: { reviewBeforeStorage: true },
      queueItems: [
        {
          id: "persisted-playlist-item",
          url: "https://www.youtube.com/watch?v=existing&list=PLpersisted",
        },
      ],
      openDetail: detail,
    })
    expect(setAutoProcessQueued).toHaveBeenCalledTimes(1)
    expect(setAutoProcessQueued).toHaveBeenCalledWith(true)
    expect(setIngestOpen).toHaveBeenCalledTimes(1)
    expect(focus).not.toHaveBeenCalled()
    expect(consumePendingQuickIngestOpen()).toBeNull()

    act(() => {
      for (const listener of finishHydrationListeners) {
        listener(useQuickIngestSessionStore.getState())
      }
    })
    expect(setAutoProcessQueued).toHaveBeenCalledTimes(1)
    expect(setIngestOpen).toHaveBeenCalledTimes(1)
  })

  it("does not recreate an open that another modal host consumes during hydration", async () => {
    const persistedDraft = {
      ...createEmptyQuickIngestSession(),
      id: "persisted-draft-claimed-by-layout",
      visibility: "hidden" as const,
      currentStep: 3 as const,
    }
    const detail: QuickIngestOpenDetail = {
      source: "extension_active_tab",
      url: "https://www.youtube.com/playlist?list=PLclaimed",
      sourceKind: "youtube_playlist",
      action: "playlist_preflight",
    }
    const persistApi = useQuickIngestSessionStore.persist
    let hydrated = false
    const finishHydrationListeners = new Set<(state: unknown) => void>()
    vi.spyOn(persistApi, "hasHydrated").mockImplementation(() => hydrated)
    vi.spyOn(persistApi, "rehydrate").mockImplementation(async () => undefined)
    vi.spyOn(persistApi, "onHydrate").mockImplementation(() => () => undefined)
    vi.spyOn(persistApi, "onFinishHydration").mockImplementation((listener) => {
      finishHydrationListeners.add(listener as (state: unknown) => void)
      return () => {
        finishHydrationListeners.delete(listener as (state: unknown) => void)
      }
    })

    const setIngestOpen = vi.fn()
    const setAutoProcessQueued = vi.fn()
    renderHook(() => {
      const handleOpenQuickIngest = useSidepanelQuickIngestOpen({
        focusTriggerRef: { current: null },
        setAutoProcessQueued,
        setIngestOpen,
      })
      useComposerEvents({ onOpenQuickIngest: handleOpenQuickIngest })
    })

    act(() => {
      requestQuickIngestOpen(detail)
    })
    expect(consumePendingQuickIngestOpen("normal")?.detail).toBe(detail)

    act(() => {
      useQuickIngestSessionStore.setState({ session: persistedDraft })
      hydrated = true
      for (const listener of finishHydrationListeners) {
        listener(useQuickIngestSessionStore.getState())
      }
    })
    await act(async () => {
      await Promise.resolve()
    })

    expect(useQuickIngestSessionStore.getState().session).toMatchObject({
      id: "persisted-draft-claimed-by-layout",
      currentStep: 3,
      openDetail: null,
    })
    expect(setAutoProcessQueued).not.toHaveBeenCalled()
    expect(setIngestOpen).not.toHaveBeenCalled()
    expect(consumePendingQuickIngestOpen()).toBeNull()
  })

  it("keeps the pending intent inert when rehydration rejects after unmount", async () => {
    const detail: QuickIngestOpenDetail = {
      source: "extension_active_tab",
      url: "https://www.youtube.com/playlist?list=PLunmounted",
      sourceKind: "youtube_playlist",
      action: "playlist_preflight",
    }
    const persistApi = useQuickIngestSessionStore.persist
    let rejectRehydrate!: (reason: unknown) => void
    const rehydrate = new Promise<void>((_resolve, reject) => {
      rejectRehydrate = reject
    })
    vi.spyOn(persistApi, "hasHydrated").mockReturnValue(false)
    vi.spyOn(persistApi, "rehydrate").mockReturnValue(rehydrate)
    vi.spyOn(persistApi, "onHydrate").mockImplementation(() => () => undefined)
    vi.spyOn(persistApi, "onFinishHydration").mockImplementation(
      () => () => undefined
    )

    const setIngestOpen = vi.fn()
    const setAutoProcessQueued = vi.fn()
    const { unmount } = renderHook(() => {
      const handleOpenQuickIngest = useSidepanelQuickIngestOpen({
        focusTriggerRef: { current: null },
        setAutoProcessQueued,
        setIngestOpen,
      })
      useComposerEvents({ onOpenQuickIngest: handleOpenQuickIngest })
    })

    act(() => {
      requestQuickIngestOpen(detail)
    })
    unmount()
    await act(async () => {
      rejectRehydrate(new Error("storage unavailable"))
      await rehydrate.catch(() => undefined)
    })

    expect(useQuickIngestSessionStore.getState().session).toBeNull()
    expect(setAutoProcessQueued).not.toHaveBeenCalled()
    expect(setIngestOpen).not.toHaveBeenCalled()
    expect(consumePendingQuickIngestOpen("normal")?.detail).toBe(detail)
  })
})
