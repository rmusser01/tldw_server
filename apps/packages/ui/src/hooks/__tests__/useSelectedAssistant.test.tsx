import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { Character } from "@/types/character"

const mocks = vi.hoisted(() => {
  const assistantLocal = new Map<string, unknown>()
  const assistantSync = new Map<string, unknown>()
  const characterLocal = new Map<string, unknown>()
  const characterSync = new Map<string, unknown>()
  const operations: string[] = []
  const assistantSetBarriers: Promise<void>[] = []

  const createStorageMock = (name: string, map: Map<string, unknown>) => ({
    get: vi.fn(async (key: string) => (map.has(key) ? map.get(key) : null)),
    set: vi.fn(async (key: string, value: unknown) => {
      operations.push(`${name}.set:${key}`)
      if (value == null) {
        map.delete(key)
        return
      }
      map.set(key, value)
    }),
    remove: vi.fn(async (key: string) => {
      operations.push(`${name}.remove:${key}`)
      map.delete(key)
    })
  })

  return {
    assistantLocal,
    assistantSync,
    characterLocal,
    characterSync,
    operations,
    assistantSetBarriers,
    isCurrentOwner: () => true,
    parseStoredValue: (value: unknown): Record<string, unknown> | null => {
      if (!value) return null
      if (typeof value === "string") {
        try {
          const parsed = JSON.parse(value)
          return parsed && typeof parsed === "object"
            ? (parsed as Record<string, unknown>)
            : null
        } catch {
          return null
        }
      }
      return typeof value === "object"
        ? (value as Record<string, unknown>)
        : null
    },
    assistantStorage: createStorageMock("assistantLocal", assistantLocal),
    assistantSyncStorage: createStorageMock("assistantSync", assistantSync),
    characterStorage: createStorageMock("characterLocal", characterLocal),
    characterSyncStorage: createStorageMock("characterSync", characterSync)
  }
})

vi.mock("@/hooks/useChatDraftOwner", () => ({
  useChatDraftOwner: () => ({ ownerKey: "account-a", isCurrent: mocks.isCurrentOwner })
}))

vi.mock("@plasmohq/storage/hook", async () => {
  const ReactModule =
    await vi.importActual<typeof import("react")>("react")

  return {
    useStorage: (
      config: string | { key: string; instance?: unknown },
      initialValue: unknown
    ) => {
      const key = typeof config === "string" ? config : config.key
      const instance = typeof config === "string" ? null : config.instance
      const store =
        instance === mocks.assistantStorage
          ? mocks.assistantLocal
          : instance === mocks.assistantSyncStorage
            ? mocks.assistantSync
            : instance === mocks.characterStorage
              ? mocks.characterLocal
              : instance === mocks.characterSyncStorage
                ? mocks.characterSync
                : mocks.assistantLocal

      // These existing controls use flat fixtures; the real-adapter privacy suite
      // separately verifies the account-stamped persistence format.
      const getStoredValue = () => store.has(key)
        ? { ownerKey: "account-a", selection: store.get(key) ?? null }
        : initialValue ?? null

      const [value, setRenderValue] = ReactModule.useState(getStoredValue)

      ReactModule.useEffect(() => {
        setRenderValue(getStoredValue())
      }, [key])

      const setValue = async (next: unknown) => {
        const resolved =
          typeof next === "function"
            ? (next as (prev: unknown) => unknown)(getStoredValue())
            : next
        const selection = (resolved as { selection?: unknown } | null)?.selection ?? null
        mocks.operations.push(
          `useStorage.${key}.set:${selection == null ? "null" : "value"}`
        )
        if (key.startsWith("selectedAssistant:owner:")) {
          const barrier = mocks.assistantSetBarriers.shift()
          if (barrier) await barrier
        }
        if (selection == null) {
          store.delete(key)
          setRenderValue(resolved)
          return
        }
        store.set(key, selection)
        setRenderValue(resolved)
      }

      const setRenderValueWithLog = (next: unknown) => {
        mocks.operations.push(
          `useStorage.${key}.render:${(next as { selection?: unknown } | null)?.selection == null ? "null" : "value"}`
        )
        setRenderValue(next)
      }

      return [
        value,
        setValue,
        { isLoading: false, setRenderValue: setRenderValueWithLog }
      ] as const
    }
  }
})

vi.mock("@/utils/selected-assistant-storage", () => ({
  SELECTED_ASSISTANT_STORAGE_KEY: "selectedAssistant",
  selectedAssistantStorage: mocks.assistantStorage,
  selectedAssistantSyncStorage: mocks.assistantSyncStorage,
  parseSelectedAssistantValue: mocks.parseStoredValue
}))

vi.mock("@/utils/selected-character-storage", () => ({
  SELECTED_CHARACTER_STORAGE_KEY: "selectedCharacter",
  selectedCharacterStorage: mocks.characterStorage,
  selectedCharacterSyncStorage: mocks.characterSyncStorage,
  parseSelectedCharacterValue: mocks.parseStoredValue
}))

import { useSelectedAssistant } from "../useSelectedAssistant"
import { useSelectedCharacter } from "../useSelectedCharacter"
import {
  SELECTED_ASSISTANT_STORAGE_KEY as ASSISTANT_KEY
} from "@/utils/selected-assistant-storage"
import {
  SELECTED_CHARACTER_STORAGE_KEY as CHARACTER_KEY
} from "@/utils/selected-character-storage"

const SELECTED_ASSISTANT_STORAGE_KEY = `${ASSISTANT_KEY}:owner:account-a`
const SELECTED_CHARACTER_STORAGE_KEY = `${CHARACTER_KEY}:owner:account-a`

describe("useSelectedAssistant", () => {
  beforeEach(() => {
    mocks.assistantLocal.clear()
    mocks.assistantSync.clear()
    mocks.characterLocal.clear()
    mocks.characterSync.clear()
    mocks.operations.length = 0
    mocks.assistantSetBarriers.length = 0
    mocks.assistantStorage.get.mockClear()
    mocks.assistantStorage.set.mockClear()
    mocks.assistantStorage.remove.mockClear()
    mocks.assistantSyncStorage.get.mockClear()
    mocks.assistantSyncStorage.set.mockClear()
    mocks.assistantSyncStorage.remove.mockClear()
    mocks.characterStorage.get.mockClear()
    mocks.characterStorage.set.mockClear()
    mocks.characterStorage.remove.mockClear()
    mocks.characterSyncStorage.get.mockClear()
    mocks.characterSyncStorage.set.mockClear()
    mocks.characterSyncStorage.remove.mockClear()
  })

  it("does not adopt an unowned legacy character as the verified account's selection", async () => {
    mocks.characterLocal.set("selectedCharacter", { id: 7, name: "Private Archivist" })
    const { result } = renderHook(() => useSelectedAssistant())
    await waitFor(() => expect(mocks.characterLocal.has("selectedCharacter")).toBe(false))
    expect(result.current[0]).toBeNull()
  })

  it("broadcasts persona assistant selections to subscribers", async () => {
    const first = renderHook(() => useSelectedAssistant())
    const second = renderHook(() => useSelectedAssistant())

    await act(async () => {
      await first.result.current[1]({
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper",
        avatar_url: "https://example.com/garden.png"
      })
    })

    await waitFor(() => {
      expect(second.result.current[0]).toMatchObject({
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper",
        avatar_url: "https://example.com/garden.png"
      })
    })

    expect(mocks.characterLocal.has(SELECTED_CHARACTER_STORAGE_KEY)).toBe(false)
  })

  it("keeps the latest assistant selection when an older persistence write finishes last", async () => {
    let releaseFirstWrite = () => undefined
    mocks.assistantSetBarriers.push(
      new Promise<void>((resolve) => {
        releaseFirstWrite = resolve
      })
    )
    const { result } = renderHook(() => useSelectedAssistant())

    let firstWrite: Promise<void> | undefined
    let secondWrite: Promise<void> | undefined
    act(() => {
      firstWrite = result.current[1]({
        kind: "persona",
        id: "persisted-restore",
        name: "Persisted Restore"
      })
      secondWrite = result.current[1]({
        kind: "character",
        id: "explicit-selection",
        name: "Explicit Selection"
      })
    })

    await act(async () => {
      releaseFirstWrite()
      await Promise.all([firstWrite, secondWrite])
    })

    expect(mocks.assistantLocal.get(SELECTED_ASSISTANT_STORAGE_KEY)).toMatchObject({
      kind: "character",
      id: "explicit-selection",
      name: "Explicit Selection"
    })
    await waitFor(() => {
      expect(result.current[0]).toMatchObject({
        kind: "character",
        id: "explicit-selection"
      })
    })
  })

  it("rolls back a guarded assistant write cancelled during persistence", async () => {
    const { result } = renderHook(() => useSelectedAssistant())
    await act(async () => {
      await result.current[1]({
        kind: "character",
        id: "current-selection",
        name: "Current Selection"
      })
    })
    let releaseRestoreWrite = () => undefined
    mocks.assistantSetBarriers.push(
      new Promise<void>((resolve) => {
        releaseRestoreWrite = resolve
      })
    )
    let restoreIsCurrent = true
    let restoreWrite: Promise<void> | undefined

    act(() => {
      restoreWrite = result.current[1](
        {
          kind: "persona",
          id: "stale-restore",
          name: "Stale Restore"
        },
        { isCurrent: () => restoreIsCurrent }
      )
    })
    restoreIsCurrent = false
    await act(async () => {
      releaseRestoreWrite()
      await restoreWrite
    })

    expect(mocks.assistantLocal.get(SELECTED_ASSISTANT_STORAGE_KEY)).toMatchObject({
      kind: "character",
      id: "current-selection"
    })
    await waitFor(() => {
      expect(result.current[0]).toMatchObject({
        kind: "character",
        id: "current-selection"
      })
    })
  })

  it("does not let an already stale hydration cancel a queued explicit selection", async () => {
    const { result } = renderHook(() => useSelectedAssistant())
    await act(async () => {
      await result.current[1]({ kind: "character", id: "4", name: "Cedar" })
    })
    await act(async () => {
      const selection = result.current[1]({ kind: "character", id: "5", name: "Robot" })
      const staleHydration = result.current[1](
        { kind: "character", id: "4", name: "Cedar" },
        { isCurrent: () => false }
      )
      await Promise.all([selection, staleHydration])
    })
    expect(result.current[0]).toMatchObject({ id: "5", name: "Robot" })
    expect(mocks.assistantLocal.get(SELECTED_ASSISTANT_STORAGE_KEY)).toMatchObject({ id: "5" })
  })

  it("persists character selections and broadcasts to useSelectedCharacter consumers", async () => {
    const first = renderHook(() => useSelectedAssistant())
    const second = renderHook(() => useSelectedCharacter())
    await act(async () => {
      await first.result.current[1]({ kind: "character", id: "char-next", name: "Next Character", metadata: { selectionMode: "tracked" } })
    })
    expect(second.result.current[0]).toMatchObject({ id: "char-next", name: "Next Character" })
    first.unmount()
    const reloaded = renderHook(() => useSelectedAssistant())
    expect(reloaded.result.current[0]).toMatchObject({ kind: "character", id: "char-next", name: "Next Character" })
  })

  it("does not mirror overlay character selections into legacy character storage", async () => {
    const { result } = renderHook(() => useSelectedAssistant())

    await act(async () => {
      await result.current[1]({
        kind: "character",
        id: "char-overlay",
        name: "Overlay Guide",
        metadata: {
          selectionMode: "overlay"
        }
      })
    })

    await waitFor(() => {
      expect(result.current[0]).toMatchObject({
        kind: "character",
        id: "char-overlay",
        name: "Overlay Guide",
        metadata: {
          selectionMode: "overlay"
        }
      })
    })

    expect(mocks.characterLocal.has(SELECTED_CHARACTER_STORAGE_KEY)).toBe(false)
    expect(mocks.characterSync.has(SELECTED_CHARACTER_STORAGE_KEY)).toBe(false)
  })

  it("preserves overlay mode when a same-id assistant update omits metadata", async () => {
    const { result } = renderHook(() => useSelectedAssistant())

    await act(async () => {
      await result.current[1]({
        kind: "character",
        id: "char-overlay",
        name: "Overlay Guide",
        metadata: {
          selectionMode: "overlay"
        }
      })
    })

    await act(async () => {
      await result.current[1]({
        kind: "character",
        id: "char-overlay",
        name: "Overlay Guide",
        avatar_url: "https://example.com/overlay.png"
      })
    })

    await waitFor(() => {
      expect(result.current[0]).toMatchObject({
        kind: "character",
        id: "char-overlay",
        avatar_url: "https://example.com/overlay.png",
        metadata: {
          selectionMode: "overlay"
        }
      })
    })
  })

  it("clears both assistant and Character consumers and stays clear on reload", async () => {
    mocks.assistantLocal.set(SELECTED_ASSISTANT_STORAGE_KEY, { kind: "character", id: "char-existing", name: "Existing Guide" })
    const first = renderHook(() => useSelectedAssistant())
    const second = renderHook(() => useSelectedCharacter())
    expect(second.result.current[0]).toMatchObject({ id: "char-existing" })
    await act(async () => { await first.result.current[1](null) })
    expect(first.result.current[0]).toBeNull()
    expect(second.result.current[0]).toBeNull()
    first.unmount()
    const reloaded = renderHook(() => useSelectedAssistant())
    expect(reloaded.result.current[0]).toBeNull()
  })

  it("keeps selected assistant identity stable across unchanged rerenders", async () => {
    mocks.assistantLocal.set(SELECTED_ASSISTANT_STORAGE_KEY, {
      kind: "character",
      id: "char-stable",
      name: "Stable Guide",
      greeting: "Ready"
    })

    const { result, rerender } = renderHook(() => useSelectedAssistant())

    await waitFor(() => {
      expect(result.current[0]).toMatchObject({
        kind: "character",
        id: "char-stable",
        name: "Stable Guide"
      })
    })

    const firstSelection = result.current[0]
    rerender()

    expect(result.current[0]).toBe(firstSelection)
  })

  it("normalizes and preserves mirrored persona buddy summaries from stored selections", async () => {
    mocks.assistantLocal.set(SELECTED_ASSISTANT_STORAGE_KEY, {
      kind: "persona",
      id: "garden-helper",
      name: "Garden Helper",
      buddySummary: {
        hasBuddy: "true",
        personaName: "Garden Helper",
        roleSummary: "Old greenhouse guide",
        visual: {
          speciesId: "fox",
          silhouetteId: "sprout",
          paletteId: "fern"
        }
      }
    })

    const { result } = renderHook(() => useSelectedAssistant())

    await waitFor(() => {
      expect(result.current[0]).toMatchObject({
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper",
        buddy_summary: {
          has_buddy: true,
          persona_name: "Garden Helper",
          role_summary: "Old greenhouse guide",
          visual: {
            species_id: "fox",
            silhouette_id: "sprout",
            palette_id: "fern"
          }
        }
      })
    })
  })

  it("keeps current-surface persona buddy summaries ahead of stale mirrored storage", async () => {
    mocks.assistantLocal.set(SELECTED_ASSISTANT_STORAGE_KEY, {
      kind: "persona",
      id: "garden-helper",
      name: "Garden Helper",
      buddySummary: {
        hasBuddy: true,
        personaName: "Garden Helper",
        roleSummary: "Old greenhouse guide"
      }
    })

    const { result } = renderHook(() => useSelectedAssistant())

    await waitFor(() => {
      expect(result.current[0]?.kind).toBe("persona")
    })

    await act(async () => {
      await result.current[1]({
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper",
        buddy_summary: {
          has_buddy: true,
          persona_name: "Garden Helper",
          role_summary: "Fresh route summary",
          visual: {
            species_id: "owl",
            silhouette_id: "perch",
            palette_id: "sky"
          }
        }
      })
    })

    await waitFor(() => {
      expect(result.current[0]).toMatchObject({
        kind: "persona",
        id: "garden-helper",
        buddy_summary: {
          has_buddy: true,
          persona_name: "Garden Helper",
          role_summary: "Fresh route summary",
          visual: {
            species_id: "owl",
            silhouette_id: "perch",
            palette_id: "sky"
          }
        }
      })
    })

    expect(mocks.assistantLocal.get(SELECTED_ASSISTANT_STORAGE_KEY)).toMatchObject({
      kind: "persona",
      id: "garden-helper",
      name: "Garden Helper",
      buddy_summary: {
        has_buddy: true,
        persona_name: "Garden Helper",
        role_summary: "Fresh route summary",
        visual: {
          species_id: "owl",
          silhouette_id: "perch",
          palette_id: "sky"
        }
      }
    })
  })

  it("keeps useSelectedCharacter scoped to character assistant selections", async () => {
    const { result } = renderHook(() => {
      const assistantState = useSelectedAssistant()
      const characterState = useSelectedCharacter<Character | null>(null)
      return { assistantState, characterState }
    })

    await act(async () => {
      await result.current.assistantState[1]({
        kind: "persona",
        id: "garden-helper",
        name: "Garden Helper"
      })
    })

    await waitFor(() => {
      expect(result.current.characterState[0]).toBeNull()
    })

    const nextCharacter = {
      id: "char-42",
      name: "Guide",
      greeting: "Ready when you are",
      alternateGreetings: ["Let's begin"]
    } as Character & {
      alternateGreetings: string[]
    }

    await act(async () => {
      await result.current.characterState[1](nextCharacter)
    })

    await waitFor(() => {
      expect(result.current.assistantState[0]).toMatchObject({
        kind: "character",
        id: "char-42",
        name: "Guide",
        greeting: "Ready when you are",
        alternateGreetings: ["Let's begin"]
      })
      expect(result.current.characterState[0]).toMatchObject({
        id: "char-42",
        name: "Guide",
        greeting: "Ready when you are",
        alternateGreetings: ["Let's begin"]
      })
    })
  })

  it("preserves overlay selection mode when useSelectedCharacter updates the active character", async () => {
    const { result } = renderHook(() => {
      const assistantState = useSelectedAssistant()
      const characterState = useSelectedCharacter<Character | null>(null)
      return { assistantState, characterState }
    })

    await act(async () => {
      await result.current.assistantState[1]({
        kind: "character",
        id: "char-overlay",
        name: "Overlay Guide",
        metadata: {
          selectionMode: "overlay"
        }
      })
    })

    await waitFor(() => {
      expect(result.current.assistantState[0]).toMatchObject({
        kind: "character",
        id: "char-overlay",
        metadata: {
          selectionMode: "overlay"
        }
      })
    })

    await act(async () => {
      await result.current.characterState[1]({
        id: "char-overlay",
        name: "Overlay Guide",
        avatar_url: "https://example.com/overlay.png"
      } as Character)
    })

    await waitFor(() => {
      expect(result.current.assistantState[0]).toMatchObject({
        kind: "character",
        id: "char-overlay",
        avatar_url: "https://example.com/overlay.png",
        metadata: {
          selectionMode: "overlay"
        }
      })
    })
  })

  it("does not override an explicit selection mode when useSelectedCharacter updates the active character", async () => {
    const { result } = renderHook(() => {
      const assistantState = useSelectedAssistant()
      const characterState = useSelectedCharacter<Character | null>(null)
      return { assistantState, characterState }
    })

    await act(async () => {
      await result.current.assistantState[1]({
        kind: "character",
        id: "char-explicit",
        name: "Overlay Guide",
        metadata: {
          selectionMode: "overlay"
        }
      })
    })

    await waitFor(() => {
      expect(result.current.assistantState[0]).toMatchObject({
        kind: "character",
        id: "char-explicit",
        metadata: {
          selectionMode: "overlay"
        }
      })
    })

    await act(async () => {
      await result.current.characterState[1]({
        id: "char-explicit",
        name: "Tracked Guide",
        metadata: {
          selectionMode: "tracked"
        }
      } as Character & { metadata: { selectionMode: "tracked" } })
    })

    await waitFor(() => {
      expect(result.current.assistantState[0]).toMatchObject({
        kind: "character",
        id: "char-explicit",
        name: "Tracked Guide",
        metadata: {
          selectionMode: "tracked"
        }
      })
    })
  })
})
