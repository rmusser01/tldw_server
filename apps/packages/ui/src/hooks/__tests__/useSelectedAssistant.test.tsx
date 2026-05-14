import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { Character } from "@/types/character"

const mocks = vi.hoisted(() => {
  const assistantLocal = new Map<string, unknown>()
  const assistantSync = new Map<string, unknown>()
  const characterLocal = new Map<string, unknown>()
  const characterSync = new Map<string, unknown>()

  const createStorageMock = (map: Map<string, unknown>) => ({
    get: vi.fn(async (key: string) => (map.has(key) ? map.get(key) : null)),
    set: vi.fn(async (key: string, value: unknown) => {
      if (value == null) {
        map.delete(key)
        return
      }
      map.set(key, value)
    }),
    remove: vi.fn(async (key: string) => {
      map.delete(key)
    })
  })

  return {
    assistantLocal,
    assistantSync,
    characterLocal,
    characterSync,
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
    assistantStorage: createStorageMock(assistantLocal),
    assistantSyncStorage: createStorageMock(assistantSync),
    characterStorage: createStorageMock(characterLocal),
    characterSyncStorage: createStorageMock(characterSync)
  }
})

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

      const getStoredValue = () =>
        (store.has(key) ? store.get(key) : initialValue) ?? null

      const [value, setRenderValue] = ReactModule.useState(getStoredValue)

      ReactModule.useEffect(() => {
        setRenderValue(getStoredValue())
      }, [key])

      const setValue = async (next: unknown) => {
        const resolved =
          typeof next === "function"
            ? (next as (prev: unknown) => unknown)(getStoredValue())
            : next
        if (resolved == null) {
          store.delete(key)
          setRenderValue(null)
          return
        }
        store.set(key, resolved)
        setRenderValue(resolved)
      }

      return [value, setValue, { isLoading: false, setRenderValue }] as const
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
  SELECTED_ASSISTANT_STORAGE_KEY
} from "@/utils/selected-assistant-storage"
import {
  SELECTED_CHARACTER_STORAGE_KEY
} from "@/utils/selected-character-storage"

describe("useSelectedAssistant", () => {
  beforeEach(() => {
    mocks.assistantLocal.clear()
    mocks.assistantSync.clear()
    mocks.characterLocal.clear()
    mocks.characterSync.clear()
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

  it("migrates a stored selectedCharacter record into a character assistant selection", async () => {
    mocks.characterLocal.set(SELECTED_CHARACTER_STORAGE_KEY, {
      id: 7,
      name: "Archivist",
      avatar_url: "https://example.com/avatar.png",
      greeting: "Hello there",
      alternateGreetings: ["Welcome back"]
    })

    const { result } = renderHook(() => useSelectedAssistant())

    await waitFor(() => {
      expect(result.current[0]).toMatchObject({
        kind: "character",
        id: "7",
        name: "Archivist",
        avatar_url: "https://example.com/avatar.png",
        greeting: "Hello there",
        alternateGreetings: ["Welcome back"]
      })
    })

    expect(mocks.assistantLocal.get(SELECTED_ASSISTANT_STORAGE_KEY)).toMatchObject({
      kind: "character",
      id: "7",
      name: "Archivist"
    })
    expect(mocks.characterLocal.get(SELECTED_CHARACTER_STORAGE_KEY)).toMatchObject(
      {
        id: "7",
        name: "Archivist",
        alternateGreetings: ["Welcome back"]
      }
    )
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
})
