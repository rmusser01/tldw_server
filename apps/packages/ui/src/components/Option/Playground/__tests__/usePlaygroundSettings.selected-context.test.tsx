import { renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import {
  usePlaygroundSettings,
  type UsePlaygroundSettingsDeps
} from "../hooks/usePlaygroundSettings"

const createDeps = (
  overrides: Partial<UsePlaygroundSettingsDeps> = {}
): UsePlaygroundSettingsDeps => ({
  selectedCharacterName: null,
  selectedSystemPrompt: "",
  selectedQuickPrompt: null,
  systemPrompt: "",
  ragPinnedResultsLength: 0,
  webSearch: false,
  jsonMode: false,
  compareModeActive: false,
  compareNeedsMoreModels: false,
  compareCapabilityIncompatibilities: [],
  voiceChatEnabled: false,
  showTokenBudgetWarning: false,
  tokenBudgetWarningText: "",
  summaryCheckpointSuggestion: { shouldSuggest: false },
  messagesLength: 0,
  showNonMessageContextWarning: false,
  nonMessageContextPercent: null,
  largestContextContributor: null,
  openKnowledgePanel: vi.fn(),
  openContextWindowModal: vi.fn(),
  openModelApiSelector: vi.fn(),
  setOpenModelSettings: vi.fn(),
  setModeLauncherOpen: vi.fn(),
  setOpenActorSettings: vi.fn(),
  trimLargestContextContributor: vi.fn(),
  insertSummaryCheckpointPrompt: vi.fn(),
  t: (_key, fallback) =>
    typeof fallback === "string" ? fallback : fallback?.defaultValue || _key,
  ...overrides
})

describe("usePlaygroundSettings selected context", () => {
  it("does not report pending quick insertion as shared or conflicting system behavior", () => {
    const { result } = renderHook(() =>
      usePlaygroundSettings(
        createDeps({
          selectedCharacterName: "Mira",
          selectedQuickPrompt: "Pending"
        })
      )
    )
    expect(result.current.compareHasPromptContext).toBe(false)
    expect(
      result.current.contextConflictWarnings.find(
        (item) => item.id === "character-prompt"
      )
    ).toBeUndefined()
  })

  it("describes selected character combinations as requiring review", () => {
    const { result } = renderHook(() =>
      usePlaygroundSettings(
        createDeps({
          selectedCharacterName: "Mira",
          ragPinnedResultsLength: 1,
          systemPrompt: "Be curious."
        })
      )
    )
    const warnings = result.current.contextConflictWarnings.filter((item) =>
      ["character-rag", "character-prompt"].includes(item.id)
    )
    expect(warnings).toHaveLength(2)
    expect(
      warnings.every(
        (item) =>
          item.text.includes("selected") &&
          item.text.includes("Review") &&
          !item.text.includes("both active")
      )
    ).toBe(true)
  })
})
