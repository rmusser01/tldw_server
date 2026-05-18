import { describe, expect, it } from "vitest"
import {
  buildConfiguredFirstModelOptions,
  buildConfiguredFirstModelSections,
  formatModelsLastRefreshedTime,
  sortModelsConfiguredFirst,
  summarizeProviderReadiness,
} from "../modelsDisplayUtils"

describe("models display utilities", () => {
  it.each([
    [new Date(2026, 1, 18, 9, 5).getTime(), "09:05"],
    [new Date(2026, 1, 18, 0, 7).getTime(), "00:07"],
  ])("formats %s as a dayjs-compatible HH:mm time", (value, expected) => {
    expect(formatModelsLastRefreshedTime(value)).toBe(expected)
  })

  it("orders the selected default model before other configured and catalog models", () => {
    const sorted = sortModelsConfiguredFirst([
      { id: "unavailable-a", provider: "alpha", configured: false, usable: false },
      { id: "usable-b", provider: "beta", configured: false, usable: true },
      { id: "configured-c", provider: "gamma", configured: true, usable: false },
      { id: "selected-d", provider: "delta", configured: true, usable: true, selected: true },
      { id: "configured-usable-e", provider: "epsilon", configured: true, usable: true },
    ])

    expect(sorted.map((model) => model.id)).toEqual([
      "selected-d",
      "configured-usable-e",
      "configured-c",
      "usable-b",
      "unavailable-a",
    ])
  })

  it("keeps the auto option before concrete configured-first model choices", () => {
    const options = buildConfiguredFirstModelOptions(
      [
        { id: "llama3", provider: "ollama", configured: true, usable: true },
        { id: "gpt-4o", provider: "openai", configured: false, usable: false },
      ],
      { autoLabel: "Auto (route on server)" }
    )

    expect(options.map((option) => option.value)).toEqual([
      "auto",
      "llama3",
      "gpt-4o",
    ])
  })

  it("keeps the full catalog available after the configured-first section", () => {
    const sections = buildConfiguredFirstModelSections([
      { id: "local-a", provider: "ollama", configured: true, usable: true },
      { id: "remote-b", provider: "openai", configured: false, usable: false },
      { id: "remote-c", provider: "anthropic", configured: false, usable: false },
    ])

    expect(sections.configuredFirst.map((model) => model.id)).toEqual(["local-a"])
    expect(sections.fullCatalog.map((model) => model.id)).toEqual([
      "local-a",
      "remote-c",
      "remote-b",
    ])
  })

  it("summarizes provider readiness without treating unknown providers as configured", () => {
    const summary = summarizeProviderReadiness([
      { id: "local-a", provider: "ollama", configured: true, usable: true },
      { id: "local-b", provider: "ollama", configured: true, usable: true },
      { id: "remote-a", provider: "openai", configured: false, usable: false },
      { id: "remote-b", provider: "anthropic", usable: true },
    ])

    expect(summary).toEqual({
      totalProviders: 3,
      configuredProviders: 1,
      usableProviders: 2,
      unavailableProviders: 1,
      selectedModelIds: [],
      hasConfiguredUsableProvider: true,
    })
  })
})
