import { beforeEach, describe, expect, it } from "vitest"
import { useStoreChatModelSettings } from "../model"
import {
  mergeGlobalAndScopedSettings,
  normalizeModelSettingsScope,
  stripUndefinedScopedSettings
} from "../model-settings-scope"

describe("provider:model scoped chat model settings", () => {
  beforeEach(() => {
    useStoreChatModelSettings.getState().reset()
  })

  it("normalizes provider:model setting scope keys", () => {
    expect(normalizeModelSettingsScope(" OpenAI ", " gpt-4o ")).toBe(
      "openai:gpt-4o"
    )
    expect(normalizeModelSettingsScope("Anthropic", "claude:3")).toBe(
      "anthropic:claude:3"
    )
    expect(normalizeModelSettingsScope("", "gpt-4o")).toBeNull()
    expect(normalizeModelSettingsScope("openai", "")).toBeNull()
  })

  it("merges global defaults with scoped overrides without mutating inputs", () => {
    const globalDefaults = {
      temperature: 0.7,
      topP: 0.9,
      systemPrompt: "global prompt"
    }
    const scopedOverrides = {
      temperature: 0.2,
      systemPrompt: undefined
    }

    expect(
      mergeGlobalAndScopedSettings(globalDefaults, scopedOverrides)
    ).toEqual({
      temperature: 0.2,
      topP: 0.9,
      systemPrompt: "global prompt"
    })
    expect(globalDefaults).toEqual({
      temperature: 0.7,
      topP: 0.9,
      systemPrompt: "global prompt"
    })
  })

  it("strips undefined scoped settings so they fall back to global defaults", () => {
    expect(
      stripUndefinedScopedSettings({
        temperature: 0.5,
        topP: undefined,
        systemPrompt: ""
      })
    ).toEqual({
      temperature: 0.5,
      systemPrompt: ""
    })
  })

  it("keeps compatibility setters scoped to the active provider:model", () => {
    const store = useStoreChatModelSettings.getState()

    store.updateSettings({ temperature: 0.7, topP: 0.9 })
    store.setActiveSettingsScope("openai:gpt-4o")
    useStoreChatModelSettings.getState().setTemperature(0.2)

    expect(useStoreChatModelSettings.getState().temperature).toBe(0.2)
    expect(
      useStoreChatModelSettings.getState().scopedSettingsByModelKey[
        "openai:gpt-4o"
      ]
    ).toMatchObject({ temperature: 0.2 })

    useStoreChatModelSettings
      .getState()
      .setActiveSettingsScope("anthropic:claude-3-5-sonnet")

    expect(useStoreChatModelSettings.getState().temperature).toBe(0.7)
    useStoreChatModelSettings.getState().setTemperature(0.4)

    expect(
      useStoreChatModelSettings.getState().getEffectiveSettings("openai:gpt-4o")
        .temperature
    ).toBe(0.2)
    expect(
      useStoreChatModelSettings
        .getState()
        .getEffectiveSettings("anthropic:claude-3-5-sonnet").temperature
    ).toBe(0.4)
  })

  it("canonicalizes scoped setting keys when casing or whitespace differs", () => {
    const store = useStoreChatModelSettings.getState()

    store.updateSettings({ temperature: 0.7 })
    store.updateScopedSetting(" OpenAI:gpt-4o ", "temperature", 0.2)
    store.setActiveSettingsScope("openai:gpt-4o")

    expect(useStoreChatModelSettings.getState().temperature).toBe(0.2)
    expect(
      useStoreChatModelSettings.getState().scopedSettingsByModelKey[
        "openai:gpt-4o"
      ]
    ).toMatchObject({ temperature: 0.2 })
  })

  it("persists scoped updates even when they match the current global value", () => {
    const store = useStoreChatModelSettings.getState()

    store.updateSettings({ temperature: 0.7 })
    store.setActiveSettingsScope("openai:gpt-4o")
    useStoreChatModelSettings.getState().updateSetting("temperature", 0.7)

    expect(
      useStoreChatModelSettings.getState().scopedSettingsByModelKey[
        "openai:gpt-4o"
      ]
    ).toMatchObject({ temperature: 0.7 })

    useStoreChatModelSettings.getState().setActiveSettingsScope(undefined)
    useStoreChatModelSettings.getState().updateSetting("temperature", 0.4)
    useStoreChatModelSettings.getState().setActiveSettingsScope("openai:gpt-4o")

    expect(useStoreChatModelSettings.getState().temperature).toBe(0.7)
  })

  it("hydrates effective settings when switching active model scopes", () => {
    const store = useStoreChatModelSettings.getState()

    store.updateSettings({ temperature: 0.6, topK: 40 })
    store.updateScopedSetting("openai:gpt-4o", "temperature", 0.1)
    store.updateScopedSetting("anthropic:claude-3-5-sonnet", "topK", 12)

    store.setActiveSettingsScope("openai:gpt-4o")
    expect(useStoreChatModelSettings.getState().temperature).toBe(0.1)
    expect(useStoreChatModelSettings.getState().topK).toBe(40)

    useStoreChatModelSettings
      .getState()
      .setActiveSettingsScope("anthropic:claude-3-5-sonnet")
    expect(useStoreChatModelSettings.getState().temperature).toBe(0.6)
    expect(useStoreChatModelSettings.getState().topK).toBe(12)
  })
})
