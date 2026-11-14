import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useStoreChatModelSettings } from "@/store/model"
import { captureModelSettingsOperation } from "../model-settings-operation"

describe("original model-settings operation", () => {
  const cleanup: (() => void)[] = []
  beforeEach(() => useStoreChatModelSettings.getState().reset())
  afterEach(() => {
    cleanup.splice(0).forEach((stop) => stop())
  })

  it("accepts unchanged noops and individually issued scope, behavior and preset writes", () => {
    const frame = captureModelSettingsOperation()
    expect(
      frame.write(() =>
        useStoreChatModelSettings.getState().setActiveSettingsScope(null)
      )
    ).toBe(true)
    expect(
      frame.write(() =>
        useStoreChatModelSettings
          .getState()
          .setActiveSettingsScope("openai:scene")
      )
    ).toBe(true)
    expect(
      frame.write(() =>
        useStoreChatModelSettings.getState().setSystemPrompt("Owned behavior")
      )
    ).toBe(true)
    expect(
      frame.write(() =>
        useStoreChatModelSettings
          .getState()
          .updateSettings({ temperature: 0.2, topP: 0.8 })
      )
    ).toBe(true)
    expect(frame.isCurrent()).toBe(true)
    expect(useStoreChatModelSettings.getState()).toMatchObject({
      activeSettingsScope: "openai:scene",
      systemPrompt: "Owned behavior",
      temperature: 0.2,
      topP: 0.8
    })
  })

  it.each(["new value", "same value", "edit then Undo"] as const)(
    "refuses old writes after a foreign %s setting publication",
    (change) => {
      useStoreChatModelSettings.getState().setTemperature(0.7)
      const frame = captureModelSettingsOperation()
      useStoreChatModelSettings
        .getState()
        .setTemperature(change === "same value" ? 0.7 : 1.3)
      if (change === "edit then Undo")
        useStoreChatModelSettings.getState().setTemperature(0.7)
      const foreign = useStoreChatModelSettings.getState()
      const write = vi.fn(() =>
        useStoreChatModelSettings.getState().setSystemPrompt("Old behavior")
      )
      expect(frame.write(write)).toBe(false)
      expect(write).not.toHaveBeenCalled()
      expect(useStoreChatModelSettings.getState()).toBe(foreign)
      expect(frame.isCurrent()).toBe(false)
    }
  )

  it.each(["before", "after"] as const)(
    "does not adopt a nested foreign edit from a subscriber registered %s its own observer",
    (ordering) => {
      let armed = true
      const subscribe = () =>
        useStoreChatModelSettings.subscribe((state) => {
          if (!armed || state.systemPrompt !== "Owned behavior") return
          armed = false
          useStoreChatModelSettings.getState().setTemperature(1.7)
        })
      if (ordering === "before") cleanup.push(subscribe())
      const frame = captureModelSettingsOperation()
      let stop: (() => void) | undefined
      expect(
        frame.write(() => {
          if (ordering === "after") stop = subscribe()
          useStoreChatModelSettings.getState().setSystemPrompt("Owned behavior")
          stop?.()
        })
      ).toBe(false)
      const foreign = useStoreChatModelSettings.getState()
      expect(foreign.temperature).toBe(1.7)
      expect(frame.isCurrent()).toBe(false)
      expect(
        frame.write(() =>
          useStoreChatModelSettings.getState().setTemperature(0.2)
        )
      ).toBe(false)
      expect(useStoreChatModelSettings.getState()).toBe(foreign)
    }
  )

  it("retires a failed setter even when it published no state", () => {
    const frame = captureModelSettingsOperation()
    const failure = new Error("Declared settings writer failure")
    expect(() =>
      frame.write(() => {
        throw failure
      })
    ).toThrow(failure)
    expect(frame.isCurrent()).toBe(false)
  })
})
