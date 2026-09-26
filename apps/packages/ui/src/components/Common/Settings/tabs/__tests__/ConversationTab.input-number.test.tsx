import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ConversationTab } from "../ConversationTab"
import { getChatSettingsForKey } from "@/services/chat-settings"
import type { ChatSettingsRecord } from "@/types/chat-session-settings"

// Keep AntD, the settings hook, and patch/persistence logic real. Only the
// extension storage API and server boundary are replaced in this DOM test.
const { stored } = vi.hoisted(() => ({ stored: new Map<string, unknown>() }))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async (key: string) => stored.get(key),
    set: async (key: string, value: unknown) => { stored.set(key, value) }
  })
}))

vi.mock("@plasmohq/storage/hook", async () => {
  const { useCallback, useState } = await import("react")
  return {
    useStorage: ({ key }: { key: string }) => {
      const [value, setValue] = useState(() => stored.get(key))
      const setStoredValue = useCallback(async (next: unknown) => {
        stored.set(key, next)
        setValue(next)
      }, [key])
      return [value, setStoredValue]
    }
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn(), success: vi.fn() })
}))
vi.mock("@/components/Common/Settings/PromptAssemblyPreview", () => ({
  PromptAssemblyPreview: () => null
}))
vi.mock("@/components/Common/Settings/LorebookDebugPanel", () => ({
  LorebookDebugPanel: () => null
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => undefined,
    listCharacters: async () => [],
    listChatMessages: async () => [],
    updateChatSettings: async (_id: string, settings: ChatSettingsRecord) => ({ settings })
  }
}))

const initialSettings: ChatSettingsRecord = {
  schemaVersion: 1,
  updatedAt: "2026-09-16T00:00:00.000Z",
  authorNotePosition: { mode: "depth", depth: 2 },
  chatGenerationOverride: {
    enabled: true,
    temperature: 0.7,
    top_p: 0.9,
    repetition_penalty: 1.2,
    stop: ["END"]
  },
  autoSummaryEnabled: true,
  autoSummaryThresholdMessages: 40,
  autoSummaryWindowMessages: 12
}
const storageKey = "chatSettings:server:chat-1"

const mountConversation = () => render(
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <ConversationTab
      historyId="history-1"
      selectedSystemPrompt={null}
      onSystemPromptChange={vi.fn()}
      uploadedFiles={[]}
      onRemoveFile={vi.fn()}
      serverChatId="chat-1"
      serverChatState="in-progress"
      onStateChange={vi.fn()}
      serverChatTopic={null}
      onTopicChange={vi.fn()}
      onVersionChange={vi.fn()}
    />
  </QueryClientProvider>
)

describe("ConversationTab real AntD numeric controls", () => {
  beforeEach(() => {
    stored.clear()
    stored.set(storageKey, structuredClone(initialSettings))
  })

  it("opens all six numeric controls without the addonBefore deprecation", async () => {
    const errors = vi.spyOn(console, "error").mockImplementation(() => {})
    const warnings = vi.spyOn(console, "warn").mockImplementation(() => {})
    mountConversation()
    await waitFor(() => expect(screen.getAllByRole("spinbutton")).toHaveLength(6))

    const deprecations = [...errors.mock.calls, ...warnings.mock.calls]
      .map((args) => args.map(String).join(" "))
      .filter((message) => /InputNumber.*addonBefore.*deprecated/.test(message))
    expect(deprecations).toEqual([])
  })

  it("labels each numeric control and preserves its initial value and bounds", () => {
    mountConversation()
    for (const [name, value, min, max] of [
      ["Depth", "2", "0", null],
      ["Temp", "0.70", "0", "2"],
      ["Top-p", "0.90", "0", "1"],
      ["Rep pen", "1.20", "0", "3"],
      ["Threshold", "40", "2", "5000"],
      ["Recent window", "12", "1", "39"]
    ] as const) {
      const input = screen.getByRole("spinbutton", { name })
      expect(input).toHaveValue(value)
      expect(input).toHaveAttribute("aria-valuemin", min)
      if (max) expect(input).toHaveAttribute("aria-valuemax", max)
      else expect(input).not.toHaveAttribute("aria-valuemax")
      expect(input).toBeEnabled()
    }
  })

  it("persists edits only on blur and restores all six values after reopening", async () => {
    const view = mountConversation()
    for (const [name, value, patch] of [
      ["Depth", "5", { authorNotePosition: { mode: "depth", depth: 5 } }],
      ["Temp", "1.25", { chatGenerationOverride: expect.objectContaining({ temperature: 1.25 }) }],
      ["Top-p", "0.65", { chatGenerationOverride: expect.objectContaining({ top_p: 0.65 }) }],
      ["Rep pen", "1.35", { chatGenerationOverride: expect.objectContaining({ repetition_penalty: 1.35 }) }],
      ["Threshold", "30", { autoSummaryThresholdMessages: 30 }],
      ["Recent window", "8", { autoSummaryWindowMessages: 8 }]
    ] as const) {
      const input = screen.getByRole("spinbutton", { name })
      const beforeEdit = await getChatSettingsForKey("server:chat-1")
      fireEvent.change(input, { target: { value } })
      expect(await getChatSettingsForKey("server:chat-1")).toEqual(beforeEdit)
      fireEvent.blur(input)
      await waitFor(async () => {
        expect(await getChatSettingsForKey("server:chat-1")).toMatchObject(patch)
      })
    }

    expect(await getChatSettingsForKey("server:chat-1")).toMatchObject({
      chatGenerationOverride: { enabled: true, stop: ["END"] },
      autoSummaryEnabled: true
    })
    view.unmount()
    mountConversation()
    for (const [name, value] of [
      ["Depth", "5"], ["Temp", "1.25"], ["Top-p", "0.65"],
      ["Rep pen", "1.35"], ["Threshold", "30"], ["Recent window", "8"]
    ]) {
      expect(screen.getByRole("spinbutton", { name })).toHaveValue(value)
    }
  }, 10_000)

  it("keeps generation and summary numeric controls disabled when their modes are off", () => {
    stored.set(storageKey, {
      ...initialSettings,
      chatGenerationOverride: { ...initialSettings.chatGenerationOverride, enabled: false },
      autoSummaryEnabled: false
    })
    mountConversation()
    for (const name of ["Temp", "Top-p", "Rep pen", "Threshold", "Recent window"]) {
      expect(screen.getByRole("spinbutton", { name })).toBeDisabled()
    }
    expect(screen.getByRole("spinbutton", { name: "Depth" })).toBeEnabled()
  })

  it("keeps the summary window below an edited threshold on blur", async () => {
    mountConversation()
    const threshold = screen.getByRole("spinbutton", { name: "Threshold" })
    fireEvent.change(threshold, { target: { value: "5" } })
    fireEvent.blur(threshold)
    await waitFor(async () => {
      expect(await getChatSettingsForKey("server:chat-1")).toMatchObject({
        autoSummaryThresholdMessages: 5,
        autoSummaryWindowMessages: 4
      })
    })
    expect(screen.getByRole("spinbutton", { name: "Recent window" }))
      .toHaveAttribute("aria-valuemax", "4")
  })
})
