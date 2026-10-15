import "@/hooks/chat/__tests__/native-plasmo-storage-fixture"
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ConversationTab } from "../ConversationTab"
import { getChatSettingsForKey } from "@/services/chat-settings"
import type { ChatSettingsRecord } from "@/types/chat-session-settings"

// Real AntD, owned storage hook and shared scoped persistence; replace only H1
// authority and the transport boundary. This is DOM evidence, not a live server.
import { Storage } from "@plasmohq/storage"
import { useStoreMessageOption } from "@/store/option"
import { installSettingsLocks, settingsTestOwner } from "@/services/__tests__/chat-settings-fixture"
import { getChatSettingsStorageKey, resolveChatSettingsKey } from "@/services/chat-settings"
const boundary = vi.hoisted(() => ({ native: true, settings: null as ChatSettingsRecord | null, write: vi.fn() }))
const owner = { kind: "native" as const, conversation_id: "chat-1", owner_key: "test-owner",
  request_scope: settingsTestOwner.requestScope, validate_lease: () => true }
const localOwner = { kind: "local" as const, conversation_id: "local-history-1", owner_key: "local-test-owner" }
const controller = { settingsMode: () => "ordinary" as const, getCurrent: () => ({ owner: boundary.native ? owner : localOwner }), fence: () => () => true, getSignal: () => undefined }
import { memory } from "@/hooks/chat/__tests__/local-history-fixture"
vi.mock("@/db/dexie/schema", async () => ({ db: (await import("@/hooks/chat/__tests__/local-history-fixture")).memory }))
vi.mock("@/hooks/chat/useHistorySelection", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/hooks/chat/useHistorySelection")>(),
  useHistorySelectionContext: () => controller
}))

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
    listCharacters: async () => [{ id: 4, name: "Saved primary" }, { id: 9, name: "Selected alias" }],
    listChatMessages: async () => [],
    getChat: async (id: string) => ({ id, version: 5, state: "in-progress", topic_label: null }),
    getChatSettings: async (id: string) => ({ conversation_id: id, settings: boundary.settings }),
    updateChatSettings: (...args: unknown[]) => boundary.write(...args)
  }
}))

const initialSettings: ChatSettingsRecord = {
  schemaVersion: 2,
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
const chatKey = resolveChatSettingsKey({ historyId: null, serverChatId: "chat-1", requestScope: settingsTestOwner.requestScope })
const storageKey = getChatSettingsStorageKey(chatKey)

const localKey = "local:local-history-1"
const localStorageKey = getChatSettingsStorageKey(localKey)
const mountConversation = (native = true, primary: string | number | null = 4) => {
  boundary.native = native
  useStoreMessageOption.setState({ historyId: native ? null : "local-history-1", serverChatId: native ? "chat-1" : null })
  return render(
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <ConversationTab
      historyId={native ? null : "local-history-1"}
      selectedSystemPrompt={null}
      onSystemPromptChange={vi.fn()}
      uploadedFiles={[]}
      onRemoveFile={vi.fn()}
      serverChatId={native ? "chat-1" : null}
      serverChatCharacterId={primary}
      serverChatState="in-progress"
      onStateChange={vi.fn()}
      serverChatTopic={null}
      onTopicChange={vi.fn()}
      onVersionChange={vi.fn()}
    />
  </QueryClientProvider>
  )
}

describe("ConversationTab real AntD numeric controls", () => {
  beforeEach(async () => {
    localStorage.clear(); sessionStorage.clear(); installSettingsLocks()
    useStoreMessageOption.setState({ historyId: null, serverChatId: "chat-1", chatDraftAddress: null, chatDraftRevision: 0 })
    boundary.settings = structuredClone(initialSettings)
    boundary.write.mockReset().mockImplementation(async (id: string, settings: ChatSettingsRecord) => {
      boundary.settings = { ...boundary.settings!, ...settings, updatedAt: "2026-10-08T00:00:00Z",
        ...(settings.chatGenerationOverride ? { chatGenerationOverride: { ...boundary.settings?.chatGenerationOverride, ...settings.chatGenerationOverride } } : {}) }
      return { conversation_id: id, settings: boundary.settings }
    })
    await new Storage({ area: "local" }).set(storageKey, { value: structuredClone(initialSettings), updatedAt: 1 })
    await new Storage({ area: "local" }).set(localStorageKey, structuredClone(initialSettings))
    memory.chatHistories.rows.clear()
    await memory.chatHistories.put({ id: "local-history-1", local_settings_guard: { initialized: true, revision: "numeric-test", pending: [] } })
  })

  it("keeps unsupported native controls read-only without promising prompt effects", async () => {
    mountConversation()
    await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Temp" })).toBeEnabled())
    for (const name of ["Author note", "Author note insertion", "Memory scope", "Turn-taking mode", "Participants", "Directed responder", "Auto summary"])
      expect(screen.getByLabelText(name)).toBeDisabled()
    for (const name of ["Depth", "Threshold", "Recent window"])
      expect(screen.getByRole("spinbutton", { name })).toBeDisabled()
    fireEvent.change(screen.getByRole("textbox", { name: "Author note" }), { target: { value: "Unsupported native note" } })
    expect(boundary.write).not.toHaveBeenCalled()
    expect(screen.getByRole("button", { name: "Retry conversation changes" })).toBeDisabled()
    expect(screen.getAllByText("Author notes, memory, directed replies and summaries are unavailable in saved server chat. Existing values are shown for reference.").length).toBeGreaterThan(0)
    expect(screen.queryByText("Pinned messages are excluded from auto-summary compression.")).not.toBeInTheDocument()
  })

  it("resets native participants to the numeric saved primary instead of a selected alias", async () => {
    boundary.settings = { ...initialSettings, participantCharacterIds: ["4", "9"] }
    await new Storage({ area: "local" }).set(storageKey, { value: boundary.settings, updatedAt: 1 })
    mountConversation(true, "4")
    const reset = await screen.findByRole("button", { name: "Reset to primary character" })
    await waitFor(() => expect(reset).toBeEnabled())
    fireEvent.click(reset)
    expect(boundary.write).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Apply conversation changes" }))
    await waitFor(() => expect(boundary.write).toHaveBeenCalledOnce())
    expect(boundary.write.mock.calls[0][1]).toMatchObject({ participantCharacterIds: [4] })
    await waitFor(async () => expect(await getChatSettingsForKey(chatKey)).toMatchObject({ participantCharacterIds: [4] }))
  })

  it.each([null, "alias-9", 0, -1, Number.MAX_SAFE_INTEGER + 1])("refuses a native participant reset with unknown or unsafe saved primary %s", async primary => {
    mountConversation(true, primary)
    await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Temp" })).toBeEnabled())
    expect(screen.getByRole("combobox", { name: "Participants" })).toBeDisabled()
    const reset = screen.getByRole("button", { name: "Reset to primary character" })
    expect(reset).toBeDisabled()
    fireEvent.click(reset)
    expect(boundary.write).not.toHaveBeenCalled()
  })

  it.each([false, true])("verifies a retained unsupported native change without resending it (already applied=%s)", async committed => {
    const { applyChatSettingsPatch, getChatSettingsRecovery } = await import("@/services/chat-settings")
    const target = { historyId: null, serverChatId: "chat-1", requestScope: settingsTestOwner.requestScope, validate_lease: () => true }
    boundary.write.mockImplementationOnce(async (_id: string, patch: Partial<ChatSettingsRecord>) => {
      if (committed) boundary.settings = { ...boundary.settings!, ...patch }
      throw new Error("Uncertain legacy response")
    })
    await expect(applyChatSettingsPatch({ ...target, patch: { authorNote: "Legacy pending note" } })).rejects.toThrow("chat_settings_server_unconfirmed")
    const receipt = await getChatSettingsRecovery(target)
    mountConversation()
    const review = await screen.findByRole("button", { name: "Review previous settings change" })
    await waitFor(() => expect(review).toBeEnabled())
    fireEvent.click(review)
    const verify = await screen.findByRole("button", { name: "Verify previous change" })
    await waitFor(() => expect(verify).toBeEnabled())
    expect(screen.getByText(/Legacy pending note/, { selector: "pre" })).toBeInTheDocument()
    fireEvent.click(verify)
    if (committed) {
      await waitFor(() => expect(review).not.toBeInTheDocument())
      expect(screen.queryByText("Review previous settings change", { exact: true })).not.toBeInTheDocument()
    } else {
      await screen.findByText(/The server has not confirmed this change/, { selector: '[role="alert"]' })
      expect(await getChatSettingsRecovery(target)).toEqual(receipt)
      expect(screen.getByText(/Legacy pending note/, { selector: "pre" })).toBeInTheDocument()
    }
    expect(boundary.write).toHaveBeenCalledTimes(1)
  })

  it("opens all six numeric controls without the addonBefore deprecation", async () => {
    const errors = vi.spyOn(console, "error").mockImplementation(() => {})
    const warnings = vi.spyOn(console, "warn").mockImplementation(() => {})
    mountConversation(false)
    await waitFor(() => expect(screen.getAllByRole("spinbutton")).toHaveLength(6))

    const deprecations = [...errors.mock.calls, ...warnings.mock.calls]
      .map((args) => args.map(String).join(" "))
      .filter((message) => /InputNumber.*addonBefore.*deprecated/.test(message))
    expect(deprecations).toEqual([])
  })

  it("labels each numeric control and preserves its initial value and bounds", async () => {
    mountConversation(false)
    await screen.findByRole("spinbutton", { name: "Depth" })
    for (const [name, value, min, max] of [
      ["Depth", "2", "0", null],
      ["Temp", "0.70", "0", "2"],
      ["Top-p", "0.90", "0", "1"],
      ["Rep pen", "1.20", "0", "3"],
      ["Threshold", "40", "2", "5000"],
      ["Recent window", "12", "1", "39"]
    ] as const) {
      const input = screen.getByRole("spinbutton", { name })
      await waitFor(() => expect(input).toHaveValue(value))
      expect(input).toHaveAttribute("aria-valuemin", min)
      if (max) expect(input).toHaveAttribute("aria-valuemax", max)
      else expect(input).not.toHaveAttribute("aria-valuemax")
      expect(input).toBeEnabled()
    }
  })

  it("names the actual AntD selectors and text inputs for assistive technology", async () => {
    boundary.settings = { ...initialSettings, presetScope: "chat", memoryScope: "both", summary: { content: "Saved summary" } } as ChatSettingsRecord
    await new Storage({ area: "local" }).set(storageKey, { value: boundary.settings, updatedAt: 1 })
    await new Storage({ area: "local" }).set(localStorageKey, boundary.settings)
    mountConversation(false)
    await screen.findByRole("spinbutton", { name: "Depth" })
    for (const name of ["Conversation status", "Author note insertion", "Greeting scope", "Preset scope", "Chat preset override", "Chat generation override", "Memory scope", "Turn-taking mode", "Participants", "Directed responder", "Memory character", "Auto summary"]) {
      const input = screen.getByLabelText(name, { selector: '[role="combobox"]' })
      expect(input).toHaveAttribute("role", "combobox")
      expect(input).toHaveAccessibleName(name)
      expect(input).toBeVisible()
    }
    for (const name of ["Model default system prompt", "Conversation tag", "Author note", "Stop sequences", "Memory note", "Current summary"]) {
      const input = screen.getByLabelText(name, { selector: "input, textarea" })
      expect(input).toHaveRole("textbox")
      expect(input).toHaveAccessibleName(name)
      expect(input).toBeVisible()
    }
  })

  it("stages all six numeric edits until Apply and restores them through owned storage after reopening", async () => {
    const view = mountConversation(false)
    await waitFor(() => expect(screen.getByRole("spinbutton", { name: "Depth" })).toBeEnabled())
    for (const [name, value] of [
      ["Depth", "5"], ["Temp", "1.25"], ["Top-p", "0.65"],
      ["Rep pen", "1.35"], ["Threshold", "30"], ["Recent window", "8"]
    ] as const) {
      const input = screen.getByLabelText(name, { selector: "input" })
      expect(input).toHaveAttribute("role", "spinbutton")
      expect(input).toBeVisible()
      fireEvent.change(input, { target: { value } })
      fireEvent.blur(input)
      expect(boundary.write).not.toHaveBeenCalled()
      expect(await getChatSettingsForKey(localKey)).toEqual(initialSettings)
    }
    const apply = screen.getByRole("button", { name: "Apply conversation changes" })
    fireEvent.click(apply)
    await waitFor(async () => expect(await getChatSettingsForKey(localKey)).toMatchObject({ authorNotePosition: { mode: "depth", depth: 5 } }))
    await waitFor(() => expect(screen.getByLabelText("Depth", { selector: "input" })).toBeEnabled())
    expect(apply).toBeInTheDocument()
    expect(apply).toBeDisabled()
    expect(boundary.write).not.toHaveBeenCalled()
    expect(await getChatSettingsForKey(localKey)).toMatchObject({
      authorNotePosition: { mode: "depth", depth: 5 },
      chatGenerationOverride: { enabled: true, temperature: 1.25, top_p: 0.65, repetition_penalty: 1.35, stop: ["END"] },
      autoSummaryEnabled: true, autoSummaryThresholdMessages: 30, autoSummaryWindowMessages: 8
    })
    expect(memory.chatHistories.rows.size).toBe(1)
    view.unmount()
    mountConversation(false)
    await waitFor(() => {
      for (const [name, value] of [
        ["Depth", "5"], ["Temp", "1.25"], ["Top-p", "0.65"],
        ["Rep pen", "1.35"], ["Threshold", "30"], ["Recent window", "8"]
      ]) {
        const input = screen.getByLabelText(name, { selector: "input" })
        expect(input).toHaveAttribute("role", "spinbutton")
        expect(input).toBeVisible()
        expect(input).toHaveValue(value)
      }
    })
  }, 10_000)

  it("recovers a lost acknowledgement after Cancel and remount without replaying the committed settings PUT", async () => {
    boundary.write.mockImplementationOnce(async (id: string, settings: ChatSettingsRecord) => {
      boundary.settings = { ...boundary.settings!, ...settings, updatedAt: "2026-10-08T00:00:00Z" }
      throw new Error("Response lost after committing")
    })
    let view = mountConversation()
    const note = await screen.findByRole("spinbutton", { name: "Temp" })
    await waitFor(() => expect(note).toBeEnabled())
    fireEvent.change(note, { target: { value: "0.45" } })
    fireEvent.click(screen.getByRole("button", { name: "Apply conversation changes" }))
    await screen.findByRole("alert")
    await waitFor(() => expect(screen.getByRole("button", { name: "Cancel conversation changes" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "Cancel conversation changes" }))
    expect(await screen.findByRole("button", { name: "Review previous settings change" })).toBeEnabled()
    fireEvent.change(screen.getByLabelText("Temp", { selector: "input" }), { target: { value: "0.66" } })
    expect(screen.getByRole("button", { name: "Review previous settings change" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Apply conversation changes" })).toBeDisabled()
    expect(boundary.write).toHaveBeenCalledTimes(1)
    view.unmount()
    view = mountConversation()
    const review = await screen.findByRole("button", { name: "Review previous settings change" })
    await waitFor(() => expect(review).toBeEnabled())
    fireEvent.click(review)
    expect(screen.getByLabelText("Temp", { selector: "input" })).toHaveAttribute("role", "spinbutton")
    expect(screen.getByLabelText("Temp", { selector: "input" })).toBeVisible()
    await waitFor(() => expect(screen.getByLabelText("Temp", { selector: "input" })).toHaveValue("0.45"))
    expect(screen.getByText(/0.45/, { selector: "pre" })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole("button", { name: "Apply conversation changes" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "Apply conversation changes" }))
    await waitFor(async () => expect(await getChatSettingsForKey(chatKey)).toMatchObject({ chatGenerationOverride: { temperature: 0.45 } }))
    await waitFor(() => expect(screen.queryByRole("button", { name: "Review previous settings change" })).not.toBeInTheDocument())
    expect(boundary.write).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(screen.getByLabelText("Temp", { selector: "input" })).toBeEnabled())
    fireEvent.change(screen.getByLabelText("Temp", { selector: "input" }), { target: { value: "0.66" } })
    fireEvent.click(screen.getByRole("button", { name: "Apply conversation changes" }))
    await waitFor(async () => expect(await getChatSettingsForKey(chatKey)).toMatchObject({ chatGenerationOverride: { temperature: 0.66 } }))
    expect(boundary.write).toHaveBeenCalledTimes(2)
  }, 15_000)

  it("keeps generation and summary numeric controls disabled when their modes are off", async () => {
    boundary.settings = {
      ...initialSettings,
      chatGenerationOverride: { ...initialSettings.chatGenerationOverride, enabled: false }, autoSummaryEnabled: false
    }
    await new Storage({ area: "local" }).set(storageKey, { value: boundary.settings, updatedAt: 1 })
    await new Storage({ area: "local" }).set(localStorageKey, boundary.settings)
    mountConversation(false)
    await screen.findByRole("spinbutton", { name: "Depth" })
    for (const name of ["Temp", "Top-p", "Rep pen", "Threshold", "Recent window"])
      expect(screen.getByRole("spinbutton", { name })).toBeDisabled()
    expect(screen.getByRole("spinbutton", { name: "Depth" })).toBeEnabled()
  })

  it("keeps the staged summary window below an edited threshold and saves it only on Apply", async () => {
    mountConversation(false)
    await screen.findByRole("spinbutton", { name: "Depth" })
    const threshold = screen.getByRole("spinbutton", { name: "Threshold" })
    fireEvent.change(threshold, { target: { value: "5" } })
    fireEvent.blur(threshold)
    expect(boundary.write).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Apply conversation changes" }))
    await waitFor(async () => {
      expect(await getChatSettingsForKey(localKey)).toMatchObject({
        autoSummaryThresholdMessages: 5,
        autoSummaryWindowMessages: 4
      })
    })
    expect(screen.getByRole("spinbutton", { name: "Recent window" }))
      .toHaveAttribute("aria-valuemax", "4")
  })
})
