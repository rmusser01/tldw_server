import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import type { Character } from "@/types/character"
import type { ChatHistory, Message } from "@/store/option/types"
import { ChatGreetingPicker } from "../ChatGreetingPicker"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import { normalizeChatSettingsRecord } from "@/services/chat-settings"
import type { ChatSettingsRecord } from "@/types/chat-session-settings"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  buildGreetingOptionsFromEntries,
  buildGreetingsChecksumFromOptions,
  collectGreetingEntries
} from "@/utils/character-greetings"

const notificationErrorMock = vi.hoisted(() => vi.fn())

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue || key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [""]
}))

vi.mock("antd", () => ({
  notification: {
    error: notificationErrorMock
  },
  Select: ({ value, onChange, options, disabled }: any) => (
    <select
      data-testid="greeting-select"
      value={value ?? ""}
      onChange={(event) => onChange?.(event.target.value)}
      disabled={disabled}
    >
      {(options ?? []).map((option: any) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
  Switch: ({ checked, onChange }: any) => (
    <input
      type="checkbox"
      role="switch"
      checked={Boolean(checked)}
      onChange={(event) => onChange?.(event.target.checked)}
    />
  )
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(),
    addChatMessage: vi.fn()
  }
}))

const character = {
  id: "char-1",
  name: "Guide",
  greeting: "Welcome aboard",
  alternateGreetings: ["Good to see you"]
} as Character

const greetingEntries = collectGreetingEntries(character)
const greetingOptions = buildGreetingOptionsFromEntries(greetingEntries)
const checksum = buildGreetingsChecksumFromOptions(greetingOptions)
const defaultGreetingId = greetingOptions[0]?.id ?? null
const alternateGreetingId = greetingOptions[1]?.id ?? null

const renderPicker = (
  settingsPatch: Partial<ChatSettingsRecord>,
  updateSettings: (patch: Partial<ChatSettingsRecord>) => Promise<ChatSettingsRecord | null>,
  extraProps: Record<string, unknown> = {}
) => {
  const settings = normalizeChatSettingsRecord(settingsPatch)
  vi.mocked(useChatSettingsRecord).mockReturnValue({
    settings,
    updateSettings,
    chatKey: "chat:test"
  })

  render(
    <ChatGreetingPicker
      selectedCharacter={character}
      messages={[]}
      historyId="history-1"
      serverChatId={null}
      {...extraProps}
    />
  )
}

describe("ChatGreetingPicker", () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it("falls back to current selection when disabling default without stored selection", () => {
    const updateSettings = vi.fn(
      async (_patch: Partial<ChatSettingsRecord>) => null
    )
    renderPicker(
      {
        useCharacterDefault: true,
        greetingSelectionId: null,
        greetingsChecksum: checksum
      },
      updateSettings
    )

    const [useDefaultSwitch] = screen.getAllByRole("switch")
    fireEvent.click(useDefaultSwitch)

    expect(updateSettings).toHaveBeenCalledWith({
      useCharacterDefault: false,
      greetingSelectionId: defaultGreetingId,
      greetingsChecksum: checksum
    })
  })

  it("keeps stored greeting selection when disabling default", () => {
    const updateSettings = vi.fn(
      async (_patch: Partial<ChatSettingsRecord>) => null
    )
    renderPicker(
      {
        useCharacterDefault: true,
        greetingSelectionId: alternateGreetingId,
        greetingsChecksum: checksum
      },
      updateSettings
    )

    const [useDefaultSwitch] = screen.getAllByRole("switch")
    fireEvent.click(useDefaultSwitch)

    expect(updateSettings).toHaveBeenCalledWith({
      useCharacterDefault: false,
      greetingSelectionId: alternateGreetingId,
      greetingsChecksum: checksum
    })
  })

  it("updates selection when picking a different greeting", () => {
    const updateSettings = vi.fn(
      async (_patch: Partial<ChatSettingsRecord>) => null
    )
    renderPicker(
      {
        useCharacterDefault: false,
        greetingSelectionId: defaultGreetingId,
        greetingsChecksum: checksum
      },
      updateSettings
    )

    const select = screen.getByTestId("greeting-select")
    fireEvent.change(select, { target: { value: alternateGreetingId } })

    expect(updateSettings).toHaveBeenCalledWith({
      greetingSelectionId: alternateGreetingId,
      greetingsChecksum: checksum,
      useCharacterDefault: false
    })
  })

  it("resolves legacy index-based selection ids to the matching option", () => {
    const updateSettings = vi.fn(
      async (_patch: Partial<ChatSettingsRecord>) => null
    )
    renderPicker(
      {
        useCharacterDefault: false,
        greetingSelectionId: "greeting:1:selected",
        greetingsChecksum: checksum
      },
      updateSettings
    )

    const select = screen.getByTestId("greeting-select") as HTMLSelectElement
    expect(select.value).toBe(alternateGreetingId)
  })

  it("selects the current greeting as the first character message", async () => {
    const updateSettings = vi.fn(
      async (_patch: Partial<ChatSettingsRecord>) => null
    )
    vi.mocked(tldwClient.initialize).mockResolvedValue(null)
    vi.mocked(tldwClient.addChatMessage).mockResolvedValue({
      id: "server-message-1",
      version: 3
    } as any)
    let messageState: Message[] = []
    let historyState: ChatHistory = []
    const setMessages = vi.fn(
      (next: Message[] | ((prev: Message[]) => Message[])) => {
        messageState =
          typeof next === "function" ? next(messageState) : next
      }
    )
    const setHistory = vi.fn(
      (next: ChatHistory | ((prev: ChatHistory) => ChatHistory)) => {
        historyState =
          typeof next === "function" ? next(historyState) : next
      }
    )

    renderPicker(
      {
        useCharacterDefault: false,
        greetingSelectionId: alternateGreetingId,
        greetingsChecksum: checksum,
        greetingEnabled: true
      },
      updateSettings,
      {
        serverChatId: "server-chat-1",
        setMessages,
        setHistory
      }
    )

    fireEvent.click(screen.getByRole("button", { name: /select greeting/i }))

    await waitFor(() => {
      expect(messageState[0]).toEqual(
        expect.objectContaining({
          isBot: true,
          role: "assistant",
          name: "Guide",
          message: "Good to see you",
          messageType: "character:greeting",
          serverMessageId: "server-message-1",
          serverMessageVersion: 3
        })
      )
    })
    expect(historyState).toEqual([
      expect.objectContaining({
        role: "assistant",
        content: "Good to see you",
        messageType: "character:greeting"
      })
    ])
    expect(tldwClient.addChatMessage).toHaveBeenCalledWith("server-chat-1", {
      role: "assistant",
      content: "Good to see you"
    })
  })

  it("does not persist the same greeting twice during a rapid double select", async () => {
    const updateSettings = vi.fn(
      async (_patch: Partial<ChatSettingsRecord>) => null
    )
    vi.mocked(tldwClient.initialize).mockResolvedValue(null)
    let resolveAdd: (value: unknown) => void = () => {}
    vi.mocked(tldwClient.addChatMessage).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveAdd = resolve
        }) as any
    )
    let messageState: Message[] = []
    let historyState: ChatHistory = []
    const setMessages = vi.fn(
      (next: Message[] | ((prev: Message[]) => Message[])) => {
        messageState =
          typeof next === "function" ? next(messageState) : next
      }
    )
    const setHistory = vi.fn(
      (next: ChatHistory | ((prev: ChatHistory) => ChatHistory)) => {
        historyState =
          typeof next === "function" ? next(historyState) : next
      }
    )

    renderPicker(
      {
        useCharacterDefault: false,
        greetingSelectionId: alternateGreetingId,
        greetingsChecksum: checksum,
        greetingEnabled: true
      },
      updateSettings,
      {
        serverChatId: "server-chat-1",
        setMessages,
        setHistory
      }
    )

    const button = screen.getByRole("button", { name: /select greeting/i })
    fireEvent.click(button)
    fireEvent.click(button)

    await waitFor(() => {
      expect(tldwClient.addChatMessage).toHaveBeenCalledTimes(1)
    })

    resolveAdd({ id: "server-message-1", version: 3 })
    await waitFor(() => {
      expect(messageState).toHaveLength(1)
    })
    expect(historyState).toHaveLength(1)
  })

  it("notifies when a selected greeting cannot be synced to the server chat", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    vi.mocked(tldwClient.addChatMessage).mockRejectedValueOnce(
      new Error("server unavailable")
    )
    const updateSettings = vi.fn(
      async (patch: Partial<ChatSettingsRecord>) =>
        normalizeChatSettingsRecord({
          greetingSelectionId: patch.greetingSelectionId,
          greetingsChecksum: patch.greetingsChecksum,
          greetingEnabled: true
        })
    )
    let messageState: Message[] = []
    let historyState: ChatHistory = []
    const setMessages = vi.fn((next: Message[] | ((prev: Message[]) => Message[])) => {
      messageState = typeof next === "function" ? next(messageState) : next
    })
    const setHistory = vi.fn((next: ChatHistory | ((prev: ChatHistory) => ChatHistory)) => {
      historyState = typeof next === "function" ? next(historyState) : next
    })

    try {
      renderPicker(
        {
          greetingSelectionId: alternateGreetingId,
          greetingsChecksum: checksum,
          greetingEnabled: true
        },
        updateSettings,
        {
          serverChatId: "server-chat-1",
          setMessages,
          setHistory
        }
      )

      fireEvent.click(screen.getByRole("button", { name: /select greeting/i }))

      await waitFor(() => {
        expect(notificationErrorMock).toHaveBeenCalledWith(
          expect.objectContaining({
            message: "Greeting sync failed",
            description:
              "The greeting was added locally but could not be saved to the server chat."
          })
        )
      })
      expect(messageState[0]).toEqual(
        expect.objectContaining({
          message: "Good to see you",
          messageType: "character:greeting"
        })
      )
      expect(historyState[0]).toEqual(
        expect.objectContaining({
          content: "Good to see you",
          messageType: "character:greeting"
        })
      )
    } finally {
      warnSpy.mockRestore()
    }
  })
})
