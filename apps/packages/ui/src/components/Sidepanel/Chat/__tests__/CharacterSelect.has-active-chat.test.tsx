// @vitest-environment jsdom
import React from "react"
import { act, render } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

// Real zustand store so the derived-boolean selector semantics (re-render only
// when the boolean flips) are exercised end to end.
import { useStoreMessageOption } from "@/store/option"

const dropdownRenders = vi.hoisted(() => ({ count: 0 }))
const useQueryMock = vi.hoisted(() => vi.fn())

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) => {
      if (typeof fallback === "string") return fallback
      if (fallback?.defaultValue) return fallback.defaultValue
      return key
    }
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: (...args: unknown[]) => useQueryMock(...args),
  useQueryClient: () => ({ invalidateQueries: vi.fn() })
}))

vi.mock("@plasmohq/storage/hook", async () => {
  const ReactModule = await vi.importActual<typeof import("react")>("react")
  return {
    useStorage: (_key: string, initialValue: unknown) =>
      ReactModule.useState(initialValue)
  }
})

vi.mock("wxt/browser", () => ({
  browser: {
    tabs: {
      create: vi.fn(async () => undefined)
    }
  }
}))

vi.mock("antd", async () => {
  const ReactModule = await vi.importActual<typeof import("react")>("react")

  // Render counter: the antd Dropdown mock re-renders exactly once per
  // CharacterSelect render (its children element is recreated each render),
  // so its invocation count observes CharacterSelect re-renders.
  const Dropdown = ({ children }: { children?: React.ReactNode }) => {
    dropdownRenders.count += 1
    return <>{children ?? null}</>
  }
  const Tooltip = ({ children }: { children?: React.ReactNode }) => (
    <>{children ?? null}</>
  )
  type MockInputProps = React.InputHTMLAttributes<HTMLInputElement> & {
    allowClear?: boolean
    prefix?: React.ReactNode
  }
  const Input = ReactModule.forwardRef<HTMLInputElement, MockInputProps>(
    ({ allowClear: _allowClear, prefix: _prefix, ...props }, ref) => (
      <input ref={ref} {...props} />
    )
  )
  const Select = () => null
  const Empty = () => null
  const Avatar = () => null

  return { Avatar, Dropdown, Empty, Input, Select, Tooltip }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => null),
    listPersonaProfiles: vi.fn(async () => []),
    fetchWithAuth: vi.fn(async () => ({ ok: false, json: async () => [], text: async () => "" }))
  }
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasCharacters: true, hasPersona: true }
  })
}))

vi.mock("@/utils/character-greetings", () => ({
  collectGreetings: vi.fn(() => []),
  pickGreeting: vi.fn(() => null)
}))

vi.mock("@/utils/character-mood", () => ({
  CHARACTER_MOOD_OPTIONS: [],
  getCharacterMoodImagesFromExtensions: vi.fn(() => ({})),
  removeCharacterMoodImage: vi.fn(),
  upsertCharacterMoodImage: vi.fn()
}))

vi.mock("@/utils/message-steering", () => ({
  DEFAULT_MESSAGE_STEERING_PROMPTS: {
    continueAsUser: "",
    impersonateUser: "",
    forceNarrate: ""
  },
  MESSAGE_STEERING_PROMPTS_STORAGE_KEY: "messageSteeringPrompts",
  normalizeMessageSteeringPrompts: vi.fn((value) => value)
}))

vi.mock("@/components/Common/MyChatIdentityMenu", () => ({
  MyChatIdentityMenu: () => null
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  })
}))

vi.mock("@/hooks/useAntdModal", () => ({
  useAntdModal: () => ({
    confirm: vi.fn(() => ({
      destroy: vi.fn(),
      update: vi.fn()
    }))
  })
}))

vi.mock("@/hooks/useConfirmModal", () => ({
  useConfirmModal: () => vi.fn(async () => true)
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, vi.fn(async () => undefined)]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, vi.fn(async () => undefined)]
}))

vi.mock("@/hooks/chat/useClearChat", () => ({
  useClearChat: () => vi.fn()
}))

vi.mock("@/utils/browser-runtime", () => ({
  getBrowserRuntime: vi.fn(() => null),
  isExtensionRuntime: vi.fn(() => false)
}))

vi.mock("@/utils/characters-route", () => ({
  buildCharactersHash: vi.fn(() => "#/characters"),
  buildCharactersRoute: vi.fn(() => "/characters"),
  resolveCharactersDestinationMode: vi.fn(() => "tab")
}))

import SidepanelCharacterSelect from "../CharacterSelect"
import { CharacterSelect as CommonCharacterSelect } from "@/components/Common/CharacterSelect"

const greetingMessage = {
  id: "m-greeting",
  message: "Hello there",
  messageType: "character:greeting",
  isBot: true
}
const userMessage = {
  id: "m-user",
  message: "user asks something",
  isBot: false
}

const resetStore = () => {
  useStoreMessageOption.setState({
    messages: [] as never[],
    serverChatId: null
  })
}

describe.each([
  {
    label: "Sidepanel/Chat/CharacterSelect",
    mount: () =>
      render(
        <SidepanelCharacterSelect
          selectedCharacterId={null}
          setSelectedCharacterId={vi.fn()}
        />
      )
  },
  {
    label: "Common/CharacterSelect",
    mount: () => render(<CommonCharacterSelect />)
  }
])("$label derived-boolean selector (WP-06)", ({ mount }) => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetStore()
    dropdownRenders.count = 0
    useQueryMock.mockImplementation(() => ({
      data: [],
      isLoading: false,
      refetch: vi.fn()
    }))
  })

  it("does not re-render on last-message text changes and re-renders only when hasActiveChat flips", async () => {
    useStoreMessageOption.setState({
      messages: [greetingMessage] as never[],
      serverChatId: null
    })

    const view = mount()
    await act(async () => {
      await Promise.resolve()
    })
    expect(view.container.textContent ?? "").toBeDefined()
    const baseline = dropdownRenders.count
    expect(baseline).toBeGreaterThan(0)

    // Greeting-only transcript text change: hasActiveChat stays false.
    act(() => {
      useStoreMessageOption.setState({
        messages: [{ ...greetingMessage, message: "a much longer greeting" }] as never[]
      })
    })
    expect(dropdownRenders.count).toBe(baseline)

    // A real user message flips the boolean false -> true: re-render.
    act(() => {
      useStoreMessageOption.setState({
        messages: [greetingMessage, userMessage] as never[]
      })
    })
    expect(dropdownRenders.count).toBeGreaterThan(baseline)

    // Last-message text change (streaming-flush shape: replaced object in a
    // new array): hasActiveChat stays true, no re-render.
    const withUser = dropdownRenders.count
    act(() => {
      useStoreMessageOption.setState({
        messages: [
          greetingMessage,
          { ...userMessage, message: "user asks a much longer question" }
        ] as never[]
      })
    })
    expect(dropdownRenders.count).toBe(withUser)

    // Clearing the transcript flips the boolean back to false: re-render.
    act(() => {
      useStoreMessageOption.setState({ messages: [] as never[] })
    })
    expect(dropdownRenders.count).toBeGreaterThan(withUser)
  })
})
