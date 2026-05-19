import React from "react"
import { render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => {
  const useQuery = vi.fn()
  const setSelectedAssistant = vi.fn(async () => undefined)
  const setSelectedCharacter = vi.fn(async () => undefined)
  const initialize = vi.fn(async () => null)
  const listPersonaProfiles = vi.fn(async () => [])

  return {
    useQuery,
    setSelectedAssistant,
    setSelectedCharacter,
    initialize,
    listPersonaProfiles
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: mocks.useQuery
}))

vi.mock("@plasmohq/storage/hook", async () => {
  const ReactModule =
    await vi.importActual<typeof import("react")>("react")

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
  const React = await import("react")

  const Avatar = ({
    src,
    className
  }: {
    src?: string
    className?: string
  }) =>
    src ? (
      <img
        alt="assistant avatar"
        src={src}
        className={className}
        data-testid="persona-avatar"
      />
    ) : null

  const Dropdown = ({ children }: { children: React.ReactNode }) => <>{children}</>
  const Tooltip = ({ children }: { children: React.ReactNode }) => <>{children}</>
  const Input = React.forwardRef<HTMLInputElement, any>((props, ref) => (
    <input ref={ref} {...props} />
  ))
  const Select = () => null
  const Empty = () => null

  return {
    Avatar,
    Dropdown,
    Empty,
    Input,
    Select,
    Tooltip
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => mocks.initialize(...args),
    listPersonaProfiles: (...args: unknown[]) =>
      mocks.listPersonaProfiles(...args)
  }
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: {
      hasCharacters: true,
      hasPersona: true
    }
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
  useSelectedCharacter: () => [null, mocks.setSelectedCharacter]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [
    {
      kind: "persona",
      id: "persona-1",
      name: "Guide Persona"
    },
    mocks.setSelectedAssistant
  ]
}))

vi.mock("@/hooks/chat/useClearChat", () => ({
  useClearChat: () => vi.fn()
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: { messages: never[]; serverChatId: string | null }) => unknown) =>
    selector({
      messages: [],
      serverChatId: null
    })
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

import CharacterSelect from "../CharacterSelect"

describe("CharacterSelect persona avatar", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.useQuery.mockImplementation(({ queryKey }: { queryKey: string[] }) => {
      if (queryKey[0] === "persona-profiles") {
        return {
          data: [
            {
              id: "persona-1",
              name: "Guide Persona",
              avatar_url: "https://example.com/guide.png"
            }
          ]
        }
      }

      return {
        data: [],
        isLoading: false,
        refetch: vi.fn()
      }
    })
  })

  it("renders the selected persona avatar from current persona data in the trigger button", () => {
    render(
      <CharacterSelect
        selectedCharacterId={null}
        setSelectedCharacterId={vi.fn()}
      />
    )

    const trigger = screen.getByTestId("chat-character-select")
    const avatar = within(trigger).getByTestId("persona-avatar")

    expect(avatar).toHaveAttribute("src", "https://example.com/guide.png")
  })

  it("lets persona profile loading errors reject instead of caching an empty catalog", async () => {
    const queryConfigs: Array<{
      queryKey: string[]
      queryFn?: () => Promise<unknown>
    }> = []

    mocks.listPersonaProfiles.mockRejectedValueOnce(
      new Error("persona catalog unavailable")
    )
    mocks.useQuery.mockImplementation((config: { queryKey: string[] }) => {
      queryConfigs.push(config)
      return {
        data: [],
        isLoading: false,
        refetch: vi.fn()
      }
    })

    render(
      <CharacterSelect
        selectedCharacterId={null}
        setSelectedCharacterId={vi.fn()}
      />
    )

    const personaQuery = queryConfigs.find(
      (config) => config.queryKey[0] === "persona-profiles"
    )

    expect(personaQuery?.queryFn).toBeTypeOf("function")
    await expect(personaQuery!.queryFn!()).rejects.toThrow(
      "persona catalog unavailable"
    )
  })
})
