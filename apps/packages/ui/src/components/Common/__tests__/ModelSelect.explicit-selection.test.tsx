import { validateNativeCharacterConfiguration } from "@/hooks/chat/native-character-preflight"
import { useStoreChatModelSettings } from "@/store/model"
import { fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ModelSelect } from "../ModelSelect"

type MenuItem = {
  key: string
  label: React.ReactNode
  onClick?: () => void
  children?: MenuItem[]
}

const selection = vi.hoisted(() => vi.fn())
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))
vi.mock("@/hooks/useMessage", () => ({
  useMessage: () => {
    const [selectedModel, setSelectedModel] = React.useState<string | null>(
      "tldw:Gemma:Q4"
    )
    return {
      selectedModel,
      setSelectedModel: (value: string | null) => {
        selection(value)
        setSelectedModel(value)
      }
    }
  }
}))
vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({
    data: [
      { model: "tldw:Gemma:Q4", provider: "openai", nickname: "Hosted Gemma" },
      { model: "tldw:Gemma:Q4", provider: "local-llm", nickname: "Local Gemma" }
    ],
    isLoading: false
  })
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, initial: unknown) => React.useState(initial)
}))
vi.mock("@/services/tldw-server", () => ({ fetchChatModels: vi.fn() }))
vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: () => null
}))
vi.mock("antd", () => ({
  Avatar: () => null,
  Input: () => null,
  Select: () => null,
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Dropdown: ({
    children,
    menu
  }: {
    children: React.ReactNode
    menu: { selectedKeys: string[]; items: MenuItem[] }
  }) => (
    <div>
      {children}
      <output data-testid="selected-menu-key">
        {menu.selectedKeys.join(",")}
      </output>
      {menu.items
        .flatMap((item: MenuItem) => item.children ?? [item])
        .map((item: MenuItem) => (
          <div role="menuitem" key={item.key} onClick={item.onClick}>
            {item.label}
          </div>
        ))}
    </div>
  )
}))

describe("compact explicit model selection", () => {
  beforeEach(() => {
    selection.mockClear()
    useStoreChatModelSettings.getState().reset()
  })
  it("commits the chosen provider with duplicate model IDs and retains a repeated choice", () => {
    render(<ModelSelect showSelectedName />)
    expect(screen.getByTestId("selected-menu-key")).toHaveTextContent(/^$/)
    const choice = screen.getByRole("menuitem", { name: /local gemma/i })
    fireEvent.click(choice)
    expect(selection).toHaveBeenLastCalledWith("local:Gemma:Q4")
    expect(screen.getByTestId("selected-menu-key")).toHaveTextContent(
      "local:Gemma:Q4"
    )
    expect(
      validateNativeCharacterConfiguration({
        temporary: false,
        clientToolCount: 0,
        settings: {},
        model: selection.mock.lastCall![0]
      })
    ).toMatchObject({ provider: "local-llm", model: "Gemma:Q4" })
    fireEvent.click(choice)
    expect(selection.mock.calls).toEqual([
      ["local:Gemma:Q4"],
      ["local:Gemma:Q4"]
    ])
  })
})
