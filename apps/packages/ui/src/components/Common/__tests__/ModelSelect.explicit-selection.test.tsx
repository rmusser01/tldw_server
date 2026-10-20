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
const discovery = vi.hoisted(() => ({ failed: false, retry: vi.fn() }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))
vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => {
    const [selectedModel, setSelectedModel] = React.useState<string | null>(
      "tldw:Gemma:Q4"
    )
    return {
      selectedModel,
      setSelectedModel: (value: string | null) => {
        selection(value)
        setSelectedModel(value)
        return Promise.resolve(true)
      },
      selectedModelSaveFailed: false,
      retrySelectedModelSave: () => Promise.resolve(false)
    }
  }
}))
vi.mock("@tanstack/react-query", async (original) => {
  const actual = await original<typeof import("@tanstack/react-query")>()
  const queryClient = new actual.QueryClient()
  return {
    ...actual,
    useQueryClient: () => queryClient,
    useQuery: () => ({
      data: discovery.failed ? undefined : [
        { model: "tldw:Gemma:Q4", provider: "openai", nickname: "Hosted Gemma" },
        { model: "tldw:Gemma:Q4", provider: "local-llm", nickname: "Local Gemma" },
        { model: "tldw:llama3.2:3b", provider: "ollama" }
      ],
      isLoading: false,
      isSuccess: !discovery.failed,
      isError: discovery.failed,
      isFetching: false,
      refetch: discovery.retry
    })
  }
})
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
        .filter((item: MenuItem) => item.children)
        .map((item: MenuItem) => (
          <div data-testid="menu-group" key={`group-${item.key}`}>
            {item.label}
          </div>
        ))}
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
    discovery.failed = false
    discovery.retry.mockClear()
    useStoreChatModelSettings.getState().reset()
  })
  it("offers a keyboard accessible Retry when discovery fails", () => {
    discovery.failed = true
    render(<ModelSelect />)
    const retry = screen.getByRole("button", { name: "Could not check chat models. Try again" })
    fireEvent.click(retry)
    expect(discovery.retry).toHaveBeenCalledOnce()
    expect(selection).not.toHaveBeenCalled()
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

  it("XS-02: rows name the model first and the provider second", () => {
    render(<ModelSelect showSelectedName />)
    const hosted = screen.getByRole("menuitem", { name: /hosted gemma/i })
    expect(hosted.textContent ?? "").toMatch(/^Hosted Gemma/)
    expect(hosted).toHaveTextContent("OpenAI")
    expect(hosted.textContent).not.toContain(" - ")
  })

  it("XS-02: groups a local provider under its own name, not 'Custom'", () => {
    render(<ModelSelect showSelectedName />)
    const groups = screen.getAllByTestId("menu-group").map((group) => group.textContent)
    expect(groups).toContain("Local · Ollama")
    expect(groups).not.toContain("Custom")
  })

  it("XS-02: the chip names the selected model without the internal prefix", () => {
    render(<ModelSelect showSelectedName />)
    const chip = screen.getByTestId("chat-model-select")
    expect(chip).toHaveTextContent("Gemma:Q4")
    expect(chip.textContent).not.toContain("tldw:")
    expect(chip).toHaveAccessibleName(expect.stringContaining("Gemma:Q4"))
  })
})
