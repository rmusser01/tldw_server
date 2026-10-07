/**
 * CC-05 (#3112): the chat Prompt picker offers the user's server prompt
 * library next to the prompts on this device.
 *
 * bgRequest is answered by a fake of the real endpoints (see
 * services/__tests__/server-prompts-api-fixture.ts), so paging, params and
 * fields follow the backend's PaginatedPromptsResponse / PromptResponse.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  buildFakeServerPrompt,
  createFakePromptsApi,
  type FakeServerPrompt
} from "@/services/__tests__/server-prompts-api-fixture"
import { PromptSelect } from "../PromptSelect"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  getAllPrompts: vi.fn(async () => [] as unknown[]),
  getPromptById: vi.fn(async () => undefined),
  upsertServerLibraryPromptCopy: vi.fn()
}))

vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next")
  const i18n = createInstance()
  await i18n.init({ lng: "en", resources: {}, interpolation: { escapeValue: false } })
  return {
    useTranslation: () => ({
      t: (...args: Parameters<typeof i18n.t>) => i18n.t(...args)
    })
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

vi.mock("@/db/dexie/helpers", () => ({
  updatePrompt: vi.fn(),
  markPromptSyncError: vi.fn(),
  getAllPrompts: mocks.getAllPrompts,
  getPromptById: mocks.getPromptById,
  upsertServerLibraryPromptCopy: (...args: unknown[]) =>
    mocks.upsertServerLibraryPromptCopy(...args)
}))

type MockInputProps = React.InputHTMLAttributes<HTMLInputElement> & {
  "aria-label"?: string
}
type MockTextAreaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement> & {
  "aria-label"?: string
}
type MockMenuItem = {
  key?: React.Key
  type?: string
  label?: React.ReactNode
  children?: MockMenuItem[]
  onClick?: () => void
} | null
type MockDropdownProps = {
  open?: boolean
  onOpenChange?: (open: boolean) => void
  menu?: { items?: MockMenuItem[] }
  popupRender?: (menu: React.ReactNode) => React.ReactNode
  children?: React.ReactNode
}

vi.mock("antd", async () => {
  const React = await import("react")

  const TextArea = React.forwardRef<HTMLTextAreaElement, MockTextAreaProps>((props, ref) => (
    <textarea ref={ref} aria-label={props["aria-label"] ?? "System prompt"} value={props.value} onChange={props.onChange} />
  ))
  const Input = Object.assign(
    React.forwardRef<HTMLInputElement, MockInputProps>((props, ref) => (
      <input
        ref={ref}
        aria-label={props["aria-label"] ?? props.placeholder}
        value={props.value}
        onChange={props.onChange}
        onKeyDown={props.onKeyDown}
      />
    )),
    { TextArea }
  )

  const renderMenuItems = (items: MockMenuItem[] = []): React.ReactNode =>
    items.map((item) => {
      if (!item) return null
      if (item.type === "group") {
        return (
          <div key={String(item.key ?? item.label)} role="group" aria-label={typeof item.label === "string" ? item.label : undefined}>
            <div>{item.label}</div>
            {renderMenuItems(item.children)}
          </div>
        )
      }
      if (item.type === "divider") return <hr key={item.key} />
      if (item.key === "empty") return <div key="empty">{item.label}</div>
      return (
        <button key={item.key} type="button" role="menuitem" onClick={() => item.onClick?.()}>
          {item.label}
        </button>
      )
    })

  const Dropdown = ({ open, onOpenChange, menu, popupRender, children }: MockDropdownProps) => {
    const menuNode = <div role="menu">{renderMenuItems(menu?.items)}</div>
    return (
      <div>
        <div onClick={() => onOpenChange?.(!open)}>{children}</div>
        {open ? (popupRender ? popupRender(menuNode) : menuNode) : null}
      </div>
    )
  }

  return {
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Dropdown,
    Empty: ({ description }: { description?: React.ReactNode }) => (
      <div>{description ?? "Empty"}</div>
    ),
    Input,
    Modal: ({ open, children }: { open?: boolean; children?: React.ReactNode }) =>
      open ? <div role="dialog">{children}</div> : null
  }
})

const localPrompt = (overrides: Record<string, unknown> = {}) => ({
  id: "local-1",
  title: "Local quick prompt",
  content: "Summarise this",
  is_system: false,
  createdAt: 1,
  ...overrides
})

const serveServerPrompts = (prompts: FakeServerPrompt[]) => {
  const api = createFakePromptsApi(prompts)
  mocks.bgRequest.mockImplementation(async (request) => api.handle(request) ?? {})
  return api
}

const renderPromptSelect = () => {
  const props = {
    selectedSystemPrompt: undefined as string | undefined,
    systemPrompt: "",
    setSystemPrompt: vi.fn(),
    setSelectedSystemPrompt: vi.fn(),
    setSelectedQuickPrompt: vi.fn()
  }
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <PromptSelect {...props} />
    </QueryClientProvider>
  )
  return props
}

const openPicker = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(await screen.findByRole("button", { name: "selectAPrompt" }))
  return screen.findByRole("menu")
}

describe("PromptSelect server prompt library (CC-05 #3112)", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getAllPrompts.mockResolvedValue([localPrompt()])
    serveServerPrompts([])
  })

  it("lists every server prompt across pages next to local prompts", async () => {
    const api = serveServerPrompts(
      Array.from({ length: 120 }, (_, index) => buildFakeServerPrompt(index + 1))
    )
    const user = userEvent.setup()
    renderPromptSelect()

    await openPicker(user)

    const serverGroup = await screen.findByRole("group", { name: "Server library" })
    expect(await within(serverGroup).findByText("Server prompt 001")).toBeInTheDocument()
    expect(within(serverGroup).getByText("Server prompt 120")).toBeInTheDocument()
    expect(within(serverGroup).getAllByRole("menuitem")).toHaveLength(120)
    expect(screen.getByText("Local quick prompt")).toBeInTheDocument()
    expect(api.listRequests.map((params) => params.get("page"))).toEqual(["1", "2"])
  })

  it("shows a synced copy once and labels where each prompt lives", async () => {
    mocks.getAllPrompts.mockResolvedValue([
      localPrompt(),
      localPrompt({ id: "local-2", title: "daily standup", content: "Standup notes" }),
      localPrompt({
        id: "local-3",
        title: "Renamed on device",
        content: "Tutor",
        is_system: true,
        serverLibraryUuid: buildFakeServerPrompt(3).uuid
      })
    ])
    serveServerPrompts([
      buildFakeServerPrompt(2, { name: "Daily standup" }),
      buildFakeServerPrompt(3, { name: "Socratic tutor" }),
      buildFakeServerPrompt(4, { name: "Server only" })
    ])
    const user = userEvent.setup()
    renderPromptSelect()

    await openPicker(user)

    const serverGroup = await screen.findByRole("group", { name: "Server library" })
    expect(within(serverGroup).getAllByRole("menuitem")).toHaveLength(1)
    expect(within(serverGroup).getByText("Server only")).toBeInTheDocument()
    expect(within(serverGroup).getByText("Server")).toBeInTheDocument()
    expect(screen.queryByText("Daily standup")).not.toBeInTheDocument()
    expect(screen.queryByText("Socratic tutor")).not.toBeInTheDocument()
    const standup = screen.getByRole("menuitem", { name: /daily standup/i })
    expect(within(standup).getByText("Local + server")).toBeInTheDocument()
    const renamed = screen.getByRole("menuitem", { name: /Renamed on device/ })
    expect(within(renamed).getByText("Local + server")).toBeInTheDocument()
    const localOnly = screen.getByRole("menuitem", { name: /Local quick prompt/ })
    expect(within(localOnly).queryByText(/server/i)).not.toBeInTheDocument()
  })

  it("applies a server quick prompt exactly like a local quick prompt", async () => {
    const api = serveServerPrompts([
      buildFakeServerPrompt(5, { name: "Rewrite plainly", user_prompt: "Rewrite this plainly" })
    ])
    const user = userEvent.setup()
    const props = renderPromptSelect()

    await openPicker(user)
    await user.click(await screen.findByRole("menuitem", { name: /Local quick prompt/ }))
    const localCalls = {
      system: props.setSelectedSystemPrompt.mock.calls,
      quick: props.setSelectedQuickPrompt.mock.calls
    }
    expect(localCalls).toEqual({ system: [[undefined]], quick: [["Summarise this"]] })
    props.setSelectedSystemPrompt.mockClear()
    props.setSelectedQuickPrompt.mockClear()

    await openPicker(user)
    await user.click(await screen.findByRole("menuitem", { name: /Rewrite plainly/ }))

    await waitFor(() =>
      expect(props.setSelectedQuickPrompt).toHaveBeenCalledWith("Rewrite this plainly")
    )
    expect(props.setSelectedSystemPrompt.mock.calls).toEqual([[undefined]])
    expect(api.detailRequests).toEqual(["5"])
    expect(mocks.upsertServerLibraryPromptCopy).not.toHaveBeenCalled()
    await waitFor(() => expect(screen.queryByRole("menu")).not.toBeInTheDocument())
  })

  it("applies a server system prompt through a local copy so chat can resolve it by id", async () => {
    serveServerPrompts([
      buildFakeServerPrompt(6, {
        name: "Socratic tutor",
        system_prompt: "Ask guiding questions.",
        user_prompt: null,
        keywords: ["teaching"]
      })
    ])
    mocks.upsertServerLibraryPromptCopy.mockResolvedValue({
      id: "copy-6",
      title: "Socratic tutor",
      content: "Ask guiding questions.",
      is_system: true,
      createdAt: 2
    })
    const user = userEvent.setup()
    const props = renderPromptSelect()

    await openPicker(user)
    await user.click(await screen.findByRole("menuitem", { name: /Socratic tutor/ }))

    await waitFor(() =>
      expect(props.setSelectedSystemPrompt).toHaveBeenCalledWith("copy-6")
    )
    expect(mocks.upsertServerLibraryPromptCopy).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 6,
        uuid: buildFakeServerPrompt(6).uuid,
        name: "Socratic tutor",
        system_prompt: "Ask guiding questions.",
        keywords: ["teaching"]
      })
    )
    expect(props.setSelectedQuickPrompt).not.toHaveBeenCalled()
  })

  it("keeps the picker open with a retryable error when a server prompt cannot be loaded", async () => {
    let failDetail = true
    const api = createFakePromptsApi([
      buildFakeServerPrompt(7, { name: "Flaky prompt", user_prompt: "Flaky text" })
    ])
    mocks.bgRequest.mockImplementation(async (request: { path?: string }) => {
      if (failDetail && /\/api\/v1\/prompts\/7$/.test(String(request.path))) {
        throw new Error("Request failed: 500")
      }
      return api.handle(request) ?? {}
    })
    const user = userEvent.setup()
    const props = renderPromptSelect()

    await openPicker(user)
    await user.click(await screen.findByRole("menuitem", { name: /Flaky prompt/ }))

    expect(
      await screen.findByText("Couldn't load this prompt from the server. Select it to retry.")
    ).toBeInTheDocument()
    expect(screen.getByRole("menu")).toBeInTheDocument()
    expect(props.setSelectedQuickPrompt).not.toHaveBeenCalled()

    failDetail = false
    await user.click(screen.getByRole("menuitem", { name: /Flaky prompt/ }))
    await waitFor(() =>
      expect(props.setSelectedQuickPrompt).toHaveBeenCalledWith("Flaky text")
    )
  })

  it("shows a loading row for the server library without hiding local prompts", async () => {
    let resolveList: (value: unknown) => void = () => {}
    mocks.bgRequest.mockImplementation(
      () => new Promise((resolve) => { resolveList = resolve })
    )
    const user = userEvent.setup()
    renderPromptSelect()

    await openPicker(user)

    expect(await screen.findByRole("status", { name: "Loading server prompts" })).toBeInTheDocument()
    expect(screen.getByText("Local quick prompt")).toBeInTheDocument()
    expect(screen.queryByText("No saved prompts")).not.toBeInTheDocument()

    resolveList(createFakePromptsApi([buildFakeServerPrompt(8)]).handle({ path: "/api/v1/prompts?page=1&per_page=100" }))
    expect(await screen.findByText("Server prompt 008")).toBeInTheDocument()
    expect(screen.queryByRole("status", { name: "Loading server prompts" })).not.toBeInTheDocument()
  })

  it("reports an unavailable server library and retries it", async () => {
    let fail = true
    const api = createFakePromptsApi([buildFakeServerPrompt(9, { name: "Recovered prompt" })])
    mocks.bgRequest.mockImplementation(async (request) => {
      if (fail) throw new Error("Request failed: 503")
      return api.handle(request) ?? {}
    })
    const user = userEvent.setup()
    renderPromptSelect()

    await openPicker(user)

    expect(await screen.findByText("Server prompts unavailable")).toBeInTheDocument()
    expect(screen.getByText("Local quick prompt")).toBeInTheDocument()
    fail = false
    await user.click(screen.getByRole("menuitem", { name: "Retry server prompts" }))
    expect(await screen.findByText("Recovered prompt")).toBeInTheDocument()
    expect(screen.queryByText("Server prompts unavailable")).not.toBeInTheDocument()
  })

  it("stays quiet about the server library when no server is configured", async () => {
    mocks.getAllPrompts.mockResolvedValue([])
    mocks.bgRequest.mockRejectedValue(new Error("tldw server not configured"))
    const user = userEvent.setup()
    renderPromptSelect()

    await openPicker(user)

    expect(await screen.findByText("No saved prompts")).toBeInTheDocument()
    expect(screen.queryByText("Server prompts unavailable")).not.toBeInTheDocument()
  })

  it("does not claim there are no prompts when only the server has them", async () => {
    mocks.getAllPrompts.mockResolvedValue([])
    serveServerPrompts([buildFakeServerPrompt(10, { name: "Only on server" })])
    const user = userEvent.setup()
    renderPromptSelect()

    await openPicker(user)

    expect(await screen.findByText("Only on server")).toBeInTheDocument()
    expect(screen.queryByText("No saved prompts")).not.toBeInTheDocument()
  })

  it("says the library is empty only once both sources are empty", async () => {
    mocks.getAllPrompts.mockResolvedValue([])
    const api = serveServerPrompts([])
    const user = userEvent.setup()
    renderPromptSelect()

    await openPicker(user)

    expect(await screen.findByText("No saved prompts")).toBeInTheDocument()
    expect(api.listRequests).toHaveLength(1)
  })

  it("explains an empty device library when the server library is unavailable", async () => {
    mocks.getAllPrompts.mockResolvedValue([])
    mocks.bgRequest.mockRejectedValue(new Error("Request failed: 503"))
    const user = userEvent.setup()
    renderPromptSelect()

    await openPicker(user)

    expect(await screen.findByText("Server prompts unavailable")).toBeInTheDocument()
    expect(screen.getByText("No prompts on this device")).toBeInTheDocument()
    expect(screen.queryByText("No saved prompts")).not.toBeInTheDocument()
  })

  it("searches server prompts by name", async () => {
    serveServerPrompts([
      buildFakeServerPrompt(11, { name: "Socratic tutor" }),
      buildFakeServerPrompt(12, { name: "Release notes" })
    ])
    const user = userEvent.setup()
    renderPromptSelect()

    await openPicker(user)
    await screen.findByText("Release notes")
    await user.type(screen.getByRole("textbox", { name: "Search prompts..." }), "socratic")

    expect(await screen.findByText("Socratic tutor")).toBeInTheDocument()
    expect(screen.queryByText("Release notes")).not.toBeInTheDocument()
    expect(screen.queryByText("No matching prompts")).not.toBeInTheDocument()
  })
})
