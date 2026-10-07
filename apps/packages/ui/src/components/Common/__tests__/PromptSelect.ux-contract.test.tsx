/**
 * UX review 2026-10 contract reproduction CC-05 (#3112).
 *
 * The bgRequest mock answers like the real prompts API
 * (tldw_Server_API/app/api/v1/endpoints/prompts.py list_all_prompts returns a
 * PaginatedPromptsResponse of PromptBriefResponse items). The test asserts the
 * CORRECT behaviour; it was an `it.fails` reproduction until the fix landed and
 * now guards against regression (fuller coverage:
 * PromptSelect.server-library.test.tsx).
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PromptSelect } from "../PromptSelect"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  getAllPrompts: vi.fn(async () => [] as unknown[]),
  getPromptById: vi.fn(async () => undefined)
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
  getPromptById: mocks.getPromptById
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
          <div key={String(item.key ?? item.label)}>
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

const SERVER_PROMPT_NAME = "Server research brief"

/** PaginatedPromptsResponse from GET /api/v1/prompts (schemas/prompt_schemas.py:363). */
const serverPromptList = {
  items: [
    {
      id: 7,
      uuid: "6f1f3a52-2f0e-4b8e-9d55-1b7f0f6d2a11",
      name: SERVER_PROMPT_NAME,
      author: "server",
      last_modified: "2026-09-30T12:00:00Z",
      usage_count: 3,
      last_used_at: null
    }
  ],
  total_pages: 1,
  current_page: 1,
  total_items: 1,
  pagination: { mode: "page", page: 1, per_page: 10, total: 1, total_pages: 1, has_more: false }
}

const renderPromptSelect = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <PromptSelect
        selectedSystemPrompt={undefined}
        systemPrompt=""
        setSystemPrompt={vi.fn()}
        setSelectedSystemPrompt={vi.fn()}
        setSelectedQuickPrompt={vi.fn()}
      />
    </QueryClientProvider>
  )

describe("PromptSelect UX contract reproductions (#3112)", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getAllPrompts.mockResolvedValue([
      { id: "local-1", title: "Local quick prompt", content: "Summarise this", is_system: false, createdAt: 1 }
    ])
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request?.path || "")
      const method = String(request?.method || "GET").toUpperCase()
      if (method === "GET" && /^\/api\/v1\/prompts\/?(\?.*)?$/.test(path)) return serverPromptList
      return {}
    })
  })

  // CC-05 (#3112): PromptSelect listed prompts only from Dexie getAllPrompts; it now also pages the server library.
  it("CC-05 (#3112): the chat Prompt picker lists the user's server prompts", async () => {
    const user = userEvent.setup()
    renderPromptSelect()

    await user.click(await screen.findByRole("button", { name: "selectAPrompt" }))
    // The local library has loaded into the open menu ...
    await screen.findByText("Local quick prompt")

    // ... and the server library must be offered alongside it.
    expect(await screen.findByText(SERVER_PROMPT_NAME)).toBeInTheDocument()
  })
})
