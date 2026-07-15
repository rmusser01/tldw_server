import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { SkillDetailsDrawer } from "../SkillDetailsDrawer"

const getSkill = vi.hoisted(() => vi.fn())

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getSkill }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string; [key: string]: unknown }) =>
      (options?.defaultValue ?? _key).replace(
        /\{\{(\w+)\}\}/g,
        (_match, token: string) => String(options?.[token] ?? "")
      )
  })
}))

const renderDrawer = (props: Partial<React.ComponentProps<typeof SkillDetailsDrawer>> = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <SkillDetailsDrawer
        scopeKey="server-one"
        skillName="research-skill"
        onClose={vi.fn()}
        onTest={vi.fn()}
        onEdit={vi.fn()}
        onUseInChat={vi.fn()}
        onCopyInvocation={vi.fn()}
        onDuplicate={vi.fn()}
        {...props}
      />
    </QueryClientProvider>
  )
}

describe("SkillDetailsDrawer", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getSkill.mockResolvedValue({
      id: "skill-1",
      name: "research-skill",
      description: "Research a topic with sources",
      argument_hint: "[topic]",
      disable_model_invocation: false,
      user_invocable: true,
      allowed_tools: ["Read", "WebSearch"],
      model: "gpt-4o-mini",
      context: "fork",
      runtime: {
        execution_mode: "fork",
        test_run_may_call_model: true,
        declares_tools: true,
        declared_tool_count: 2,
        model_override: "gpt-4o-mini",
        auto_invocation_enabled: false
      },
      content: "Research $ARGUMENTS and return citations.",
      raw_content: null,
      supporting_files: { "rubric.md": "rubric" },
      directory_path: "/tmp/research-skill",
      created_at: "2026-07-14T00:00:00Z",
      last_modified: "2026-07-14T00:00:00Z",
      version: 2
    })
  })

  it("shows an understandable read-only skill overview", async () => {
    renderDrawer()

    expect(
      await screen.findByRole("dialog", { name: "Skill details: research-skill" })
    ).toBeInTheDocument()
    expect(await screen.findByText("Research a topic with sources")).toBeInTheDocument()
    expect(screen.getByText("Research $ARGUMENTS and return citations.")).toBeInTheDocument()
    expect(screen.getByText("Read")).toBeInTheDocument()
    expect(screen.getByText("WebSearch")).toBeInTheDocument()
    expect(screen.getByText("rubric.md")).toBeInTheDocument()
    expect(screen.getByText("Version")).toBeInTheDocument()
    expect(screen.getByText("2", { selector: "dd" })).toBeInTheDocument()
    expect(document.querySelector('time[datetime="2026-07-14T00:00:00Z"]')).not.toBeNull()
    expect(screen.getByRole("heading", { name: "Runtime impact" })).toBeInTheDocument()
    expect(screen.getByText("Test may call model")).toBeInTheDocument()
    expect(screen.getByText("2 tools declared")).toBeInTheDocument()
    expect(screen.getByText("Auto invocation off")).toBeInTheDocument()
  })

  it("reloads the same skill name when the server identity scope changes", async () => {
    const firstSkill = {
      id: "skill-1",
      name: "research-skill",
      description: "Server one details",
      argument_hint: null,
      disable_model_invocation: false,
      user_invocable: true,
      allowed_tools: null,
      model: null,
      context: "inline",
      runtime: null,
      content: "Server one body",
      raw_content: null,
      supporting_files: {},
      directory_path: "/tmp/research-skill",
      created_at: "2026-07-14T00:00:00Z",
      last_modified: "2026-07-14T00:00:00Z",
      version: 1
    }
    getSkill
      .mockResolvedValueOnce(firstSkill)
      .mockResolvedValueOnce({
        ...firstSkill,
        description: "Server two details",
        content: "Server two body",
        version: 2
      })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const renderAtScope = (scopeKey: string) => (
      <QueryClientProvider client={queryClient}>
        <SkillDetailsDrawer
          scopeKey={scopeKey}
          skillName="research-skill"
          onClose={vi.fn()}
          onTest={vi.fn()}
          onEdit={vi.fn()}
          onUseInChat={vi.fn()}
          onCopyInvocation={vi.fn()}
          onDuplicate={vi.fn()}
        />
      </QueryClientProvider>
    )
    const rendered = render(renderAtScope("server-one"))
    expect(await screen.findByText("Server one details")).toBeInTheDocument()

    rendered.rerender(renderAtScope("server-two"))

    expect(await screen.findByText("Server two details")).toBeInTheDocument()
    expect(getSkill).toHaveBeenCalledTimes(2)
  })

  it("exposes the complete next-action workflow", async () => {
    const onUseInChat = vi.fn()
    const onCopyInvocation = vi.fn()
    const onTest = vi.fn()
    const onEdit = vi.fn()
    const onDuplicate = vi.fn()
    renderDrawer({ onUseInChat, onCopyInvocation, onTest, onEdit, onDuplicate })
    await screen.findByText("Research a topic with sources")

    fireEvent.click(screen.getByRole("button", { name: "Use in chat" }))
    fireEvent.click(screen.getByRole("button", { name: "Copy invocation" }))
    fireEvent.click(screen.getByRole("button", { name: "Test run" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit" }))
    fireEvent.click(screen.getByRole("button", { name: "Duplicate" }))

    for (const name of ["Use in chat", "Copy invocation", "Test run", "Edit", "Duplicate"]) {
      expect(screen.getByRole("button", { name })).toHaveClass("min-h-11")
    }

    expect(onUseInChat).toHaveBeenCalledWith("research-skill")
    expect(onCopyInvocation).toHaveBeenCalledWith("research-skill")
    expect(onTest).toHaveBeenCalledWith("research-skill")
    expect(onEdit).toHaveBeenCalledWith("research-skill")
    expect(onDuplicate).toHaveBeenCalledWith("research-skill")
  })

  it("keeps retry available when details fail to load", async () => {
    getSkill.mockRejectedValueOnce(new Error("backend unavailable"))
    renderDrawer()

    expect(await screen.findByRole("alert")).toHaveTextContent("Failed to load skill details")
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    await waitFor(() => expect(getSkill).toHaveBeenCalledTimes(2))
  })
})
