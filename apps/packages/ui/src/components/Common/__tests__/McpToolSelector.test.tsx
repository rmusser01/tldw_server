// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { McpToolSelector } from "../McpToolSelector"
import { buildChatToolFilterState } from "@/utils/chat-tools"

const t = (_key: string, fallback?: string, options?: Record<string, unknown>) => {
  const template = fallback ?? _key
  if (!options) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_match, token) =>
    options[token] == null ? "" : String(options[token])
  )
}

describe("McpToolSelector", () => {
  it("renders tool counts and degraded state labels", () => {
    const filterState = buildChatToolFilterState({
      tools: [
        { name: "notes.search", canExecute: true },
        { name: "slides.list", canExecute: true },
        { name: "media.search", canExecute: false },
        { name: "docs.search", canExecute: true },
        { name: "docs_search", canExecute: true }
      ],
      disabledToolNames: ["slides_list"]
    })

    render(
      <McpToolSelector
        discoveredTools={filterState.discoveredTools}
        toolCounts={filterState.counts}
        toolsLoading={false}
        hasMcp
        healthState="healthy"
        onToolEnabledChange={vi.fn()}
        t={t}
      />
    )

    expect(screen.getByText("1 enabled")).toBeInTheDocument()
    expect(screen.getByText("1 disabled")).toBeInTheDocument()
    expect(screen.getByText("1 unavailable")).toBeInTheDocument()
    expect(screen.getAllByText("Name conflict")).toHaveLength(2)
    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.getByText("Off")).toBeInTheDocument()
  })

  it("toggles executable non-conflicting tools by normalized chat name", async () => {
    const user = userEvent.setup()
    const onToolEnabledChange = vi.fn()
    const filterState = buildChatToolFilterState({
      tools: [
        { name: "notes.search", canExecute: true },
        { name: "slides.list", canExecute: true }
      ],
      disabledToolNames: ["slides_list"]
    })

    render(
      <McpToolSelector
        discoveredTools={filterState.discoveredTools}
        toolCounts={filterState.counts}
        toolsLoading={false}
        hasMcp
        healthState="healthy"
        onToolEnabledChange={onToolEnabledChange}
        t={t}
      />
    )

    await user.click(screen.getByRole("switch", { name: /notes.search/i }))
    await user.click(screen.getByRole("switch", { name: /slides.list/i }))

    expect(onToolEnabledChange).toHaveBeenNthCalledWith(1, "notes_search", false)
    expect(onToolEnabledChange).toHaveBeenNthCalledWith(2, "slides_list", true)
  })

  it("does not toggle unavailable or colliding tools", async () => {
    const user = userEvent.setup()
    const onToolEnabledChange = vi.fn()
    const filterState = buildChatToolFilterState({
      tools: [
        { name: "media.search", canExecute: false },
        { name: "docs.search", canExecute: true },
        { name: "docs_search", canExecute: true }
      ]
    })

    render(
      <McpToolSelector
        discoveredTools={filterState.discoveredTools}
        toolCounts={filterState.counts}
        toolsLoading={false}
        hasMcp
        healthState="healthy"
        onToolEnabledChange={onToolEnabledChange}
        t={t}
      />
    )

    expect(screen.getByRole("switch", { name: /media.search/i })).toBeDisabled()
    expect(
      screen.getByRole("switch", { name: "docs.search MCP tool" })
    ).toBeDisabled()
    await user.click(screen.getByRole("switch", { name: /media.search/i }))
    await user.click(
      screen.getByRole("switch", { name: "docs.search MCP tool" })
    )

    expect(onToolEnabledChange).not.toHaveBeenCalled()
  })
})
