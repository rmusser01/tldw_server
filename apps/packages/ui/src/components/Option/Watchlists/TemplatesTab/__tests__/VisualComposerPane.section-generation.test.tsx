import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { VisualComposerPane } from "../VisualComposerPane"
import type { ComposerAst } from "../composer-types"

vi.mock("antd", () => {
  const Input: any = ({ value, onChange, ...rest }: any) => (
    <input value={(value as string) || ""} onChange={onChange} {...rest} />
  )
  Input.TextArea = ({ value, onChange, ...rest }: any) => (
    <textarea value={(value as string) || ""} onChange={onChange} {...rest} />
  )

  return {
    Alert: ({ title, description }: any) => (
      <div>
        {title}
        {description}
      </div>
    ),
    Button: ({ children, onClick, loading, disabled, danger: _danger, ...rest }: any) => (
      <button
        type="button"
        disabled={Boolean(loading || disabled)}
        onClick={() => onClick?.()}
        {...rest}
      >
        {children}
      </button>
    ),
    Input,
    Select: ({ value, onChange, options = [] }: any) => (
      <select
        value={value == null ? "" : String(value)}
        onChange={(event) => onChange?.(event.currentTarget.value || undefined)}
      >
        <option value="" />
        {options.map((option: any) => (
          <option key={String(option.value)} value={String(option.value)}>
            {String(option.label)}
          </option>
        ))}
      </select>
    ),
    Space: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
  }
})

describe("VisualComposerPane section generation", () => {
  it("runs section generation and updates block content", async () => {
    const ast: ComposerAst = {
      schema_version: "1.0.0",
      nodes: [
        {
          id: "intro-1",
          type: "IntroSummaryBlock",
          source: "",
          config: {
            prompt: "Write a concise intro."
          }
        }
      ]
    }

    const onChange = vi.fn()
    const onGenerateSection = vi.fn().mockResolvedValue({
      block_id: "intro-1",
      content: "Generated introduction paragraph.",
      warnings: [],
      diagnostics: {}
    })

    render(
      <VisualComposerPane
        ast={ast}
        onChange={onChange}
        runs={[{ id: 88, label: "Run #88" }]}
        selectedRunId={88}
        onSelectedRunIdChange={vi.fn()}
        onGenerateSection={onGenerateSection}
      />
    )

    fireEvent.click(screen.getByTestId("visual-generate-intro-1"))

    await waitFor(() =>
      expect(onGenerateSection).toHaveBeenCalledWith({
        run_id: 88,
        block_id: "intro-1",
        prompt: "Write a concise intro.",
        input_scope: "all_items",
        style: undefined,
        length_target: "medium"
      })
    )

    await waitFor(() => expect(onChange).toHaveBeenCalled())
    const lastAstArg = onChange.mock.calls.at(-1)?.[0] as ComposerAst
    expect(lastAstArg.nodes[0].source).toBe("Generated introduction paragraph.")
  })

  it("avoids duplicate node ids when adding blocks to existing AST", () => {
    const ast: ComposerAst = {
      schema_version: "1.0.0",
      nodes: [
        {
          id: "header-1",
          type: "HeaderBlock",
          source: "# {{ title }}"
        }
      ]
    }

    const onChange = vi.fn()
    render(
      <VisualComposerPane
        ast={ast}
        onChange={onChange}
        runs={[]}
        selectedRunId={undefined}
        onSelectedRunIdChange={vi.fn()}
      />
    )

    fireEvent.click(screen.getByTestId("visual-add-HeaderBlock"))

    const nextAst = onChange.mock.calls.at(-1)?.[0] as ComposerAst
    const ids = nextAst.nodes.map((node) => node.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it("renders the empty composer callout through the design-system Alert primitive", () => {
    render(
      <VisualComposerPane
        ast={{ schema_version: "1.0.0", nodes: [] }}
        onChange={vi.fn()}
      />
    )

    const emptyTitle = screen.getByText(/No visual blocks yet/)
    expect(emptyTitle).toBeInTheDocument()
    expect(
      emptyTitle.closest('[data-ds-component="Alert"]')
    ).not.toBeNull()
    expect(
      screen.getByText("Add blocks above to start building this template.")
    ).toBeInTheDocument()
  })

  it("renders generation error and warning callouts through the design-system Alert primitive", async () => {
    const ast: ComposerAst = {
      schema_version: "1.0.0",
      nodes: [
        {
          id: "intro-1",
          type: "IntroSummaryBlock",
          source: "",
          config: {
            prompt: "Write a concise intro."
          }
        }
      ]
    }

    const onGenerateSection = vi
      .fn()
      .mockRejectedValueOnce(new Error("Section generation unavailable"))
      .mockResolvedValueOnce({
        block_id: "intro-1",
        content: "Generated introduction paragraph.",
        warnings: ["Prompt trimmed", "No source summary"],
        diagnostics: {}
      })

    render(
      <VisualComposerPane
        ast={ast}
        onChange={vi.fn()}
        runs={[{ id: 88, label: "Run #88" }]}
        selectedRunId={88}
        onSelectedRunIdChange={vi.fn()}
        onGenerateSection={onGenerateSection}
      />
    )

    fireEvent.click(screen.getByTestId("visual-generate-intro-1"))

    await waitFor(() => {
      expect(screen.getByText("Section generation unavailable")).toBeInTheDocument()
    })
    expect(
      screen
        .getByText("Section generation unavailable")
        .closest('[data-ds-component="Alert"]')
    ).not.toBeNull()

    fireEvent.click(screen.getByTestId("visual-generate-intro-1"))

    await waitFor(() => {
      expect(screen.getByText("Prompt trimmed; No source summary")).toBeInTheDocument()
    })
    expect(
      screen
        .getByText("Prompt trimmed; No source summary")
        .closest('[data-ds-component="Alert"]')
    ).not.toBeNull()
  })
})
