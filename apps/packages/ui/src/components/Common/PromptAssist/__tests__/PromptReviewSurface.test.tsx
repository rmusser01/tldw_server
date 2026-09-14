import type { PromptImproveFinding } from "@/services/prompt-improvement"
import { contrastRatio } from "@/themes/contrast"
import { getBuiltinPresets } from "@/themes/presets"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it, vi } from "vitest"

import { PromptReviewSurface } from "../PromptReviewSurface"

const sharedTailwindStyles = readFileSync(
  resolve(process.cwd(), "src/assets/tailwind-shared.css"),
  "utf8"
)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValue?: string,
      options?: Record<string, string | number>
    ) =>
      (defaultValue ?? _key).replace(/{{(\w+)}}/g, (_, key) =>
        String(options?.[key] ?? "")
      )
  })
}))

const findings: PromptImproveFinding[] = Array.from(
  { length: 7 },
  (_, index) => ({
    category: "clarity",
    issue: `Issue ${index + 1}`,
    change: `Change ${index + 1}`
  })
)

const defaultProps = () => ({
  original: "Write a report about birds.",
  candidate: "Write a concise report about coastal birds.",
  findings,
  warnings: [] as string[],
  notice: null,
  resolvedModel: {
    provider: "openai",
    model: "gpt-5-mini",
    display_name: "GPT-5 mini"
  },
  onCandidateChange: vi.fn(),
  onApply: vi.fn(),
  onConfirmReplace: vi.fn(),
  onCancel: vi.fn()
})

describe("PromptReviewSurface", () => {
  it("shows at most five model observations and keeps the candidate editable", async () => {
    const user = userEvent.setup()
    const props = defaultProps()
    render(<PromptReviewSurface {...props} />)

    const observations = screen.getByRole("list", {
      name: "Model observations"
    })
    expect(within(observations).getAllByRole("listitem")).toHaveLength(5)
    expect(screen.getByText("Model observations")).toBeInTheDocument()
    expect(screen.getByText("Used GPT-5 mini (openai)")).toBeInTheDocument()

    const candidate = screen.getByRole("textbox", {
      name: "Improved prompt candidate"
    })
    await user.clear(candidate)
    await user.type(candidate, "My edited candidate")
    expect(props.onCandidateChange).toHaveBeenLastCalledWith(
      "My edited candidate"
    )
  })

  it("renders semantic additions and removals without interpreting candidate HTML", async () => {
    const user = userEvent.setup()
    const props = {
      ...defaultProps(),
      original: "Keep old wording",
      candidate: "Keep <img src=x onerror=alert(1)> new wording"
    }
    const { container } = render(<PromptReviewSurface {...props} />)

    await user.click(screen.getByRole("button", { name: "Changes" }))
    expect(
      container.querySelector("del[data-change='removed']")
    ).toHaveTextContent("old")
    const addition = container.querySelector("ins[data-change='added']")
    expect(addition).toHaveTextContent("<img src=x onerror=alert(1)> new")
    expect(addition).toHaveClass("underline")
    expect(addition).toHaveAttribute("data-change-label", "Added")
    expect(container.querySelector("[data-diff-legend]")).not.toHaveAttribute(
      "aria-hidden"
    )
    expect(container.querySelector("img")).not.toBeInTheDocument()
    expect(container.querySelector("[data-diff-legend]")).toHaveTextContent(
      "Removed"
    )
    expect(container.querySelector("[data-diff-legend]")).toHaveTextContent(
      "Added"
    )
  })

  it("falls back to a bounded plain candidate view for excessive diff output", async () => {
    const user = userEvent.setup()
    render(
      <PromptReviewSurface
        {...defaultProps()}
        original={Array.from(
          { length: 900 },
          (_, index) => `old-${index}`
        ).join("\n")}
        candidate={Array.from(
          { length: 900 },
          (_, index) => `new-${index}`
        ).join("\n")}
      />
    )

    await user.click(screen.getByRole("button", { name: "Changes" }))
    expect(screen.getByRole("status")).toHaveTextContent(
      "This comparison is too large to highlight safely. Showing the plain candidate."
    )
    expect(
      screen.getByLabelText("Plain improved prompt candidate")
    ).toBeInTheDocument()
  })

  it("shows safety and stale notices, then requires in-panel replacement confirmation", async () => {
    const user = userEvent.setup()
    const props = {
      ...defaultProps(),
      warnings: ["protected_token_changed"],
      notice: "draft_changed" as const,
      replaceConfirmationRequired: true
    }
    render(<PromptReviewSurface {...props} />)

    expect(screen.getByRole("alert")).toHaveTextContent(
      "The draft changed while this result was open. Applying normally will not overwrite it."
    )
    expect(
      screen.getByText("Review the safety notices before applying.")
    ).toBeInTheDocument()
    await user.click(
      screen.getByRole("button", { name: "Replace current draft" })
    )
    expect(
      screen.getByText("Replace the current draft with this candidate?")
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Confirm replace" }))
    expect(props.onConfirmReplace).toHaveBeenCalledTimes(1)
  })

  it("does not allow an empty edited candidate to replace a newer draft", async () => {
    const user = userEvent.setup()
    const props = {
      ...defaultProps(),
      candidate: "   ",
      notice: "draft_changed" as const,
      replaceConfirmationRequired: true
    }
    render(<PromptReviewSurface {...props} />)

    const replace = screen.getByRole("button", {
      name: "Replace current draft"
    })
    expect(replace).toBeDisabled()
    await user.click(replace)
    expect(
      screen.queryByText("Replace the current draft with this candidate?")
    ).not.toBeInTheDocument()
    expect(props.onConfirmReplace).not.toHaveBeenCalled()
  })

  it("uses honest disclosure buttons and instance-local observation associations", async () => {
    const user = userEvent.setup()
    const first = defaultProps()
    const second = defaultProps()
    render(
      <>
        <PromptReviewSurface {...first} />
        <PromptReviewSurface {...second} />
      </>
    )

    expect(screen.queryByRole("tablist")).not.toBeInTheDocument()
    const changesButtons = screen.getAllByRole("button", { name: "Changes" })
    expect(changesButtons[0]).toHaveAttribute("aria-pressed", "false")
    await user.click(changesButtons[0])
    expect(changesButtons[0]).toHaveAttribute("aria-pressed", "true")

    const observationHeadings = screen.getAllByText("Model observations")
    const ids = observationHeadings.map((heading) => heading.id)
    expect(ids[0]).toBeTruthy()
    expect(ids[0]).not.toBe(ids[1])
    expect(observationHeadings[0].closest("section")).toHaveAttribute(
      "aria-labelledby",
      ids[0]
    )
  })

  it("supports Apply, Copy, and Cancel in review mode", async () => {
    const user = userEvent.setup()
    const writeText = vi
      .spyOn(navigator.clipboard, "writeText")
      .mockResolvedValue(undefined)
    const props = defaultProps()
    render(<PromptReviewSurface {...props} />)

    await user.click(screen.getByRole("button", { name: "Copy" }))
    await user.click(screen.getByRole("button", { name: "Apply to draft" }))
    await user.click(screen.getByRole("button", { name: "Cancel" }))

    expect(writeText).toHaveBeenCalledWith(props.candidate)
    expect(props.onApply).toHaveBeenCalledTimes(1)
    expect(props.onCancel).toHaveBeenCalledTimes(1)
  })

  it("uses inspection actions without exposing a second Apply", async () => {
    const user = userEvent.setup()
    const onUndo = vi.fn()
    const onCancel = vi.fn()
    render(
      <PromptReviewSurface
        {...defaultProps()}
        mode="inspection"
        onUndo={onUndo}
        onCancel={onCancel}
      />
    )

    expect(
      screen.queryByRole("button", { name: "Apply to draft" })
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("textbox", { name: "Improved prompt candidate" })
    ).toHaveAttribute("readonly")
    await user.click(screen.getByRole("button", { name: "Undo improvement" }))
    await user.click(screen.getByRole("button", { name: "Close" }))
    expect(onUndo).toHaveBeenCalledTimes(1)
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it("keeps review controls and the containing drawer on contrast-safe tokens", () => {
    const { rerender } = render(<PromptReviewSurface {...defaultProps()} />)

    expect(
      screen.getByRole("textbox", { name: "Improved prompt candidate" })
    ).toHaveClass("bg-bg")
    const editTab = screen.getByRole("button", { name: "Edit" })
    const changesTab = screen.getByRole("button", { name: "Changes" })
    expect(editTab).toHaveClass(
      "border-transparent",
      "aria-pressed:border-text",
      "aria-pressed:text-text",
      "aria-pressed:font-semibold"
    )
    expect(changesTab).toHaveClass(
      "border-transparent",
      "aria-pressed:border-text",
      "aria-pressed:text-text",
      "aria-pressed:font-semibold"
    )
    expect(screen.getByRole("button", { name: "Apply to draft" })).toHaveClass(
      "!bg-text",
      "!text-bg"
    )
    expect(sharedTailwindStyles).toContain(
      ".ant-drawer-body {\n  background-color: rgb(var(--color-elevated)) !important;"
    )

    rerender(<PromptReviewSurface {...defaultProps()} mode="inspection" />)
    expect(screen.getByRole("button", { name: "Close" })).toHaveClass(
      "!bg-text",
      "!text-bg"
    )

    for (const preset of getBuiltinPresets()) {
      for (const mode of ["light", "dark"] as const) {
        const tokens = preset.palette[mode]
        expect(
          contrastRatio(tokens.text, tokens.elevated),
          `${preset.name} ${mode} active review label`
        ).toBeGreaterThanOrEqual(4.5)
        expect(
          contrastRatio(tokens.bg, tokens.text),
          `${preset.name} ${mode} review action`
        ).toBeGreaterThanOrEqual(4.5)
        expect(
          contrastRatio(tokens.text, tokens.bg),
          `${preset.name} ${mode} review textarea`
        ).toBeGreaterThanOrEqual(4.5)
      }
    }
  })
})
