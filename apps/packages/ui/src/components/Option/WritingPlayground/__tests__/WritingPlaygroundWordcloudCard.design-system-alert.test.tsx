import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { WritingPlaygroundWordcloudCard } from "../WritingPlaygroundWordcloudCard"
import type { WordcloudCardProps } from "../WritingPlaygroundDiagnostics.types"

const t: WordcloudCardProps["t"] = (_key, defaultValue) => defaultValue

const makeProps = (
  overrides: Partial<WordcloudCardProps> = {}
): WordcloudCardProps => ({
  t,
  wordcloudStatus: null,
  wordcloudStatusColor: "default",
  canGenerateWordcloud: true,
  isGeneratingWordcloud: false,
  onGenerateWordcloud: vi.fn(),
  wordcloudError: "Unable to build wordcloud",
  onClearWordcloud: vi.fn(),
  wordcloudWords: [],
  ...overrides
})

describe("WritingPlaygroundWordcloudCard product-state alerts", () => {
  it("renders wordcloud errors through the design-system Alert and keeps clear behavior", () => {
    const onClearWordcloud = vi.fn()

    render(
      <WritingPlaygroundWordcloudCard
        {...makeProps({ onClearWordcloud })}
      />
    )

    const errorMessage = screen.getByText("Unable to build wordcloud")
    expect(errorMessage.closest('[data-ds-component="Alert"]')).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Clear" }))
    expect(onClearWordcloud).toHaveBeenCalledTimes(1)
  })
})
