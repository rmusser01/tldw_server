import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PromptDiff } from "../PromptDiff"

const mocks = vi.hoisted(() => ({
  diffWordsWithSpace: vi.fn(),
  diffLines: vi.fn()
}))

vi.mock("diff", () => mocks)
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: string) => defaultValue ?? _key
  })
}))

describe("PromptDiff bounded computation", () => {
  beforeEach(() => {
    mocks.diffWordsWithSpace.mockReset()
    mocks.diffLines.mockReset()
  })

  it("passes strict jsdiff work limits and treats an aborted word diff as plain text", () => {
    mocks.diffWordsWithSpace.mockReturnValue(undefined)
    const original = "a ".repeat(2_500)
    const candidate = "b ".repeat(2_500)

    render(<PromptDiff original={original} candidate={candidate} />)

    expect(mocks.diffWordsWithSpace).toHaveBeenCalledWith(
      original,
      candidate,
      expect.objectContaining({
        timeout: expect.any(Number),
        maxEditLength: expect.any(Number)
      })
    )
    expect(screen.getByRole("status")).toHaveTextContent(
      "This comparison is too large to highlight safely. Showing the plain candidate."
    )
  })

  it("passes the same strict limits to line fallback and handles its abort", () => {
    mocks.diffLines.mockReturnValue(undefined)
    const original = "old line long enough here\n".repeat(350)
    const candidate = "new line long enough here\n".repeat(350)

    render(<PromptDiff original={original} candidate={candidate} />)

    expect(mocks.diffLines).toHaveBeenCalledWith(
      original,
      candidate,
      expect.objectContaining({
        timeout: expect.any(Number),
        maxEditLength: expect.any(Number)
      })
    )
    expect(
      screen.getByLabelText("Plain improved prompt candidate")
    ).toBeInTheDocument()
  })
})
