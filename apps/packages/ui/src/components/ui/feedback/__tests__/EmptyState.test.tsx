import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { EmptyState } from "../EmptyState"

describe("EmptyState", () => {
  it("renders plain text descriptions as paragraph copy", () => {
    const { container } = render(
      <EmptyState title="No results" description="Try another search." />
    )

    expect(container.querySelector("p")?.textContent).toBe("Try another search.")
  })

  it("does not wrap rich descriptions in a paragraph", () => {
    const { container } = render(
      <EmptyState
        title="Start research"
        description={
          <div data-testid="rich-description">
            <p>Use selected sources for grounded answers.</p>
            <div>Choose a source to continue.</div>
          </div>
        }
      />
    )

    expect(screen.getByTestId("rich-description").parentElement?.tagName).toBe("DIV")
    expect(container.querySelector("p [data-testid='rich-description']")).toBeNull()
  })
})
