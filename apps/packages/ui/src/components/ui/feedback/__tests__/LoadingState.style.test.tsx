import React from "react"
import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { LoadingState } from "../LoadingState"

describe("LoadingState style", () => {
  it("applies surface styles to the standard loading container", () => {
    const { container } = render(
      <LoadingState mode="spinner" style={{ height: 240 }} />
    )

    const loadingState = container.querySelector(
      '[data-ds-component="LoadingState"]'
    )
    expect(loadingState).toBeInTheDocument()
    expect(loadingState).toHaveStyle({ height: "240px" })
  })

  it("keeps fullscreen sizing controlled by the fixed inset layout", () => {
    const { container } = render(
      <LoadingState fullscreen mode="spinner" style={{ height: 240 }} />
    )

    const loadingState = container.querySelector(
      '[data-ds-component="LoadingState"]'
    )
    expect(loadingState).toBeInTheDocument()
    expect(loadingState).not.toHaveStyle({ height: "240px" })
  })
})
