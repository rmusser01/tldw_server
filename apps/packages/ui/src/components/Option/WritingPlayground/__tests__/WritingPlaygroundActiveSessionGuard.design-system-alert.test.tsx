import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { WritingPlaygroundActiveSessionGuard } from "../WritingPlaygroundActiveSessionGuard"
import type { TranslateFn } from "../WritingPlaygroundDiagnostics.types"

const t: TranslateFn = (_key, defaultValue) => defaultValue

describe("WritingPlaygroundActiveSessionGuard product-state alerts", () => {
  it("renders load failures through the design-system Alert", () => {
    render(
      <WritingPlaygroundActiveSessionGuard
        hasActiveSession
        isLoading={false}
        hasError
        t={t}>
        <div>ready content</div>
      </WritingPlaygroundActiveSessionGuard>
    )

    const errorTitle = screen.getByText("Unable to load session settings.")
    expect(errorTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })
})
