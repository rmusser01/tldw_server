import { describe, expect, it } from "vitest"

import { isBenign } from "../e2e/utils/helpers"

describe("E2E diagnostics benign classification", () => {
  it("treats the observed React DOM cleanup race as non-critical diagnostics noise", () => {
    expect(
      isBenign(
        "Failed to execute 'removeChild' on 'Node': The node to be removed is not a child of this node."
      )
    ).toBe(true)
  })
})
