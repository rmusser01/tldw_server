import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useSearchParams } from "@web/extension/shims/react-router-dom"

// push/replace return a resolved Promise like the real Next.js router so the
// shim's `navigation.catch(...)` has something to chain onto.
const mockPush = vi.fn(() => Promise.resolve(true))
const mockReplace = vi.fn(() => Promise.resolve(true))

// Dynamic route: pathname is the `[bracket]` pattern, asPath is the resolved URL.
const mockRouter = {
  asPath: "/sources/source-123?tab=notes",
  pathname: "/sources/[id]",
  query: { id: "source-123" } as Record<string, string | string[] | undefined>,
  push: mockPush,
  replace: mockReplace,
  back: vi.fn()
}

vi.mock("next/router", () => ({
  useRouter: () => mockRouter
}))

const SearchParamsButton = () => {
  const [, setSearchParams] = useSearchParams()
  return (
    <button type="button" onClick={() => setSearchParams({ tab: "summary" })}>
      search
    </button>
  )
}

describe("useSearchParams on dynamic routes", () => {
  beforeEach(() => {
    mockPush.mockClear()
    mockReplace.mockClear()
    mockRouter.asPath = "/sources/source-123?tab=notes"
    mockRouter.pathname = "/sources/[id]"
  })

  it("builds the URL from the resolved path, not the [bracket] pattern", async () => {
    const user = userEvent.setup()
    render(<SearchParamsButton />)

    await user.click(screen.getByRole("button", { name: "search" }))

    // Must push the concrete path (/sources/source-123), never /sources/[id].
    expect(mockPush).toHaveBeenCalledWith("/sources/source-123?tab=summary")
  })
})
