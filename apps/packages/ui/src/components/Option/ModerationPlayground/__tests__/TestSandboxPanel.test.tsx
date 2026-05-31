// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTester(overrides: Record<string, any> = {}) {
  return {
    phase: "input" as "input" | "output",
    setPhase: vi.fn(),
    text: "",
    setText: vi.fn(),
    userId: "",
    setUserId: vi.fn(),
    result: null,
    history: [],
    running: false,
    runTest: vi.fn().mockResolvedValue(undefined),
    runTestWith: vi.fn().mockResolvedValue(undefined),
    clearHistory: vi.fn(),
    loadFromHistory: vi.fn(),
    ...overrides
  }
}

function makeMessageApi() {
  return {
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn()
  }
}

// ---------------------------------------------------------------------------
// Import component under test (after mocks)
// ---------------------------------------------------------------------------

import TestSandboxPanel from "../TestSandboxPanel"

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("TestSandboxPanel", () => {
  let messageApi: ReturnType<typeof makeMessageApi>

  beforeEach(() => {
    vi.clearAllMocks()
    messageApi = makeMessageApi()
  })

  it("renders phase selector with 'User message' and 'AI response' buttons", () => {
    const tester = makeTester()
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    expect(screen.getByRole("radio", { name: "User message" })).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "AI response" })).toBeInTheDocument()
  })

  it("renders all quick sample buttons", () => {
    const tester = makeTester()
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    expect(screen.getByRole("button", { name: "PII: email" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "PII: phone" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Profanity" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Violence" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Clean text" })).toBeInTheDocument()
  })

  it("renders Run Test button", () => {
    const tester = makeTester()
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    expect(screen.getByRole("button", { name: /run test/i })).toBeInTheDocument()
  })

  it("clicking a quick sample calls setText with the sample text", () => {
    const tester = makeTester()
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    fireEvent.click(screen.getByRole("button", { name: "PII: email" }))
    expect(tester.setText).toHaveBeenCalledWith(
      "Contact me at john.doe@example.com for details"
    )
  })

  it("shows results section when result is set", () => {
    const tester = makeTester({
      text: "test text",
      result: {
        flagged: true,
        action: "block",
        sample: "test",
        redacted_text: null,
        effective: { enabled: true },
        category: "violence"
      }
    })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    expect(screen.getByTestId("results-section")).toBeInTheDocument()
    expect(screen.getByText("Content Blocked")).toBeInTheDocument()
  })

  it("does not show results section when result is null", () => {
    const tester = makeTester({ result: null })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    expect(screen.queryByTestId("results-section")).not.toBeInTheDocument()
  })

  it("renders Test History heading", () => {
    const tester = makeTester()
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    expect(screen.getByText("Test History")).toBeInTheDocument()
  })

  it("shows empty state when no history", () => {
    const tester = makeTester({ history: [] })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    expect(screen.getByText("No tests run yet")).toBeInTheDocument()
  })

  it("renders history table when history has entries", () => {
    const tester = makeTester({
      history: [
        {
          phase: "input",
          text: "Some test text that might be a bit long for display",
          userId: "",
          result: { flagged: false, action: "pass", sample: null, redacted_text: null, effective: {}, category: null },
          timestamp: 1000
        }
      ]
    })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    expect(screen.getByTestId("history-table")).toBeInTheDocument()
    expect(screen.getByText("Content Allowed")).toBeInTheDocument()
  })

  it("renders redacted text when present in result", () => {
    const tester = makeTester({
      text: "My phone number is 555-123-4567",
      result: {
        flagged: true,
        action: "redact",
        sample: "555-123-4567",
        redacted_text: "My phone number is [REDACTED]",
        effective: {},
        category: "pii_phone"
      }
    })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    expect(screen.getByText("Content Redacted")).toBeInTheDocument()
    expect(screen.getByText("My phone number is [REDACTED]")).toBeInTheDocument()
  })

  it("explains when the moderation engine is disabled", () => {
    const tester = makeTester({
      text: "sample",
      result: {
        flagged: false,
        action: "pass",
        sample: null,
        redacted_text: null,
        effective: { enabled: false },
        category: null
      }
    })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)

    expect(screen.getByText("Why this result?")).toBeInTheDocument()
    expect(screen.getByText(/moderation engine is disabled/i)).toBeInTheDocument()
  })

  it("explains when the selected phase is disabled", () => {
    const tester = makeTester({
      phase: "output",
      text: "sample",
      result: {
        flagged: false,
        action: "pass",
        sample: null,
        redacted_text: null,
        effective: { enabled: true, output_enabled: false },
        category: null
      }
    })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)

    expect(screen.getByText(/AI response moderation is disabled/i)).toBeInTheDocument()
  })

  it("explains no-match pass results", () => {
    const tester = makeTester({
      text: "clean sample",
      result: {
        flagged: false,
        action: "pass",
        sample: null,
        redacted_text: null,
        effective: { enabled: true, input_enabled: true },
        category: null
      }
    })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)

    expect(screen.getByText(/no active rule matched/i)).toBeInTheDocument()
  })

  it("explains matched rule and category results", () => {
    const tester = makeTester({
      text: "contains secret",
      result: {
        flagged: true,
        action: "block",
        sample: "secret",
        redacted_text: null,
        effective: { enabled: true, input_enabled: true },
        category: "confidential"
      }
    })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)

    expect(screen.getByText(/matched an active confidential rule/i)).toBeInTheDocument()
    expect(screen.getByText(/sample "secret"/i)).toBeInTheDocument()
  })

  it("explains per-user override and global fallback outcomes", () => {
    const overrideTester = makeTester({
      userId: "alice",
      text: "sample",
      result: {
        flagged: true,
        action: "warn",
        sample: null,
        redacted_text: null,
        effective: { enabled: true, user_override: { enabled: true } },
        category: null
      }
    })
    const { rerender } = render(<TestSandboxPanel tester={overrideTester as any} messageApi={messageApi} />)
    expect(screen.getByText(/per-user override/i)).toBeInTheDocument()

    const fallbackTester = makeTester({
      text: "sample",
      result: {
        flagged: true,
        action: "warn",
        sample: null,
        redacted_text: null,
        effective: { enabled: true },
        category: null
      }
    })
    rerender(<TestSandboxPanel tester={fallbackTester as any} messageApi={messageApi} />)
    expect(screen.getByText(/global policy fallback/i)).toBeInTheDocument()
  })

  it("shows 'Running...' text when running is true", () => {
    const tester = makeTester({ running: true, text: "test" })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    expect(screen.getByRole("button", { name: /running/i })).toBeInTheDocument()
  })

  it("clicking Load on history entry calls loadFromHistory", () => {
    const entry = {
      phase: "input" as const,
      text: "test text",
      userId: "",
      result: { flagged: false, action: "pass" as const, sample: null, redacted_text: null, effective: {}, category: null },
      timestamp: 1000
    }
    const tester = makeTester({ history: [entry] })
    render(<TestSandboxPanel tester={tester as any} messageApi={messageApi} />)
    fireEvent.click(screen.getByRole("button", { name: "Load" }))
    expect(tester.loadFromHistory).toHaveBeenCalledWith(entry)
  })
})
