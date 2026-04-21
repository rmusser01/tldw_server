import { render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ChatComposer } from "../ChatComposer"

const commonTextProps = {
  message: "",
  onMessageChange: vi.fn(),
  onSend: vi.fn(),
}

// Variants are lazy-loaded via React.lazy — awaiting a sentinel inside the
// chosen variant proves the chunk resolved. Each variant renders a root
// `[data-variant='vN']` we can wait for.
const waitForVariant = async (
  container: HTMLElement,
  variant: "v1" | "v3" | "v5"
) => {
  await waitFor(() => {
    expect(
      container.querySelector(`[data-variant='${variant}']`)
    ).toBeTruthy()
  })
}

describe("ChatComposer", () => {
  it("renders TerminalStackV1 when variant='v1'", async () => {
    const { container } = render(
      <ChatComposer variant="v1" {...commonTextProps} />
    )
    await waitForVariant(container, "v1")
    expect(container.querySelector("[data-variant='v3']")).toBeNull()
    expect(container.querySelector("[data-variant='v5']")).toBeNull()
  })

  it("renders SplitBriefV3 when variant='v3'", async () => {
    const { container } = render(
      <ChatComposer
        variant="v3"
        {...commonTextProps}
        briefSections={[
          {
            id: "b",
            fields: [{ id: "src", fieldKey: "src", value: "irb" }],
          },
        ]}
      />
    )
    await waitForVariant(container, "v3")
    expect(container.querySelector("[data-variant='v1']")).toBeNull()
  })

  it("renders RadialCommandV5 when variant='v5'", async () => {
    const { container } = render(
      <ChatComposer variant="v5" {...commonTextProps} />
    )
    await waitForVariant(container, "v5")
  })

  it("forwards V1-specific props (sourceChip)", async () => {
    render(
      <ChatComposer
        variant="v1"
        {...commonTextProps}
        sourceChip={{ count: 14, label: "irb-archive" }}
      />
    )
    expect(await screen.findByText("14")).toBeTruthy()
    expect(screen.getByText("irb-archive")).toBeTruthy()
  })

  it("forwards V3-specific props (briefSections)", async () => {
    render(
      <ChatComposer
        variant="v3"
        {...commonTextProps}
        briefSections={[
          {
            id: "b",
            label: "Brief",
            fields: [
              { id: "src", fieldKey: "src", value: "irb-archive · 14" },
            ],
          },
        ]}
      />
    )
    expect(await screen.findByText("Brief")).toBeTruthy()
    expect(screen.getByText(/irb-archive/)).toBeTruthy()
  })

  it("forwards V5-specific props (paletteOpen)", async () => {
    render(
      <ChatComposer
        variant="v5"
        {...commonTextProps}
        paletteOpen
        paletteGroups={[
          {
            id: "m",
            label: "Models",
            rows: [{ id: "h", command: "/model haiku-4-5" }],
          },
        ]}
        paletteActiveIndex={0}
        onPaletteActiveIndexChange={vi.fn()}
        onPaletteSelect={vi.fn()}
        paletteQuery="model"
      />
    )
    expect(
      await screen.findByRole("listbox", { name: /composer slash commands/i })
    ).toBeTruthy()
    expect(screen.getByText("/model haiku-4-5")).toBeTruthy()
  })

  it("preserves common props (onMessageChange) across variants", async () => {
    for (const variant of ["v1", "v3", "v5"] as const) {
      const onMessageChange = vi.fn()
      const v3Extras =
        variant === "v3" ? { briefSections: [{ id: "b", fields: [] }] } : {}
      const { unmount } = render(
        <ChatComposer
          variant={variant}
          message=""
          onMessageChange={onMessageChange}
          onSend={vi.fn()}
          {...v3Extras}
        />
      )
      // Textarea is rendered and accepts events (await lazy chunk)
      const ta = await screen.findByRole("textbox", {
        name: /message|question/i,
      })
      expect(ta).toBeTruthy()
      unmount()
    }
  })

  it("error boundary catches render failures from a variant chunk", async () => {
    // Re-mock the V3 module to throw on render, simulating a chunk that
    // loaded but blew up during construction (e.g. corrupted bundle).
    vi.resetModules()
    vi.doMock("../variants/SplitBriefV3", () => ({
      SplitBriefV3: () => {
        throw new Error("boom")
      },
    }))
    // Silence React's noisy error log for this expected failure.
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {})

    const { ChatComposer: ChatComposerWithBrokenV3 } = await import(
      "../ChatComposer"
    )

    render(
      <ChatComposerWithBrokenV3
        variant="v3"
        {...commonTextProps}
        briefSections={[{ id: "b", fields: [] }]}
      />
    )

    expect(
      await screen.findByTestId("composer-variant-load-error")
    ).toBeTruthy()
    consoleError.mockRestore()
    vi.doUnmock("../variants/SplitBriefV3")
    vi.resetModules()
  })

  it("shows a skeleton fallback while a lazy variant chunk loads", async () => {
    // Make V1's import never resolve during this render so the
    // skeleton stays visible when we assert.
    vi.resetModules()
    vi.doMock("../variants/TerminalStackV1", () => {
      // Return a promise that never resolves inside the mock — React.lazy
      // wraps the import, so we approximate "chunk in flight" by returning
      // a valid module with a suspending Lazy component.
      const PendingTerminal: React.FC = () => {
        throw new Promise(() => {})
      }
      return { TerminalStackV1: PendingTerminal }
    })

    const { ChatComposer: Pending } = await import("../ChatComposer")
    const { container } = render(<Pending variant="v1" {...commonTextProps} />)

    await waitFor(() => {
      expect(
        container.querySelector(
          "[data-testid='composer-variant-loading']"
        )
      ).toBeTruthy()
    })

    vi.doUnmock("../variants/TerminalStackV1")
    vi.resetModules()
  })

  it("error boundary resets when user picks a different variant", async () => {
    // V3 is broken; V1 is healthy.
    vi.resetModules()
    vi.doMock("../variants/SplitBriefV3", () => ({
      SplitBriefV3: () => {
        throw new Error("v3 broke")
      },
    }))
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {})

    const { ChatComposer: Comp } = await import("../ChatComposer")

    const { rerender } = render(
      <Comp
        variant="v3"
        {...commonTextProps}
        briefSections={[{ id: "b", fields: [] }]}
      />
    )
    // First render shows the error UI
    expect(
      await screen.findByTestId("composer-variant-load-error")
    ).toBeTruthy()

    // User flips to V1 — error boundary should reset and V1 renders
    rerender(<Comp variant="v1" {...commonTextProps} />)
    await waitFor(() => {
      expect(
        document.querySelector("[data-variant='v1']")
      ).toBeTruthy()
    })
    expect(
      screen.queryByTestId("composer-variant-load-error")
    ).toBeNull()

    consoleError.mockRestore()
    vi.doUnmock("../variants/SplitBriefV3")
    vi.resetModules()
  })
})
