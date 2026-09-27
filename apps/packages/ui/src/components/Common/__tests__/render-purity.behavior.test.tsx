import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ProviderIcons } from "../ProviderIcon"
import { TableBlock } from "../TableBlock"
import { RenderStrip } from "@/components/Option/Speech/RenderStrip"

vi.mock("@/components/Common/WaveformCanvas", () => ({ default: () => null }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback || key
  })
}))

const NativeURL = URL
const downloads: string[] = []

beforeEach(() => {
  downloads.length = 0
  vi.useFakeTimers()
  vi.setSystemTime(new Date("2026-09-10T12:00:00Z"))
  vi.stubGlobal(
    "URL",
    class extends NativeURL {
      static createObjectURL = () => "blob:test"
      static revokeObjectURL = () => {}
    }
  )
  vi.spyOn(HTMLMediaElement.prototype, "load").mockImplementation(() => {})
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(
    function () {
      downloads.push(this.download)
    }
  )
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe("render purity and event boundaries", () => {
  it("keeps a registry-selected provider icon mounted on rerender", () => {
    const { container, rerender } = render(
      <ProviderIcons provider="openai" className="before" />
    )
    const icon = container.querySelector("svg")
    expect(icon).not.toBeNull()
    rerender(<ProviderIcons provider="openai" className="after" />)
    expect(container.querySelector("svg")).toBe(icon)
    expect(icon).toHaveClass("after")
  })

  it("timestamps CSV downloads when clicked, without downloading during render", () => {
    render(
      <TableBlock>
        <table>
          <thead>
            <tr>
              <th>Name</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Ada</td>
            </tr>
          </tbody>
        </table>
      </TableBlock>
    )
    expect(downloads).toEqual([])
    vi.setSystemTime(new Date("2026-09-10T12:00:05Z"))
    fireEvent.click(screen.getByRole("button", { name: "Download CSV" }))
    expect(downloads).toEqual([
      `table-${Date.parse("2026-09-10T12:00:05Z")}.csv`
    ])
  })

  it("keeps the audio download filename stable across unrelated rerenders", () => {
    const props = {
      id: "render-1",
      state: "ready" as const,
      config: { provider: "tldw", voice: "af_heart", format: "mp3" },
      audioUrl: "blob:audio"
    }
    const { rerender } = render(<RenderStrip {...props} />)
    fireEvent.click(screen.getByRole("button", { name: "Download audio" }))
    vi.setSystemTime(new Date("2026-09-10T12:00:05Z"))
    rerender(<RenderStrip {...props} forcePaused />)
    fireEvent.click(screen.getByRole("button", { name: "Download audio" }))
    expect(downloads[1]).toBe(downloads[0])
  })
})
