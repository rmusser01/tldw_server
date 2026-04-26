import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { TokenMeter } from "../shared/TokenMeter"

describe("TokenMeter", () => {
  it("renders used / max with the 'tok' suffix by default", () => {
    render(<TokenMeter used={127} max={8000} />)
    const meter = screen.getByLabelText(/of 8K tokens used/i)
    const text = meter.textContent ?? ""
    expect(text).toContain("127")
    expect(text).toContain("8K")
    expect(text).toContain("tok")
  })

  it("hides the unit suffix when showUnit=false", () => {
    render(<TokenMeter used={84} max={8000} showUnit={false} />)
    const meter = screen.getByLabelText(/tokens used/i)
    const text = meter.textContent ?? ""
    expect(text).toContain("84")
    expect(text).toContain("8K")
    expect(text).not.toContain("tok")
  })

  it("formats 999 as '999', not as a K value", () => {
    render(<TokenMeter used={1} max={999} />)
    const meter = screen.getByLabelText(/of 999 tokens used/i)
    expect(meter.textContent).toContain("999")
  })

  it("formats 1500 as '1.5K'", () => {
    render(<TokenMeter used={1} max={1500} />)
    expect(screen.getByLabelText(/of 1.5K tokens used/i)).toBeTruthy()
  })

  it("formats 10000+ as rounded 'NK'", () => {
    render(<TokenMeter used={1} max={10000} />)
    expect(screen.getByLabelText(/of 10K tokens used/i)).toBeTruthy()
  })

  it("uses primary color for bar below 80%", () => {
    const { container } = render(<TokenMeter used={100} max={1000} />)
    const bar = container.querySelector("span > span > span") as HTMLElement
    expect(bar.className).toContain("bg-primary")
  })

  it("uses warn color for bar at 80–95%", () => {
    const { container } = render(<TokenMeter used={850} max={1000} />)
    const bar = container.querySelector("span > span > span") as HTMLElement
    expect(bar.className).toContain("bg-warn")
  })

  it("uses danger color for bar at ≥95%", () => {
    const { container } = render(<TokenMeter used={970} max={1000} />)
    const bar = container.querySelector("span > span > span") as HTMLElement
    expect(bar.className).toContain("bg-danger")
  })

  it("clamps ratio to [0, 1] when used > max", () => {
    const { container } = render(<TokenMeter used={5000} max={1000} />)
    const bar = container.querySelector("span > span > span") as HTMLElement
    expect(bar.style.width).toBe("100%")
  })

  it("handles max=0 without dividing by zero", () => {
    const { container } = render(<TokenMeter used={0} max={0} />)
    const bar = container.querySelector("span > span > span") as HTMLElement
    expect(bar.style.width).toBe("0%")
  })
})
