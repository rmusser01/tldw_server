import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { CalendarFilterRail, type CalendarFilterRailProps } from "../CalendarFilterRail"

const calendars = [7, 42].map((id) => ({
  id, tenant_id: "default", owner_user_id: 1, org_id: null,
  name: id === 7 ? "Research" : "Team", color: null, timezone: "UTC", visibility: "private",
  created_at: "2026-06-01T00:00:00Z", updated_at: "2026-06-01T00:00:00Z"
}))

const filterProps = (): CalendarFilterRailProps => ({
  calendars, selectedCalendarIds: [7],
  selectedSources: ["local", "org", "provider", "linked"], selectedKinds: ["event", "todo"],
  onCalendarChange: vi.fn(), onSourceChange: vi.fn(), onKindChange: vi.fn()
})

describe("CalendarFilterRail emissions", () => {
  it("adds and removes numeric calendar IDs without changing source or kind filters", async () => {
    const user = userEvent.setup()
    const props = filterProps()
    const { rerender } = render(<CalendarFilterRail {...props} />)
    expect(screen.getByRole("checkbox", { name: "Research" })).toHaveProperty("checked", true)
    expect(screen.getByRole("checkbox", { name: "Team" })).toHaveProperty("checked", false)
    await user.click(screen.getByRole("checkbox", { name: "Team" }))
    expect(props.onCalendarChange).toHaveBeenLastCalledWith([7, 42])
    rerender(<CalendarFilterRail {...props} selectedCalendarIds={[7, 42]} />)
    await user.click(screen.getByRole("checkbox", { name: "Research" }))
    expect(props.onCalendarChange).toHaveBeenLastCalledWith([42])
    rerender(<CalendarFilterRail {...props} selectedCalendarIds={[42]} />)
    await user.click(screen.getByRole("checkbox", { name: "Team" }))
    expect(props.onCalendarChange).toHaveBeenLastCalledWith([])
    expect(props.onSourceChange).not.toHaveBeenCalled()
    expect(props.onKindChange).not.toHaveBeenCalled()
  })

  it.each([
    { label: "Local", remaining: ["org", "provider", "linked"] },
    { label: "Org", remaining: ["local", "provider", "linked"] },
    { label: "Provider", remaining: ["local", "org", "linked"] },
    { label: "Linked", remaining: ["local", "org", "provider"] }
  ] as const)("emits the source values when $label is removed and restored", async ({ label, remaining }) => {
    const user = userEvent.setup()
    const props = filterProps()
    const { rerender } = render(<CalendarFilterRail {...props} />)
    await user.click(screen.getByRole("checkbox", { name: label }))
    expect(props.onSourceChange).toHaveBeenLastCalledWith(remaining)
    rerender(<CalendarFilterRail {...props} selectedSources={[...remaining]} />)
    expect(screen.getByRole("checkbox", { name: label })).toHaveProperty("checked", false)
    await user.click(screen.getByRole("checkbox", { name: label }))
    expect(props.onSourceChange).toHaveBeenLastCalledWith(["local", "org", "provider", "linked"])
    expect(props.onCalendarChange).not.toHaveBeenCalled()
    expect(props.onKindChange).not.toHaveBeenCalled()
  })

  it.each([
    { label: "Events", remaining: ["todo"] },
    { label: "Todos", remaining: ["event"] }
  ] as const)("emits the kind values when $label is removed and restored", async ({ label, remaining }) => {
    const user = userEvent.setup()
    const props = filterProps()
    const { rerender } = render(<CalendarFilterRail {...props} />)
    await user.click(screen.getByRole("checkbox", { name: label }))
    expect(props.onKindChange).toHaveBeenLastCalledWith(remaining)
    rerender(<CalendarFilterRail {...props} selectedKinds={[...remaining]} />)
    await user.click(screen.getByRole("checkbox", { name: label }))
    expect(props.onKindChange).toHaveBeenLastCalledWith(["event", "todo"])
    expect(props.onCalendarChange).not.toHaveBeenCalled()
    expect(props.onSourceChange).not.toHaveBeenCalled()
  })
})
