import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { CalendarOwnershipBadge, getCalendarOwnershipLabel } from "../CalendarOwnershipBadge"

const calendar = {
  id: 1, tenant_id: "default", owner_user_id: 1, org_id: null,
  name: "Research", color: null, timezone: "UTC", visibility: "private",
  created_at: "2026-06-01T00:00:00Z", updated_at: "2026-06-01T00:00:00Z"
}

describe("Calendar ownership classification and badges", () => {
  it.each([
    { scenario: "personal local item", source_owner: "tldw", provider_owned: false, calendar, expected: "Local" },
    { scenario: "missing calendar", source_owner: "tldw", calendar: undefined, expected: "Local" },
    { scenario: "null calendar", source_owner: "tldw", calendar: null, expected: "Local" },
    { scenario: "organization item", source_owner: "tldw", calendar: { ...calendar, org_id: 9 }, expected: "Org" },
    { scenario: "provider source in an organization", source_owner: "provider", calendar: { ...calendar, org_id: 9 }, expected: "Provider" },
    { scenario: "provider-owned flag in an organization", source_owner: "tldw", provider_owned: true, calendar: { ...calendar, org_id: 9 }, expected: "Provider" },
    { scenario: "linked projection without a calendar", source_owner: "linked_projection", calendar: null, expected: "Linked" },
    { scenario: "linked projection takes priority over provider and organization", source_owner: "linked_projection", provider_owned: true, calendar: { ...calendar, org_id: 9 }, expected: "Linked" },
    { scenario: "readonly local item stays local", source_owner: "tldw", read_only_reason: "Read only", calendar, expected: "Local" }
  ])("classifies and renders $scenario as $expected", ({ scenario: _scenario, expected, calendar: itemCalendar, ...ownership }) => {
    const item = { calendar_id: itemCalendar?.id ?? null, ...ownership }
    expect(getCalendarOwnershipLabel(item, itemCalendar)).toBe(expected)
    const { container } = render(<CalendarOwnershipBadge item={item} calendar={itemCalendar} />)
    expect(screen.getByText(expected)).toBeTruthy()
    expect(container.textContent).toBe(expected)
  })
})
