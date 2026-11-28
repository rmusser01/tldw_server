import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { DigestSchedulesPanel } from "../DigestSchedulesPanel"

const apiMock = vi.hoisted(() => ({
  listReadingDigestSchedules: vi.fn(),
  createReadingDigestSchedule: vi.fn(),
  updateReadingDigestSchedule: vi.fn(),
  deleteReadingDigestSchedule: vi.fn()
}))

const interpolate = (template: string, values?: Record<string, unknown>) => {
  if (!values) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
    const value = values[token]
    return value == null ? "" : String(value)
  })
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string } | Record<string, unknown>,
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return interpolate(fallbackOrOptions, maybeOptions)
      }
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        const maybeDefault = (fallbackOrOptions as { defaultValue?: string }).defaultValue
        if (typeof maybeDefault === "string") {
          return interpolate(maybeDefault, maybeOptions)
        }
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useTldwApiClient", () => ({
  useTldwApiClient: () => apiMock
}))

const clearInputNumberValue = (testId: string) => {
  const wrapper = screen.getByTestId(testId)
  const input =
    wrapper.tagName.toLowerCase() === "input"
      ? wrapper
      : (wrapper.querySelector("input") as HTMLInputElement | null)
  if (!input) throw new Error(`No input found for ${testId}`)
  fireEvent.change(input, { target: { value: "" } })
  fireEvent.blur(input)
}

describe("DigestSchedulesPanel suggestions form", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }

    apiMock.listReadingDigestSchedules.mockResolvedValue([])
    apiMock.createReadingDigestSchedule.mockResolvedValue({ id: "new-id" })
    apiMock.updateReadingDigestSchedule.mockImplementation(
      async (_scheduleId: string, payload: any) => ({
        id: "sched-1",
        name: payload?.name || "Digest",
        cron: payload?.cron || "0 8 * * *",
        timezone: payload?.timezone || "UTC",
        enabled: payload?.enabled ?? true,
        require_online: payload?.require_online ?? false,
        format: payload?.format || "md",
        filters: payload?.filters || null
      })
    )
    apiMock.deleteReadingDigestSchedule.mockResolvedValue({ ok: true })
  })

  it("announces the schedule load error separately from its retry control", async () => {
    apiMock.listReadingDigestSchedules.mockRejectedValueOnce(new Error("Schedule loading failed"))
    render(<DigestSchedulesPanel />)

    const error = await screen.findByRole("alert")
    expect(error.textContent).toBe("Schedule loading failed")
    expect(within(error).queryByRole("button")).toBeNull()

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
    expect(apiMock.listReadingDigestSchedules).toHaveBeenCalledTimes(2)
  })

  it("shows suggestions fields only when toggle is enabled", async () => {
    render(<DigestSchedulesPanel />)

    const openButton = await screen.findByRole("button", { name: "New Schedule" })
    fireEvent.click(openButton)

    expect(screen.queryByTestId("digest-suggestions-limit-input")).toBeNull()
    fireEvent.click(screen.getByTestId("digest-suggestions-toggle"))
    expect(await screen.findByTestId("digest-suggestions-limit-input")).toBeTruthy()
    expect(screen.getByTestId("digest-suggestions-status-select")).toBeTruthy()
    expect(screen.getByTestId("digest-suggestions-exclude-tags-input")).toBeTruthy()
    expect(screen.getByTestId("digest-suggestions-max-age-input")).toBeTruthy()
    expect(screen.getByTestId("digest-suggestions-include-read-toggle")).toBeTruthy()
    expect(screen.getByTestId("digest-suggestions-include-archived-toggle")).toBeTruthy()
  })

  it("stores suggestions config under filters.suggestions in create payload", async () => {
    render(<DigestSchedulesPanel />)

    fireEvent.click(await screen.findByRole("button", { name: "New Schedule" }))
    fireEvent.click(screen.getByTestId("digest-suggestions-toggle"))

    await screen.findByTestId("digest-suggestions-limit-input")
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(apiMock.createReadingDigestSchedule).toHaveBeenCalledTimes(1)
    })

    const payload = apiMock.createReadingDigestSchedule.mock.calls[0][0]
    expect(payload).toEqual(
      expect.objectContaining({
        filters: expect.objectContaining({
          suggestions: expect.objectContaining({
            enabled: true,
            limit: 5,
            status: ["saved", "reading"],
            include_read: false,
            include_archived: false
          })
        })
      })
    )
  })

  it("rehydrates and preserves suggestions config on edit", async () => {
    apiMock.listReadingDigestSchedules.mockResolvedValueOnce([
      {
        id: "sched-1",
        name: "Daily Digest",
        cron: "0 7 * * *",
        timezone: "UTC",
        enabled: true,
        require_online: false,
        format: "md",
        filters: {
          suggestions: {
            enabled: true,
            limit: 7,
            status: ["reading", "saved"],
            exclude_tags: ["ignore-me"],
            max_age_days: 90,
            include_read: false,
            include_archived: true
          }
        }
      }
    ])

    render(<DigestSchedulesPanel />)
    expect(await screen.findByText("Daily Digest")).toBeTruthy()

    fireEvent.click(screen.getByRole("button", { name: "Edit" }))
    expect(await screen.findByTestId("digest-suggestions-limit-input")).toBeTruthy()

    const toggle = screen.getByTestId("digest-suggestions-toggle")
    expect(toggle.getAttribute("aria-checked")).toBe("true")

    fireEvent.change(screen.getByPlaceholderText("Daily reading digest"), {
      target: { value: "Daily Digest Updated" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(apiMock.updateReadingDigestSchedule).toHaveBeenCalledTimes(1)
    })

    const [scheduleId, payload] = apiMock.updateReadingDigestSchedule.mock.calls[0]
    expect(scheduleId).toBe("sched-1")
    expect(payload).toEqual(
      expect.objectContaining({
        filters: expect.objectContaining({
          suggestions: expect.objectContaining({
            enabled: true,
            limit: 7,
            status: ["reading", "saved"],
            exclude_tags: ["ignore-me"],
            max_age_days: 90,
            include_read: false,
            include_archived: true
          })
        })
      })
    )
  }, 30000)

  it("validates suggestions limit client-side against API constraints", async () => {
    render(<DigestSchedulesPanel />)

    fireEvent.click(await screen.findByRole("button", { name: "New Schedule" }))
    fireEvent.click(screen.getByTestId("digest-suggestions-toggle"))
    await screen.findByTestId("digest-suggestions-limit-input")

    clearInputNumberValue("digest-suggestions-limit-input")
    fireEvent.click(screen.getByRole("button", { name: "Create" }))

    expect(await screen.findByText("Suggestions limit is required")).toBeTruthy()
    expect(apiMock.createReadingDigestSchedule).not.toHaveBeenCalled()
  })
})
