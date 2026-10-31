import React, { Suspense } from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  request: vi.fn(),
  config: { serverUrl: "http://claims.test", authMode: "single-user", apiKey: "test-key" },
  authorityLoading: false
}))
vi.mock("next/dynamic", async () => {
  const ReactModule = await vi.importActual<typeof import("react")>("react")
  return { default: (loader: () => Promise<{ default: React.ComponentType }>) => ReactModule.lazy(loader) }
})
vi.mock("@web/components/navigation/RouteRedirect", () => ({
  RouteRedirect: () => <p>Content draft redirect</p>
}))
vi.mock("@/components/Layouts/Layout", () => ({ default: ({ children }: React.PropsWithChildren) => <>{children}</> }))
vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({ children }: React.PropsWithChildren) => <>{children}</>
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({ config: mocks.config, loading: false, authorityLoading: mocks.authorityLoading })
}))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.request(...args)
}))
import ClaimsReviewPage from "@web/pages/claims-review"

const claim = (id: number, status = "pending", version = 1) => ({
  id, media_id: 7, media_title: "Rowan source", chunk_index: 0, span_start: 0, span_end: 29,
  claim_text: id === 1 ? "Its director is Dr. Mira Vale." : `Rowan claim ${id}.`,
  review_status: status, review_version: version
})
const history = [{
  id: 1, claim_id: 1, old_status: "pending", new_status: "approved", reviewer_id: 9,
  notes: "Source confirms the director", reason_code: "supported", created_at: "2026-10-09T12:00:00Z"
}]
const mount = () => render(<Suspense fallback={<p>Loading route</p>}><ClaimsReviewPage /></Suspense>)
const queueReady = async () => {
  await screen.findByRole("heading", { name: "Claims Review" })
  await screen.findByRole("button", { name: "Inspect claim 1" })
}
const inspect = async (id = 1) => {
  fireEvent.click(await screen.findByRole("button", { name: `Inspect claim ${id}` }))
  await screen.findByLabelText("Review notes")
}
const calls = (method: string) => mocks.request.mock.calls.map(([request]) => request).filter((request) => request.method === method)

beforeEach(() => {
  mocks.config = { serverUrl: "http://claims.test", authMode: "single-user", apiKey: "test-key" }
  mocks.authorityLoading = false
  mocks.request.mockReset()
  mocks.request.mockImplementation(async (request) => {
    const path = String(request.path)
    if (path.startsWith("/api/v1/claims/review-queue")) return { items: [claim(1), claim(2), claim(3)], has_more: false, next_offset: null }
    if (path.endsWith("/history")) return []
    if (path.startsWith("/api/v1/claims/items/")) return claim(1)
    if (request.method === "PATCH") return claim(1, request.body.status, 2)
    if (request.method === "POST") return { updated: [1], conflicts: [2], missing: [], invalid: [] }
    throw new Error("Unexpected Claims request")
  })
})
afterEach(cleanup)

describe("Claims browser decision workflow", () => {
  it("mounts the real route and reads the owner-scoped queue without redirecting to drafts", async () => {
    mount()
    await queueReady()
    expect(screen.queryByText("Content draft redirect")).not.toBeInTheDocument()
    expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({
      path: expect.stringContaining("/api/v1/claims/review-queue?"),
      configSnapshot: mocks.config
    }))
    expect(screen.getByText("3 claims loaded")).toBeInTheDocument()
  })

  it("saves a versioned supported decision with reason, notes and inspectable actor history", async () => {
    mocks.request.mockImplementation(async (request) => {
      if (String(request.path).includes("review-queue")) return { items: [claim(1), claim(2)], has_more: false, next_offset: null }
      if (String(request.path).endsWith("/history")) return calls("PATCH").length ? history : []
      return claim(1, "approved", 2)
    })
    mount()
    await inspect()
    fireEvent.change(screen.getByLabelText("Review notes"), { target: { value: "Source confirms the director" } })
    fireEvent.change(screen.getByLabelText("Reason code"), { target: { value: "supported" } })
    fireEvent.click(screen.getByRole("button", { name: "Approve claim" }))
    await screen.findByText("Reviewer 9")
    expect(calls("PATCH")[0].body).toEqual({ status: "approved", review_version: 1, notes: "Source confirms the director", reason_code: "supported" })
    expect(screen.getByRole("button", { name: "Inspect claim 2" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Inspect claim 1" })).not.toBeInTheDocument()
  })

  it("saves an unsupported rejection and an optional corrected text without changing assignment", async () => {
    mount()
    await inspect()
    fireEvent.change(screen.getByLabelText("Reason code"), { target: { value: "unsupported" } })
    fireEvent.change(screen.getByLabelText("Corrected claim text"), { target: { value: "The director is Mira Vale." } })
    fireEvent.click(screen.getByRole("button", { name: "Reject claim" }))
    await waitFor(() => expect(calls("PATCH")).toHaveLength(1))
    expect(calls("PATCH")[0].body).toEqual({ status: "rejected", review_version: 1, reason_code: "unsupported", corrected_text: "The director is Mira Vale." })
    expect(calls("PATCH")[0].body).not.toHaveProperty("reviewer_id")
  })

  it("requires an explicit latest-version read after a conflict and preserves the draft", async () => {
    let attempt = 0
    mocks.request.mockImplementation(async (request) => {
      if (String(request.path).includes("review-queue")) return { items: [claim(1)], has_more: false, next_offset: null }
      if (String(request.path).endsWith("/history")) return []
      if (request.method === "PATCH" && ++attempt === 1) throw { status: 409 }
      if (request.method === "GET") return claim(1, "flagged", 4)
      return claim(1, "approved", 5)
    })
    mount()
    await inspect()
    fireEvent.change(screen.getByLabelText("Review notes"), { target: { value: "Keep my notes" } })
    fireEvent.click(screen.getByRole("button", { name: "Approve claim" }))
    await screen.findByText(/changed since it was loaded/i)
    expect(calls("PATCH")).toHaveLength(1)
    expect(screen.getByRole("button", { name: "Approve claim" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Load latest claim" }))
    await waitFor(() => expect(screen.getByRole("button", { name: "Approve claim" })).toBeEnabled())
    expect(screen.getByLabelText("Review notes")).toHaveValue("Keep my notes")
    fireEvent.click(screen.getByRole("button", { name: "Approve claim" }))
    await waitFor(() => expect(calls("PATCH")).toHaveLength(2))
    expect(calls("PATCH")[1].body.review_version).toBe(4)
  })

  it("reports partial selected-only bulk effects and retains failed selections", async () => {
    mount()
    await queueReady()
    fireEvent.click(screen.getByLabelText("Select claim 1"))
    fireEvent.click(screen.getByLabelText("Select claim 2"))
    fireEvent.change(screen.getByLabelText("Batch reason code"), { target: { value: "supported" } })
    fireEvent.click(screen.getByRole("button", { name: "Approve selected claims" }))
    await screen.findByText(/1 saved; 1 conflict; 0 missing; 0 invalid/i)
    expect(calls("POST")[0].body).toEqual({ claim_ids: [1, 2], status: "approved", reason_code: "supported" })
    expect(screen.getByLabelText("Select claim 2")).toBeChecked()
    expect(screen.getByLabelText("Select claim 3")).not.toBeChecked()
    expect(screen.getByText("2 claims loaded")).toBeInTheDocument()
  })

  it("blocks queue data and mutations on permission denial", async () => {
    mocks.request.mockRejectedValue({ status: 403 })
    mount()
    await screen.findByText(/permission to review Claims/i)
    expect(screen.queryByRole("button", { name: /Inspect claim/ })).not.toBeInTheDocument()
    expect(calls("PATCH")).toHaveLength(0)
    expect(calls("POST")).toHaveLength(0)
  })

  it("reports an admin-only batch denial without retaining protected items", async () => {
    const normal = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request) => request.method === "POST" ? Promise.reject({ status: 403 }) : normal(request))
    mount()
    await queueReady()
    fireEvent.click(screen.getByLabelText("Select claim 1"))
    fireEvent.click(screen.getByRole("button", { name: "Approve selected claims" }))
    await screen.findByText(/permission to review Claims/i)
    expect(screen.queryByRole("button", { name: /Inspect claim/ })).not.toBeInTheDocument()
  })

  it("loads real next offsets without inventing a total", async () => {
    mocks.request.mockImplementation(async (request) => String(request.path).includes("offset=25")
      ? { items: [claim(2)], has_more: false, next_offset: null }
      : { items: [claim(1)], has_more: true, next_offset: 25 })
    mount()
    await queueReady()
    fireEvent.click(screen.getByRole("button", { name: "Load more claims" }))
    await screen.findByRole("button", { name: "Inspect claim 2" })
    expect(screen.getByText("2 claims loaded")).toBeInTheDocument()
    expect(mocks.request.mock.calls[1][0].path).toContain("offset=25")
  })

  it("clears selections and detail drafts when the review filter changes", async () => {
    mount()
    await inspect()
    fireEvent.click(screen.getByLabelText("Select claim 1"))
    fireEvent.change(screen.getByLabelText("Review notes"), { target: { value: "Old filter draft" } })
    fireEvent.change(screen.getByLabelText("Review status"), { target: { value: "rejected" } })
    await waitFor(() => expect(mocks.request.mock.calls.some(([r]) => String(r.path).includes("status_filter=rejected"))).toBe(true))
    expect(screen.queryByLabelText("Review notes")).not.toBeInTheDocument()
    expect(screen.getByLabelText("Select claim 1")).not.toBeChecked()
  })

  it("retires a delayed response and local decisions on account changes", async () => {
    let resolveOld: (value: unknown) => void = () => {}
    mocks.request.mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve }))
    const view = mount()
    await waitFor(() => expect(mocks.request).toHaveBeenCalledTimes(1))
    mocks.config = { ...mocks.config, apiKey: "other-account" }
    view.rerender(<Suspense fallback={<p>Loading route</p>}><ClaimsReviewPage /></Suspense>)
    await queueReady()
    await act(async () => resolveOld({ items: [{ ...claim(8), claim_text: "Private old account claim" }], has_more: false, next_offset: null }))
    expect(screen.queryByText("Private old account claim")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Inspect claim 8" })).not.toBeInTheDocument()
  })

  it("does not offer a direct invalid reversal of an approved claim", async () => {
    mocks.request.mockImplementation(async (request) => String(request.path).endsWith("/history") ? [] : { items: [claim(1, "approved", 2)], has_more: false, next_offset: null })
    mount()
    await inspect()
    expect(screen.queryByRole("button", { name: "Reject claim" })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Return claim to pending" })).toBeInTheDocument()
  })
})
