import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ClaimsReviewShell } from "../ClaimsReviewShell"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"

const mocks = vi.hoisted(() => ({
  request: vi.fn(), config: { serverUrl: "http://claims.test", authMode: "single-user", apiKey: "key-a" },
  authorityLoading: false,
  resolveScope: vi.fn()
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({ config: mocks.config, loading: false, authorityLoading: mocks.authorityLoading })
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => mocks.request(...args) }))
vi.mock("@/services/service-prompts", () => ({ resolveServicePromptScope: (...args: unknown[]) => mocks.resolveScope(...args) }))
const claim = (id: number) => ({ id, media_id: id + 10, claim_text: `Claim ${id}`, chunk_index: 2, span_start: 4, span_end: 12, review_status: "pending", review_version: 3 })
const inspect = async (id = 1) => { fireEvent.click(await screen.findByRole("button", { name: `Inspect claim ${id}` })); await screen.findByLabelText("Review notes") }
beforeEach(() => {
  mocks.config = { serverUrl: "http://claims.test", authMode: "single-user", apiKey: "key-a" }
  mocks.authorityLoading = false
  mocks.resolveScope.mockReset().mockImplementation(async () => ({
    config: { ...mocks.config, expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope(mocks.config.authMode, mocks.config.apiKey) },
    userId: "alice", scopeKey: "claims-alice", clientPrincipalVerified: true
  }))
  mocks.request.mockReset()
  mocks.request.mockImplementation(async (request) => {
    if (String(request.path).startsWith("/api/v1/claims/review-queue?")) return { items: [claim(1), claim(2)], has_more: false, next_offset: null }
    if (String(request.path).endsWith("/history")) return []
    if (String(request.path).startsWith("/api/v1/media/")) return { content: { text: "<script>source text</script>" } }
    if (request.method === "PATCH") return { ...claim(1), review_status: request.body.status, review_version: 4 }
    if (String(request.path).startsWith("/api/v1/claims/items/")) return claim(1)
    throw new Error("Unexpected request")
  })
})
afterEach(cleanup)
describe("shared Claims review guards", () => {
  it.each(["tldw:auth-principal-changed", "tldw:auth-credentials-changed"])(
    "retires unchanged cookie authority immediately on %s and rejects late private history",
    async (eventName) => {
      Object.assign(mocks.config, { authMode: "multi-user", authSource: "cookie-session", apiKey: "" })
      let finishHistory: (value: unknown) => void = () => {}
      const normal = mocks.request.getMockImplementation()!
      mocks.request.mockImplementation((request) => String(request.path).endsWith("/history")
        ? new Promise((resolve) => { finishHistory = resolve }) : normal(request))
      render(<ClaimsReviewShell />)
      await inspect()
      const signal = mocks.request.mock.calls.find(([r]) => String(r.path).endsWith("/history"))![0].abortSignal
      mocks.resolveScope.mockImplementation(() => new Promise(() => {}))
      act(() => window.dispatchEvent(new CustomEvent(eventName)))
      expect(signal.aborted).toBe(true)
      expect(screen.queryByLabelText("Review notes")).not.toBeInTheDocument()
      await act(async () => finishHistory([{ id: 90, new_status: "approved", notes: "Old cookie account history", created_at: "today" }]))
      expect(screen.queryByText("Old cookie account history")).not.toBeInTheDocument()
    }
  )

  it("uses the real query delimiter and bounded current source transport, rendering source as text", async () => {
    render(<ClaimsReviewShell />)
    await inspect()
    expect(mocks.request.mock.calls[0][0].path).toBe("/api/v1/claims/review-queue?status_filter=pending&envelope=true&limit=25&offset=0")
    fireEvent.click(screen.getByRole("button", { name: "Load current source text" }))
    await screen.findByText("<script>source text</script>")
    const source = mocks.request.mock.calls.find(([r]) => String(r.path).startsWith("/api/v1/media/"))![0]
    expect(source).toMatchObject({ path: "/api/v1/media/11?include_content=true&include_versions=false", maxResponseBytes: 1048576, configSnapshot: mocks.config })
    expect(document.querySelector("script")).toBeNull()
  })

  it("ignores a delayed history from the previously inspected claim", async () => {
    let resolveOld: (value: unknown) => void = () => {}
    const normal = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request) => String(request.path) === "/api/v1/claims/1/history"
      ? new Promise((resolve) => { resolveOld = resolve }) : normal(request))
    render(<ClaimsReviewShell />)
    await inspect(1)
    await inspect(2)
    await act(async () => resolveOld([{ id: 90, new_status: "approved", reviewer_id: 99, notes: "Old private history", created_at: "today" }]))
    expect(screen.queryByText("Old private history")).not.toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Claim 2 · pending" })).toBeInTheDocument()
  })

  it("ignores a delayed source from the previously inspected claim", async () => {
    let resolveOld: (value: unknown) => void = () => {}
    const normal = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request) => String(request.path).startsWith("/api/v1/media/11?")
      ? new Promise((resolve) => { resolveOld = resolve }) : normal(request))
    render(<ClaimsReviewShell />)
    await inspect(1)
    fireEvent.click(screen.getByRole("button", { name: "Load current source text" }))
    await inspect(2)
    await act(async () => resolveOld({ content: { text: "Private old source" } }))
    expect(screen.queryByText("Private old source")).not.toBeInTheDocument()
  })

  it("retains a decision draft after an unconfirmed request and requires an explicit latest read", async () => {
    const normal = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request) => request.method === "PATCH" ? Promise.reject({ status: 503 }) : normal(request))
    render(<ClaimsReviewShell />)
    await inspect()
    fireEvent.change(screen.getByLabelText("Review notes"), { target: { value: "Retained draft" } })
    fireEvent.click(screen.getByRole("button", { name: "Approve claim" }))
    await screen.findByText(/Decision was not saved/i)
    expect(screen.getByLabelText("Review notes")).toHaveValue("Retained draft")
    expect(screen.getByRole("button", { name: "Approve claim" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Load latest claim" }))
    await waitFor(() => expect(screen.getByRole("button", { name: "Approve claim" })).toBeEnabled())
    expect(mocks.request.mock.calls.filter(([r]) => r.method === "PATCH")).toHaveLength(1)
  })

  it("prevents duplicate decisions while the acknowledgement is pending", async () => {
    let finish: (value: unknown) => void = () => {}
    const normal = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request) => request.method === "PATCH" ? new Promise((resolve) => { finish = resolve }) : normal(request))
    render(<ClaimsReviewShell />)
    await inspect()
    const approve = screen.getByRole("button", { name: "Approve claim" })
    fireEvent.click(approve)
    fireEvent.click(approve)
    expect(mocks.request.mock.calls.filter(([r]) => r.method === "PATCH")).toHaveLength(1)
    expect(approve).toBeDisabled()
    await act(async () => finish({ ...claim(1), review_status: "approved", review_version: 4 }))
  })

  it("retires private content and aborts requests during canonical authority rehydration", async () => {
    let finishHistory: (value: unknown) => void = () => {}
    const normal = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request) => String(request.path).endsWith("/history")
      ? new Promise((resolve) => { finishHistory = resolve }) : normal(request))
    const view = render(<ClaimsReviewShell />)
    await inspect()
    const signal = mocks.request.mock.calls.find(([r]) => String(r.path).endsWith("/history"))![0].abortSignal
    mocks.authorityLoading = true
    view.rerender(<ClaimsReviewShell />)
    expect(screen.queryByLabelText("Review notes")).not.toBeInTheDocument()
    expect(screen.getByRole("status")).toHaveTextContent("Resolving Claims review connection")
    expect(signal.aborted).toBe(true)
    await act(async () => finishHistory([{ id: 90, new_status: "approved", notes: "Old account decision", created_at: "today" }]))
    expect(screen.queryByText("Old account decision")).not.toBeInTheDocument()
  })

  it("refuses a mutation when the server has not supplied a valid review version", async () => {
    mocks.request.mockImplementation(async (request) => String(request.path).endsWith("/history") ? [] : {
      items: [{ ...claim(1), review_version: 0 }], has_more: false, next_offset: null
    })
    render(<ClaimsReviewShell />)
    await inspect()
    expect(screen.getByRole("button", { name: "Approve claim" })).toBeDisabled()
    expect(mocks.request.mock.calls.filter(([r]) => r.method === "PATCH")).toHaveLength(0)
  })

  it("reports a saved decision separately from unavailable audit history", async () => {
    const normal = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request) => String(request.path).endsWith("/history") ? Promise.reject({ status: 503 }) : normal(request))
    render(<ClaimsReviewShell />)
    await inspect()
    fireEvent.click(screen.getByRole("button", { name: "Approve claim" }))
    await screen.findByText("Claim 1 saved as approved.")
    expect(screen.getByText("Decision history could not be loaded.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry history" })).toBeInTheDocument()
  })

  it.each([401, 403])("retires a delayed decision after history permission denial %s and allows explicit refresh recovery", async (status) => {
    let rejectHistory: (cause: unknown) => void = () => {}
    let finishDecision: (value: unknown) => void = () => {}
    let historyCalls = 0
    const normal = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request) => {
      if (String(request.path).endsWith("/history")) {
        historyCalls += 1
        if (historyCalls === 1) return new Promise((_resolve, reject) => { rejectHistory = reject })
      }
      if (request.method === "PATCH") return new Promise((resolve) => { finishDecision = resolve })
      return normal(request)
    })
    render(<ClaimsReviewShell />)
    await inspect()
    fireEvent.click(screen.getByRole("button", { name: "Approve claim" }))
    const decisionSignal = mocks.request.mock.calls.find(([request]) => request.method === "PATCH")![0].abortSignal
    await act(async () => rejectHistory({ status }))
    await screen.findByText(/You do not have permission to review Claims/)
    await act(async () => finishDecision({ ...claim(1), review_status: "approved", review_version: 4 }))
    expect(screen.queryByRole("heading", { name: "Claim 1 · approved" })).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Review notes")).not.toBeInTheDocument()
    expect(screen.queryByText("Claim 1 saved as approved.")).not.toBeInTheDocument()
    expect(historyCalls).toBe(1)
    expect(decisionSignal.aborted).toBe(true)
    fireEvent.click(screen.getByRole("button", { name: "Refresh queue" }))
    await inspect()
    expect(screen.getByRole("heading", { name: "Claim 1 · pending" })).toBeInTheDocument()
    expect(screen.queryByText(/You do not have permission to review Claims/)).not.toBeInTheDocument()
  })

  it("retires a delayed decision after source authentication denial", async () => {
    let rejectSource: (cause: unknown) => void = () => {}
    let finishDecision: (value: unknown) => void = () => {}
    const normal = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request) => {
      if (String(request.path).startsWith("/api/v1/media/")) return new Promise((_resolve, reject) => { rejectSource = reject })
      if (request.method === "PATCH") return new Promise((resolve) => { finishDecision = resolve })
      return normal(request)
    })
    render(<ClaimsReviewShell />)
    await inspect()
    fireEvent.click(screen.getByRole("button", { name: "Load current source text" }))
    fireEvent.click(screen.getByRole("button", { name: "Approve claim" }))
    const decisionSignal = mocks.request.mock.calls.find(([request]) => request.method === "PATCH")![0].abortSignal
    await act(async () => rejectSource({ status: 401 }))
    await act(async () => finishDecision({ ...claim(1), review_status: "approved", review_version: 4 }))
    expect(screen.queryByRole("heading", { name: "Claim 1 · approved" })).not.toBeInTheDocument()
    expect(screen.queryByLabelText("Review notes")).not.toBeInTheDocument()
    expect(screen.getByText(/You do not have permission to review Claims/)).toBeInTheDocument()
    expect(decisionSignal.aborted).toBe(true)
    expect(mocks.request.mock.calls.filter(([request]) => String(request.path).endsWith("/history"))).toHaveLength(1)
  })

  it("keeps Claims decisions available after a source-only 403 denial", async () => {
    const normal = mocks.request.getMockImplementation()!
    mocks.request.mockImplementation((request) => String(request.path).startsWith("/api/v1/media/")
      ? Promise.reject({ status: 403 }) : normal(request))
    render(<ClaimsReviewShell />)
    await inspect()
    fireEvent.click(screen.getByRole("button", { name: "Load current source text" }))
    await screen.findByText("Current source text is unavailable. Check source access or open it in Media.")
    expect(screen.getByRole("button", { name: "Approve claim" })).toBeEnabled()
    fireEvent.click(screen.getByRole("button", { name: "Approve claim" }))
    await screen.findByText("Claim 1 saved as approved.")
  })
})
