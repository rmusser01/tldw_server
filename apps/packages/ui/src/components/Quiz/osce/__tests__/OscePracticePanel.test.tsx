import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  OscePracticePanel,
  computeElapsedSeconds,
  computeLiveElapsedSeconds
} from "../OscePracticePanel"
import type { OsceAttempt } from "@/services/osce"
import {
  useBeginOsceSelfAssessmentMutation,
  useCompleteOsceAttemptMutation,
  useOsceAttemptQuery,
  usePatchOsceAttemptMutation
} from "../../hooks/useOsceQueries"
import { osceDraftKey, saveOsceDraft } from "../osceDraftStore"

const online = vi.hoisted(() => ({ value: true }))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, value?: string | { defaultValue?: string }) =>
      typeof value === "string" ? value : value?.defaultValue ?? _key
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => online.value }))

vi.mock("../../hooks/useOsceQueries", () => ({
  useOsceAttemptQuery: vi.fn(),
  usePatchOsceAttemptMutation: vi.fn(),
  useBeginOsceSelfAssessmentMutation: vi.fn(),
  useCompleteOsceAttemptMutation: vi.fn()
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const candidate: OsceAttempt = {
  id: 7,
  quiz_id: 3,
  station_id: 11,
  client_attempt_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
  state: "in_progress",
  version: 2,
  notes: "",
  started_at: "2026-09-11T10:00:00Z",
  last_modified_at: "2026-09-11T10:02:00Z",
  server_time: "2026-09-11T10:02:00Z",
  station: {
    schema_version: "osce.station.v1",
    title: "Medicine safety",
    candidate_instructions: "Use **plain language**.",
    candidate_task: "Explain safe medicine use.",
    patient_context: {
      text: "A fictional adult recently started treatment.",
      citations: [{ source_type: "document", source_id: "doc-1", label: "Patient leaflet" }]
    },
    recommended_duration_seconds: 480
  }
}

const revealed: OsceAttempt = {
  ...candidate,
  state: "self_assessment",
  version: 3,
  self_assessment_started_at: "2026-09-11T10:08:00Z",
  completed_at: null,
  elapsed_seconds: 480,
  checklist_selections: {},
  rubric_selections: {},
  station: {
    ...candidate.station,
    checklist_items: [
      {
        id: "check-1",
        label: "Explain the medicine purpose",
        rationale: "Supports informed use.",
        citations: []
      }
    ],
    rubric_domains: [{
      id: "domain-1",
      label: "Communication",
      levels: [
        { id: "level-1", label: "Needs development", description: "The explanation was incomplete." },
        { id: "level-2", label: "Effective", description: "The explanation was clear." }
      ]
    }],
    expected_key_points: [{
      id: "point-1",
      text: "Discuss warning signs.",
      citations: [{
        source_type: "url",
        source_id: "guide-1",
        label: "Safety guidance",
        source_url: "https://example.test/safety"
      }]
    }]
  }
}

describe("OSCE timer", () => {
  it("uses server timestamps across reload, local clock jumps, background suspension, reveal, and resume", () => {
    expect(computeElapsedSeconds({
      startedAt: "2026-09-11T10:00:00Z",
      serverNow: "2026-09-11T10:05:00Z"
    })).toBe(300)
    expect(computeElapsedSeconds({
      startedAt: "2026-09-11T10:00:00Z",
      serverNow: "2026-09-11T10:07:00Z"
    })).toBe(420)

    const baseline = {
      serverElapsedSeconds: 300,
      baselineMonotonicMs: 10_000
    }
    expect(computeLiveElapsedSeconds({ ...baseline, monotonicNowMs: 15_000 })).toBe(305)
    expect(computeLiveElapsedSeconds({ ...baseline, monotonicNowMs: 45_000 })).toBe(335)
    expect(computeLiveElapsedSeconds({
      ...baseline,
      monotonicNowMs: 90_000,
      frozenElapsedSeconds: 480
    })).toBe(480)
    expect(computeElapsedSeconds({
      startedAt: "2026-09-11T10:00:00Z",
      serverNow: "2026-09-11T10:10:00Z"
    })).toBe(600)
  })
})

describe("OscePracticePanel", () => {
  const patch = vi.fn()
  const begin = vi.fn()
  const complete = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    online.value = true
    vi.mocked(useOsceAttemptQuery).mockReturnValue({
      data: candidate,
      isLoading: false,
      isError: false,
      refetch: vi.fn().mockResolvedValue({ data: candidate })
    } as any)
    patch.mockImplementation(async (_variables: unknown) => ({ ...candidate, version: 3 }))
    begin.mockResolvedValue(revealed)
    complete.mockResolvedValue({
      ...revealed,
      state: "completed",
      completed_at: "2026-09-11T10:12:00Z",
      checklist_selections: { "check-1": "met" },
      rubric_selections: { "domain-1": "level-2" },
      version: 6
    })
    vi.mocked(usePatchOsceAttemptMutation).mockReturnValue({ mutateAsync: patch, isPending: false } as any)
    vi.mocked(useBeginOsceSelfAssessmentMutation).mockReturnValue({ mutateAsync: begin, isPending: false } as any)
    vi.mocked(useCompleteOsceAttemptMutation).mockReturnValue({ mutateAsync: complete, isPending: false } as any)
  })

  it("renders only the candidate projection before reveal", () => {
    vi.mocked(useOsceAttemptQuery).mockReturnValue({
      data: {
        ...candidate,
        station: {
          ...candidate.station,
          expected_key_points: [{ text: "Hidden answer" }],
          checklist_items: [{ label: "Hidden checklist" }],
          rubric_domains: [{ label: "Hidden rubric" }]
        }
      },
      isLoading: false,
      isError: false,
      refetch: vi.fn().mockResolvedValue({ data: candidate })
    } as any)

    render(<OscePracticePanel attemptId={7} userScope="user-42" saveDebounceMs={0} />)

    expect(screen.getByText("Medicine safety")).toBeInTheDocument()
    expect(screen.getByText("plain language", { exact: false })).toBeInTheDocument()
    expect(screen.getByText("Explain safe medicine use.")).toBeInTheDocument()
    expect(screen.getByText("Patient leaflet")).toBeInTheDocument()
    expect(screen.getByLabelText("Private practice notes")).toBeInTheDocument()
    expect(screen.getByText(/Do not enter real patient information/i)).toBeInTheDocument()
    expect(screen.queryByText("Hidden answer")).not.toBeInTheDocument()
    expect(screen.queryByText("Hidden checklist")).not.toBeInTheDocument()
    expect(screen.queryByText("Hidden rubric")).not.toBeInTheDocument()
  })

  it("focuses an accessible reveal confirmation and flushes notes before transition", async () => {
    const user = userEvent.setup()
    render(<OscePracticePanel attemptId={7} userScope="user-42" saveDebounceMs={0} />)

    await user.type(screen.getByLabelText("Private practice notes"), "Check understanding")
    await user.click(screen.getByRole("button", { name: "Begin self-assessment" }))

    const dialog = await screen.findByRole("dialog", { name: "Reveal marking guide?" })
    const confirm = within(dialog).getByRole("button", { name: "Reveal marking guide" })
    await waitFor(() => expect(confirm).toHaveFocus())
    await user.click(confirm)

    await waitFor(() => expect(patch).toHaveBeenCalled())
    expect(begin).toHaveBeenCalledWith({ attemptId: 7, expectedVersion: 3 })
  })

  it("stops reveal on save error, retains the draft, and exposes a focusable error", async () => {
    patch.mockRejectedValue(Object.assign(new Error("conflict"), { status: 409 }))
    const user = userEvent.setup()
    render(<OscePracticePanel attemptId={7} userScope="user-42" saveDebounceMs={0} />)

    await user.type(screen.getByLabelText("Private practice notes"), "Unsaved note")
    await user.click(screen.getByRole("button", { name: "Begin self-assessment" }))
    await user.click(await screen.findByRole("button", { name: "Reveal marking guide" }))

    const error = await screen.findByRole("alert")
    await waitFor(() => expect(error).toHaveAttribute("tabindex", "-1"))
    await waitFor(() => expect(error).toHaveFocus())
    expect(error).toHaveTextContent(/changed on the server|could not be saved/i)
    expect(begin).not.toHaveBeenCalled()
    expect(window.localStorage.getItem(osceDraftKey("user-42", 7))).toContain("Unsaved note")
  })

  it("replays a restored draft against its saved version", async () => {
    saveOsceDraft("user-42", {
      attemptId: 7,
      version: 2,
      notes: "Offline note",
      checklistSelections: {},
      rubricSelections: {},
      savedAt: Date.now()
    })
    vi.mocked(useOsceAttemptQuery).mockReturnValue({
      data: { ...candidate, version: 3, notes: "Server note" },
      isLoading: false,
      isError: false,
      refetch: vi.fn().mockResolvedValue({ data: { ...candidate, version: 3, notes: "Server note" } })
    } as any)
    patch.mockRejectedValue(Object.assign(new Error("conflict"), { status: 409 }))

    render(<OscePracticePanel attemptId={7} userScope="user-42" saveDebounceMs={0} />)

    await waitFor(() => expect(patch).toHaveBeenCalledWith(expect.objectContaining({
      patch: expect.objectContaining({ expected_version: 2, notes: "Offline note" })
    })))
    expect(screen.getByLabelText("Private practice notes")).toHaveValue("Offline note")
    expect(window.localStorage.getItem(osceDraftKey("user-42", 7))).toContain("Offline note")
  })

  it("does not replace newer local typing with an older save acknowledgement", async () => {
    let resolveFirst: ((attempt: OsceAttempt) => void) | null = null
    patch.mockImplementationOnce(() => new Promise<OsceAttempt>((resolve) => {
      resolveFirst = resolve
    }))

    render(<OscePracticePanel attemptId={7} userScope="user-42" saveDebounceMs={0} />)

    const notes = screen.getByLabelText("Private practice notes")
    fireEvent.change(notes, { target: { value: "First draft" } })
    await waitFor(() => expect(patch).toHaveBeenCalledTimes(1))
    fireEvent.change(notes, { target: { value: "Newer local draft" } })

    await act(async () => {
      resolveFirst?.({ ...candidate, version: 3, notes: "First draft" })
    })

    expect(notes).toHaveValue("Newer local draft")
  })

  it("requires an explicit choice before reapplying a conflicted draft to refetched state", async () => {
    const conflict = Object.assign(new Error("conflict"), { status: 409 })
    const latestServerAttempt = { ...candidate, version: 3, notes: "Concurrent server note" }
    const refetch = vi.fn().mockResolvedValue({ data: latestServerAttempt })
    vi.mocked(useOsceAttemptQuery).mockReturnValue({
      data: candidate,
      isLoading: false,
      isError: false,
      refetch
    } as any)
    patch
      .mockRejectedValueOnce(conflict)
      .mockResolvedValueOnce({ ...latestServerAttempt, version: 4, notes: "Local note" })

    render(<OscePracticePanel attemptId={7} userScope="user-42" saveDebounceMs={0} />)
    fireEvent.change(screen.getByLabelText("Private practice notes"), {
      target: { value: "Local note" }
    })

    expect(await screen.findByRole("button", { name: "Reapply local draft" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Use server changes" })).toBeInTheDocument()
    expect(refetch).toHaveBeenCalledTimes(1)
    expect(patch).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByRole("button", { name: "Reapply local draft" }))

    await waitFor(() => expect(patch).toHaveBeenCalledTimes(2))
    expect(patch).toHaveBeenLastCalledWith(expect.objectContaining({
      patch: expect.objectContaining({ expected_version: 3, notes: "Local note" })
    }))
  })

  it("requires a live connection to reveal or complete", () => {
    online.value = false
    render(<OscePracticePanel attemptId={7} userScope="user-42" />)

    expect(screen.getByRole("button", { name: "Begin self-assessment" })).toBeDisabled()
    expect(screen.getByRole("status")).toHaveTextContent(/offline/i)
  })

  it("renders self-assessment radio groups and enables Complete only after every selection", async () => {
    vi.mocked(useOsceAttemptQuery).mockReturnValue({
      data: revealed,
      isLoading: false,
      isError: false,
      refetch: vi.fn().mockResolvedValue({ data: revealed })
    } as any)
    patch
      .mockResolvedValueOnce({
        ...revealed,
        version: 4,
        checklist_selections: { "check-1": "met" }
      })
      .mockResolvedValueOnce({
        ...revealed,
        version: 5,
        checklist_selections: { "check-1": "met" },
        rubric_selections: { "domain-1": "level-2" }
      })
    const user = userEvent.setup()
    render(<OscePracticePanel attemptId={7} userScope="user-42" saveDebounceMs={0} />)

    expect(screen.getByText("Discuss warning signs.")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Safety guidance" })).toHaveAttribute(
      "href",
      "https://example.test/safety"
    )
    expect(screen.getByRole("radiogroup", { name: "Explain the medicine purpose" })).toBeInTheDocument()
    expect(screen.getByRole("radiogroup", { name: "Communication" })).toBeInTheDocument()
    const completeButton = screen.getByRole("button", { name: "Complete practice" })
    expect(completeButton).toBeDisabled()

    await user.click(screen.getByRole("radio", { name: "Met" }))
    await act(async () => {})
    expect(completeButton).toBeDisabled()
    await user.click(screen.getByRole("radio", { name: /Effective/ }))
    await waitFor(() => expect(completeButton).toBeEnabled())
    await user.click(completeButton)

    await waitFor(() => expect(complete).toHaveBeenCalledWith({ attemptId: 7, expectedVersion: 5 }))
  })

  it("ignores stale drafts and keeps completed attempts read-only", async () => {
    const completedAttempt: OsceAttempt = {
      ...revealed,
      state: "completed",
      version: 6,
      notes: "Final server note",
      completed_at: "2026-09-11T10:12:00Z",
      checklist_selections: { "check-1": "met" },
      rubric_selections: { "domain-1": "level-2" }
    }
    saveOsceDraft("user-42", {
      attemptId: 7,
      version: 5,
      notes: "Stale local note",
      checklistSelections: { "check-1": "not_met" },
      rubricSelections: { "domain-1": "level-1" },
      savedAt: Date.now()
    })
    vi.mocked(useOsceAttemptQuery).mockReturnValue({
      data: completedAttempt,
      isLoading: false,
      isError: false,
      refetch: vi.fn().mockResolvedValue({ data: completedAttempt })
    } as any)

    render(<OscePracticePanel attemptId={7} userScope="user-42" saveDebounceMs={0} />)

    const notes = screen.getByLabelText("Private practice notes")
    expect(notes).toHaveValue("Final server note")
    expect(notes).toBeDisabled()
    expect(screen.getByRole("radio", { name: "Met" })).toBeDisabled()
    expect(screen.getByRole("radio", { name: /Effective/ })).toBeDisabled()
    expect(window.localStorage.getItem(osceDraftKey("user-42", 7))).toBeNull()

    fireEvent.change(notes, { target: { value: "Post-completion edit" } })
    fireEvent.click(screen.getByRole("radio", { name: "Not met" }))
    fireEvent.click(screen.getByRole("radio", { name: /Needs development/ }))

    await waitFor(() => expect(patch).not.toHaveBeenCalled())
    expect(notes).toHaveValue("Final server note")
    expect(window.localStorage.getItem(osceDraftKey("user-42", 7))).toBeNull()
  })
})
