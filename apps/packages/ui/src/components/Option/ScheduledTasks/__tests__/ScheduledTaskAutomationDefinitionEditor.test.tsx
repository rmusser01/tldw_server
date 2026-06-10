// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import type {
  ScheduledTaskDefinitionResponse,
  ScheduledTaskPreviewResponse
} from "@/services/scheduled-tasks-control-plane"
import { ScheduledTaskAutomationDefinitionEditor } from "../ScheduledTaskAutomationDefinitionEditor"

const validPreview = (
  overrides: Partial<ScheduledTaskPreviewResponse> = {}
): ScheduledTaskPreviewResponse => ({
  id: "preview_1",
  mode: "create",
  status: "valid",
  family: "recurring_question",
  normalized_config: { name: "Track answer" },
  validation_errors: [],
  warnings: [],
  visibility_policy: { visibility: "private" },
  schedule_preview: { kind: "manual" },
  redaction_policy: { redacted_fields: [] },
  expires_at: "2026-06-10T00:00:00Z",
  ...overrides
})

const definitionResponse = (
  overrides: Partial<ScheduledTaskDefinitionResponse> = {}
): ScheduledTaskDefinitionResponse => ({
  id: "definition_1",
  version: 1,
  family: "recurring_question",
  name: "Track answer",
  description: null,
  lifecycle: "configured",
  health: "execution_unavailable",
  schedule: {},
  input: {},
  config: {},
  visibility_policy: { visibility: "private" },
  notification_policy: {},
  approval_policy: {},
  ...overrides
})

describe("ScheduledTaskAutomationDefinitionEditor", () => {
  const deferred = <T,>() => {
    let resolve!: (value: T) => void
    const promise = new Promise<T>((promiseResolve) => {
      resolve = promiseResolve
    })
    return { promise, resolve }
  }

  it("previews and creates a recurring question definition", async () => {
    const user = userEvent.setup()
    const onPreview = vi.fn().mockResolvedValue(validPreview())
    const onCreate = vi.fn().mockResolvedValue(definitionResponse())

    render(
      <ScheduledTaskAutomationDefinitionEditor
        family="recurring_question"
        mode="create"
        onPreview={onPreview}
        onCreate={onCreate}
        onCancel={vi.fn()}
      />
    )

    await user.type(screen.getByLabelText("Question"), "Has the answer appeared?")
    await user.click(screen.getByRole("button", { name: "Preview" }))
    await screen.findByText("Preview ready")
    await user.click(screen.getByRole("button", { name: "Save definition" }))

    expect(onPreview).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "create",
        family: "recurring_question",
        schedule: expect.objectContaining({
          kind: "daily",
          timezone: "UTC"
        }),
        input: expect.objectContaining({
          question: "Has the answer appeared?"
        })
      })
    )
    expect(onCreate).toHaveBeenCalledWith({
      preview_id: "preview_1",
      initial_lifecycle: "configured"
    })
    expect(await screen.findByText("Execution is not available yet")).toBeInTheDocument()
  })

  it("shows agent task preview redaction copy", async () => {
    const user = userEvent.setup()
    const onPreview = vi.fn().mockResolvedValue(
      validPreview({
        family: "agent_task",
        normalized_config: { name: "Dispatch agent" },
        redaction_policy: {
          redacted_fields: ["agent_ref.api_key", "message.secret"]
        }
      })
    )

    render(
      <ScheduledTaskAutomationDefinitionEditor
        family="agent_task"
        mode="create"
        onPreview={onPreview}
        onCreate={vi.fn()}
        onCancel={vi.fn()}
      />
    )

    fireEvent.change(screen.getByLabelText("Agent ref"), {
      target: { value: '{"agent_id":"agent_1","api_key":"secret"}' }
    })
    await user.type(screen.getByLabelText("Message"), "Summarize the private report")
    await user.click(screen.getByRole("button", { name: "Preview" }))

    expect(await screen.findByText("Preview ready")).toBeInTheDocument()
    expect(onPreview).toHaveBeenCalledWith(
      expect.objectContaining({
        input: expect.objectContaining({
          agent_ref: '{"agent_id":"agent_1","api_key":"secret"}'
        })
      })
    )
    expect(screen.getByText("Redacted: agent_ref.api_key, message.secret")).toBeInTheDocument()
  })

  it("does not allow an older in-flight preview to become saveable after fields change", async () => {
    const user = userEvent.setup()
    const firstPreview = deferred<ScheduledTaskPreviewResponse>()
    const onPreview = vi.fn().mockReturnValue(firstPreview.promise)
    const onCreate = vi.fn().mockResolvedValue(definitionResponse())

    render(
      <ScheduledTaskAutomationDefinitionEditor
        family="recurring_question"
        mode="create"
        onPreview={onPreview}
        onCreate={onCreate}
        onCancel={vi.fn()}
      />
    )

    await user.type(screen.getByLabelText("Question"), "Has the answer appeared?")
    await user.click(screen.getByRole("button", { name: "Preview" }))
    fireEvent.change(screen.getByLabelText("Question"), {
      target: { value: "Has the answer changed?" }
    })
    firstPreview.resolve(validPreview({ id: "preview_stale" }))

    await waitFor(() => expect(onPreview).toHaveBeenCalledTimes(1))
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Save definition" })).toBeDisabled()
    })
    expect(screen.queryByText("Preview ready")).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Save definition" }))
    expect(onCreate).not.toHaveBeenCalled()
  })

  it("does not report saved when the create mutation handler is missing", async () => {
    const user = userEvent.setup()
    const onPreview = vi.fn().mockResolvedValue(validPreview())
    const onSaved = vi.fn()

    render(
      <ScheduledTaskAutomationDefinitionEditor
        family="recurring_question"
        mode="create"
        onPreview={onPreview}
        onCancel={vi.fn()}
        onSaved={onSaved}
      />
    )

    await user.type(screen.getByLabelText("Question"), "Has the answer appeared?")
    await user.click(screen.getByRole("button", { name: "Preview" }))
    await screen.findByText("Preview ready")
    await user.click(screen.getByRole("button", { name: "Save definition" }))

    expect(await screen.findByText("Create handler is not configured")).toBeInTheDocument()
    expect(onSaved).not.toHaveBeenCalled()
  })

  it("blocks preview and save when scope JSON is malformed", async () => {
    const user = userEvent.setup()
    const onPreview = vi.fn().mockResolvedValue(validPreview())
    const onCreate = vi.fn().mockResolvedValue(definitionResponse())

    render(
      <ScheduledTaskAutomationDefinitionEditor
        family="recurring_question"
        mode="create"
        onPreview={onPreview}
        onCreate={onCreate}
        onCancel={vi.fn()}
      />
    )

    await user.type(screen.getByLabelText("Question"), "Has the answer appeared?")
    fireEvent.change(screen.getByLabelText("Scope JSON"), {
      target: { value: '{"collection_id":' }
    })
    await user.click(screen.getByRole("button", { name: "Preview" }))

    expect(await screen.findByText("Scope JSON must be a valid JSON object")).toBeInTheDocument()
    expect(onPreview).not.toHaveBeenCalled()
    expect(screen.getByRole("button", { name: "Save definition" })).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "Save definition" }))
    expect(onCreate).not.toHaveBeenCalled()
  })

  it("prompts for another preview when the preview is expired", async () => {
    const user = userEvent.setup()
    const onCreate = vi.fn()
    const onPreview = vi.fn().mockResolvedValue(
      validPreview({
        status: "expired",
        validation_errors: [{ message: "Preview expired" }]
      })
    )

    render(
      <ScheduledTaskAutomationDefinitionEditor
        family="recurring_question"
        mode="create"
        onPreview={onPreview}
        onCreate={onCreate}
        onCancel={vi.fn()}
      />
    )

    await user.type(screen.getByLabelText("Question"), "Has the answer appeared?")
    await user.click(screen.getByRole("button", { name: "Preview" }))

    expect(await screen.findByText("Preview again before saving")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save definition" })).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "Save definition" }))
    expect(onCreate).not.toHaveBeenCalled()
  })

  it("keeps save disabled until preview is valid", async () => {
    const user = userEvent.setup()
    const onPreview = vi.fn().mockResolvedValue(
      validPreview({
        status: "invalid",
        validation_errors: [{ field: "question", message: "Question is required" }]
      })
    )

    render(
      <ScheduledTaskAutomationDefinitionEditor
        family="recurring_question"
        mode="create"
        onPreview={onPreview}
        onCreate={vi.fn()}
        onCancel={vi.fn()}
      />
    )

    expect(screen.getByRole("button", { name: "Save definition" })).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "Preview" }))

    expect(await screen.findByText("Question is required")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save definition" })).toBeDisabled()
  })

  it("sends only preview id and lifecycle when creating", async () => {
    const user = userEvent.setup()
    const onCreate = vi.fn().mockResolvedValue(definitionResponse())

    render(
      <ScheduledTaskAutomationDefinitionEditor
        family="recurring_question"
        mode="create"
        onPreview={vi.fn().mockResolvedValue(validPreview({ id: "preview_create" }))}
        onCreate={onCreate}
        onCancel={vi.fn()}
      />
    )

    await user.type(screen.getByLabelText("Question"), "Has the answer appeared?")
    await user.click(screen.getByRole("button", { name: "Preview" }))
    await screen.findByText("Preview ready")
    await user.click(screen.getByRole("button", { name: "Save definition" }))

    expect(onCreate).toHaveBeenCalledTimes(1)
    expect(onCreate).toHaveBeenCalledWith({
      preview_id: "preview_create",
      initial_lifecycle: "configured"
    })
  })

  it("sends only preview id when updating", async () => {
    const user = userEvent.setup()
    const onUpdate = vi.fn().mockResolvedValue(definitionResponse())

    render(
      <ScheduledTaskAutomationDefinitionEditor
        family="agent_task"
        mode="update"
        definitionId="definition_1"
        definitionVersion={3}
        initialValues={{
          name: "Dispatch agent",
          agentRef: '{"agent_id":"agent_1"}',
          message: "Summarize the report"
        }}
        onPreview={vi.fn().mockResolvedValue(
          validPreview({
            id: "preview_update",
            family: "agent_task",
            mode: "update"
          })
        )}
        onUpdate={onUpdate}
        onCancel={vi.fn()}
      />
    )

    await user.click(screen.getByRole("button", { name: "Preview" }))
    await screen.findByText("Preview ready")
    await user.click(screen.getByRole("button", { name: "Save definition" }))

    await waitFor(() => expect(onUpdate).toHaveBeenCalledTimes(1))
    expect(onUpdate).toHaveBeenCalledWith({ preview_id: "preview_update" })
  })
})
