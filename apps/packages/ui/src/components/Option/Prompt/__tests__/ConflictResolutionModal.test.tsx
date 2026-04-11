import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConflictResolutionModal } from "../ConflictResolutionModal"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: Record<string, any>) =>
      options?.defaultValue ?? _key
  })
}))

vi.mock("antd", () => ({
  Modal: ({ open, title, children, footer }: any) =>
    open ? (
      <div>
        <h1>{title}</h1>
        <div>{children}</div>
        <div>{footer}</div>
      </div>
    ) : null,
  Alert: ({ message, description }: any) => (
    <div>
      <p>{message}</p>
      {description ? <p>{description}</p> : null}
    </div>
  ),
  Tag: ({ children }: any) => <span>{children}</span>
}))

describe("ConflictResolutionModal", () => {
  it("renders local/server prompt details and resolves with selected strategy", async () => {
    const user = userEvent.setup()
    const onResolve = vi.fn()

    render(
      <ConflictResolutionModal
        open
        conflictInfo={{
          localPrompt: {
            id: "local-1",
            title: "Local Title",
            name: "Local Title",
            content: "local content",
            is_system: true,
            system_prompt: "local system",
            user_prompt: "local user",
            createdAt: 1,
            updatedAt: 10
          },
          serverPrompt: {
            id: 1,
            project_id: 7,
            name: "Server Title",
            system_prompt: "server system",
            user_prompt: "server user",
            version_number: 2,
            updated_at: "2026-02-17T10:00:00Z"
          },
          localUpdatedAt: 10,
          serverUpdatedAt: "2026-02-17T10:00:00Z"
        }}
        onClose={vi.fn()}
        onResolve={onResolve}
      />
    )

    expect(screen.getByText("Resolve conflict")).toBeInTheDocument()
    expect(screen.getByText("Local version")).toBeInTheDocument()
    expect(screen.getByText("Server version")).toBeInTheDocument()
    expect(screen.getAllByText("Changed").length).toBeGreaterThan(0)

    await user.click(screen.getByRole("button", { name: "Keep mine" }))
    expect(onResolve).toHaveBeenCalledWith("keep_local")
  })

  it("disables resolution actions when details are unavailable", () => {
    render(
      <ConflictResolutionModal
        open
        loading={false}
        conflictInfo={null}
        onClose={vi.fn()}
        onResolve={vi.fn()}
      />
    )

    expect(screen.getByText("Conflict details unavailable")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Keep mine" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Keep server" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Keep both" })).toBeDisabled()
  })
})
