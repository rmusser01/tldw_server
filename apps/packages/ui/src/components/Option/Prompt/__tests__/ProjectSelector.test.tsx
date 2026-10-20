import React from "react"
import { describe, expect, it, vi, beforeEach } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ProjectSelector } from "../ProjectSelector"

const createProjectMock = vi.fn()
const invalidateQueriesMock = vi.fn()
const useQueryMock = vi.fn()

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallbackOrOptions?: any, maybeOptions?: any) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue ?? _key
      }
      if (maybeOptions && typeof maybeOptions === "object") {
        return maybeOptions.defaultValue ?? _key
      }
      return _key
    }
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/services/prompt-studio", () => ({
  createProject: (...args: unknown[]) =>
    (createProjectMock as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: (...args: unknown[]) =>
    (useQueryMock as (...args: unknown[]) => unknown)(...args),
  useQueryClient: () => ({
    invalidateQueries: (...args: unknown[]) =>
      (invalidateQueriesMock as (...args: unknown[]) => unknown)(...args)
  }),
  useMutation: (options: any) => ({
    mutate: async (value: any) => {
      try {
        const result = await options.mutationFn(value)
        await options.onSuccess?.(result)
      } catch (error) {
        options.onError?.(error)
      }
    },
    isPending: false
  })
}))

vi.mock("antd", () => ({
  Modal: ({ open, children }: any) => (open ? <div>{children}</div> : null),
  Select: () => <div data-testid="project-select" />,
  Empty: ({ description }: any) => <div>{description}</div>,
  Alert: ({ title, description }: {
    title?: React.ReactNode
    description?: React.ReactNode
  }) => <div role="alert">{title}{description}</div>,
  Skeleton: () => <div>Loading</div>,
  Space: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, onClick, disabled, loading, type }: any) => (
    <button onClick={onClick} disabled={disabled || loading} data-type={type}>
      {children}
    </button>
  ),
  Input: ({ value, onChange, placeholder }: any) => (
    <input
      value={value}
      onChange={onChange}
      placeholder={placeholder}
    />
  ),
  notification: {
    error: vi.fn(),
    warning: vi.fn()
  }
}))

describe("ProjectSelector", () => {
  beforeEach(() => {
    createProjectMock.mockReset()
    invalidateQueriesMock.mockReset()
    useQueryMock.mockReset()
    useQueryMock.mockReturnValue({
      data: [],
      isLoading: false
    })
  })

  it("shows a project lookup failure without offering empty-list project creation", () => {
    useQueryMock.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("Authentication required")
    })

    render(<ProjectSelector open onClose={vi.fn()} onSelect={vi.fn()} />)

    expect(screen.getByRole("alert")).toHaveTextContent("Authentication required")
    expect(screen.queryByRole("button", { name: "Create project" })).not.toBeInTheDocument()
    expect(screen.queryByPlaceholderText("Enter project name")).not.toBeInTheDocument()
    expect(createProjectMock).not.toHaveBeenCalled()
  })

  it("creates a project inline from empty state and selects it", async () => {
    const user = userEvent.setup()
    const onSelect = vi.fn()

    createProjectMock.mockResolvedValue({
      data: {
        data: {
          id: 42,
          name: "My Sync Project"
        }
      }
    })

    render(
      <ProjectSelector
        open
        onClose={vi.fn()}
        onSelect={onSelect}
      />
    )

    await user.type(
      screen.getByPlaceholderText("Enter project name"),
      "My Sync Project"
    )
    await user.click(screen.getByRole("button", { name: "Create project" }))

    await waitFor(() => {
      expect(createProjectMock).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "My Sync Project"
        })
      )
      expect(onSelect).toHaveBeenCalledWith(42)
    })
    expect(invalidateQueriesMock).toHaveBeenCalled()
  })
})
