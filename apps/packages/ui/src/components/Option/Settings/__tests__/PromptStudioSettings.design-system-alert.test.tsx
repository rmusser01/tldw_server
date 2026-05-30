import type React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PromptStudioSettings } from "../prompt-studio"

const expectDesignSystemAlert = (text: string | RegExp) => {
  const node =
    typeof text === "string"
      ? screen.getByText(text, { exact: false })
      : screen.getByText(text)

  expect(node.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
}

const queryState = vi.hoisted(() => ({
  mutationCallIndex: 0,
  capability: {
    data: true as boolean | undefined,
    isError: false,
    isLoading: false
  },
  defaults: {
    data: {
      defaultProjectId: null,
      autoSyncWorkspacePrompts: true,
      executeProvider: "openai",
      executeModel: "gpt-4o-mini",
      executeTemperature: 0.2,
      executeMaxTokens: 256,
      evalModelName: "gpt-4o-mini",
      evalTemperature: 0.2,
      evalMaxTokens: 512,
      pageSize: 10,
      warnSeconds: 30
    },
    isError: false,
    isLoading: false
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue ?? _key
    }
  })
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const formApi = {
    getFieldValue: vi.fn(() => 30),
    setFieldsValue: vi.fn()
  }
  const Form = Object.assign(
    ({
      children,
      onFinish
    }: {
      children?: React.ReactNode
      onFinish?: (values: Record<string, unknown>) => void
    }) => (
      <form
        onSubmit={(event) => {
          event.preventDefault()
          onFinish?.({ warnSeconds: 30 })
        }}
      >
        {children}
      </form>
    ),
    {
      Item: ({ children, label }: { children?: React.ReactNode; label?: React.ReactNode }) => (
        <div>
          {label ? <span>{label}</span> : null}
          {children}
        </div>
      ),
      useForm: () => [formApi]
    }
  )

  return {
    ...actual,
    Form
  }
})

vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ queryKey }: { queryKey: string[] }) => {
    if (queryKey.includes("capability")) return queryState.capability
    return queryState.defaults
  },
  useMutation: (options?: { onError?: (err: Error) => void }) => {
    queryState.mutationCallIndex += 1

    if (queryState.mutationCallIndex % 2 === 0) {
      return {
        isPending: false,
        mutate: vi.fn(() => {
          options?.onError?.(new Error("Prompt Studio status timed out"))
        })
      }
    }

    return {
      isPending: false,
      mutate: vi.fn()
    }
  }
}))

vi.mock("@/services/prompt-studio", () => ({
  getPromptStudioStatus: vi.fn(),
  hasPromptStudio: vi.fn()
}))

vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: vi.fn(),
  setPromptStudioDefaults: vi.fn()
}))

describe("PromptStudioSettings design-system alerts", () => {
  beforeEach(() => {
    queryState.mutationCallIndex = 0
    queryState.capability.data = true
    queryState.capability.isError = false
    queryState.capability.isLoading = false
    queryState.defaults.isLoading = false
  })

  it("renders capability probe failures through the design-system alert", () => {
    queryState.capability.data = undefined
    queryState.capability.isError = true

    render(<PromptStudioSettings />)

    expectDesignSystemAlert("Unable to reach Prompt Studio")
  })

  it("renders status endpoint failures through the design-system alert", async () => {
    render(<PromptStudioSettings />)

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Test Prompt Studio" }))
    })

    await waitFor(() => {
      expectDesignSystemAlert("Status endpoint unavailable")
    })
    expectDesignSystemAlert("Prompt Studio status timed out")
  })

  it("renders unavailable server guidance through the design-system alert", () => {
    queryState.capability.data = false

    render(<PromptStudioSettings />)

    expectDesignSystemAlert("Prompt Studio isn’t available on the server yet.")
    expectDesignSystemAlert(
      "Once enabled, you can monitor queue health and set defaults here."
    )
  })
})
