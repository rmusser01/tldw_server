import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { DataManagementSettings } from "../system-settings"

const mutateMock = vi.fn()

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue ?? key
    }
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useMutation: () => ({
    isPending: false,
    mutate: mutateMock
  }),
  useQueryClient: () => ({
    invalidateQueries: vi.fn()
  })
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    clearChat: vi.fn()
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    destroy: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    success: vi.fn()
  })
}))

vi.mock("@/utils/is-private-mode", () => ({
  isFireFox: false,
  isFireFoxPrivateMode: false
}))

describe("DataManagementSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("keeps import, export, and typed-reset actions on the data surface", () => {
    render(<DataManagementSettings />)

    expect(
      screen.getByRole("heading", { name: /data management/i })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /export data/i })).toBeInTheDocument()
    expect(screen.getByText(/import data/i)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /reset all/i })).toBeInTheDocument()
  })
})
