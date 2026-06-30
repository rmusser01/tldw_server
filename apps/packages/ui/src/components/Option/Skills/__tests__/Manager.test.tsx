import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { message, Modal } from "antd"
import { SkillsManager } from "../Manager"

const tldwClientMock = vi.hoisted(() => ({
  listSkills: vi.fn(),
  getSkill: vi.fn(),
  deleteSkill: vi.fn(),
  bulkDeleteSkills: vi.fn(),
  exportSkill: vi.fn(),
  previewSkillImport: vi.fn(),
  previewSkillImportFile: vi.fn(),
  importSkill: vi.fn(),
  importSkillFile: vi.fn(),
  seedSkills: vi.fn()
}))

const notificationMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn()
}))

const skillDrawerMock = vi.hoisted(() => vi.fn())
const skillPreviewMock = vi.hoisted(() => vi.fn())

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        const defaultValue = fallbackOrOptions.defaultValue || key
        return defaultValue.replace(/\{\{(\w+)\}\}/g, (_match, token: string) =>
          String(fallbackOrOptions[token] ?? "")
        )
      }
      return key
    }
  })
}))

vi.mock("../SkillDrawer", () => ({
  SkillDrawer: (props: {
    open: boolean
    onClose: () => void
    onSaved: (skillName?: string) => void
  }) => {
    skillDrawerMock(props)
    return props.open ? (
      <div data-testid="skill-drawer-open">
        Skill drawer open
        <button type="button" onClick={props.onClose}>
          Cancel drawer
        </button>
        <button type="button" onClick={() => props.onSaved("created-skill")}>
          Complete create
        </button>
      </div>
    ) : null
  }
}))

vi.mock("../SkillPreview", () => ({
  SkillPreview: (props: {
    skillName: string | null
    runtime?: unknown
    onClose: () => void
  }) => {
    skillPreviewMock(props)
    return props.skillName ? (
      <div data-testid="skill-preview-open">
        Test run: {props.skillName}
        <button type="button" onClick={props.onClose}>
          Close test run
        </button>
      </div>
    ) : null
  }
}))

const makeSkill = (index: number) => ({
  name: `skill-${index}`,
  description: `Skill ${index}`,
  argument_hint: null,
  user_invocable: true,
  disable_model_invocation: false,
  context: "inline" as const,
  version: index + 1
})

describe("SkillsManager imports", () => {
  let queryClient: QueryClient
  let originalClipboard: Clipboard | undefined

  beforeEach(() => {
    originalClipboard = navigator.clipboard
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    vi.clearAllMocks()
    window.localStorage.clear()
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: vi.fn().mockResolvedValue(undefined)
      }
    })
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [],
      count: 0,
      total: 0,
      limit: 10,
      offset: 0
    })
    tldwClientMock.previewSkillImport.mockResolvedValue({
      valid: true,
      errors: [],
      name: "imported-skill",
      description: "imported",
      argument_hint: null,
      disable_model_invocation: false,
      user_invocable: true,
      allowed_tools: null,
      model: null,
      context: "inline",
      supporting_file_count: 0,
      conflict: false,
      can_overwrite: false,
      existing_version: null
    })
    tldwClientMock.previewSkillImportFile.mockResolvedValue({
      valid: true,
      errors: [],
      name: "imported-file-skill",
      description: "file import",
      argument_hint: null,
      disable_model_invocation: false,
      user_invocable: true,
      allowed_tools: null,
      model: null,
      context: "inline",
      supporting_file_count: 0,
      conflict: false,
      can_overwrite: false,
      existing_version: null
    })
    tldwClientMock.importSkill.mockResolvedValue({ name: "imported-skill" })
    tldwClientMock.importSkillFile.mockResolvedValue({ name: "imported-file-skill" })
    tldwClientMock.seedSkills.mockResolvedValue({
      seeded: [" summarize ", "code-review", "feynman-technique"],
      count: 3
    })

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

    if (typeof globalThis.ResizeObserver === "undefined") {
      globalThis.ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
      } as unknown as typeof ResizeObserver
    }
  })

  afterEach(() => {
    cleanup()
    Modal.destroyAll()
    message.destroy()
    if (originalClipboard) {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: originalClipboard
      })
    } else {
      Reflect.deleteProperty(navigator, "clipboard")
    }
  })

  const renderManager = () =>
    render(
      <QueryClientProvider client={queryClient}>
        <SkillsManager />
      </QueryClientProvider>
    )

  const openColumnVisibilityMenu = async () => {
    const existingMenu = screen.queryByRole("menu")
    if (existingMenu) return existingMenu

    fireEvent.click(screen.getByRole("button", { name: "Column visibility" }))
    return screen.findByRole("menu")
  }

  const getColumnVisibilityOption = async (name: string) =>
    within(await openColumnVisibilityMenu()).findByRole("menuitem", { name })

  const selectSkillRow = (name: string) => {
    const row = screen.getByText(name).closest("tr")
    expect(row).not.toBeNull()
    fireEvent.click(within(row as HTMLTableRowElement).getByRole("checkbox"))
  }

  it("orients users with a page summary and library count", async () => {
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [
        {
          name: "summarize",
          description: "Summarize source material",
          context: "inline",
          source: "builtin",
          path: "skills/summarize/SKILL.md"
        },
        {
          name: "code-review",
          description: "Review code changes",
          context: "fork",
          source: "builtin",
          path: "skills/code-review/SKILL.md"
        }
      ],
      count: 2,
      total: 2,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(
      await screen.findByRole("heading", {
        name: "Skills"
      })
    ).toBeInTheDocument()
    expect(
      screen.getByText("Discover, test, create, import, and manage reusable instructions.")
    ).toBeInTheDocument()
    expect(await screen.findByText("2 skills")).toBeInTheDocument()
  })

  it("announces Skills list loading without relying on the table spinner alone", async () => {
    let resolveList: (value: {
      skills: never[]
      count: number
      total: number
      limit: number
      offset: number
    }) => void = () => {}
    tldwClientMock.listSkills.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveList = resolve
        })
    )

    renderManager()

    const loadingStatus = await screen.findByText("Loading skills")
    expect(loadingStatus.closest('[role="status"]')).toBeInTheDocument()

    resolveList({
      skills: [],
      count: 0,
      total: 0,
      limit: 10,
      offset: 0
    })

    expect(
      await screen.findByRole("heading", {
        name: "Start with a reusable skill"
      })
    ).toBeInTheDocument()
  })

  it("shows a Skills-specific beginner empty state with first actions", async () => {
    renderManager()

    expect(
      await screen.findByRole("heading", {
        name: "Start with a reusable skill"
      })
    ).toBeInTheDocument()
    expect(
      screen.getByText("Skills are reusable instructions that can be tested here and used from chat.")
    ).toBeInTheDocument()

    const emptyState = screen.getByTestId("skills-empty-state")
    fireEvent.click(within(emptyState).getByRole("button", { name: "Seed built-ins" }))

    await waitFor(() => {
      expect(tldwClientMock.seedSkills).toHaveBeenCalledWith({ overwrite: false })
    })

    fireEvent.click(within(emptyState).getByRole("button", { name: "Create from template" }))
    expect(await screen.findByTestId("skill-drawer-open")).toBeInTheDocument()

    expect(within(emptyState).queryByRole("button", { name: "Import" })).not.toBeInTheDocument()
    fireEvent.click(within(emptyState).getByRole("button", { name: "Import from text" }))
    expect(
      await screen.findByRole("dialog", {
        name: "Import Skill from Text"
      })
    ).toBeInTheDocument()
  })

  it("shows a shared recovery state with diagnostics instead of the beginner empty state", async () => {
    tldwClientMock.listSkills.mockRejectedValueOnce(
      Object.assign(new Error("backend down"), { status: 503 })
    )

    renderManager()

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveAttribute("data-ds-component", "RecoveryCallout")
    expect(alert).toHaveTextContent("Failed to load skills")
    expect(alert).toHaveTextContent(
      "The Skills list could not be loaded. Try again or open diagnostics."
    )
    const diagnostics = within(alert).getByLabelText("Diagnostics")
    expect(diagnostics).toHaveTextContent("GET")
    expect(diagnostics).toHaveTextContent("[server-endpoint]")
    expect(diagnostics).not.toHaveTextContent("/api/v1/skills")
    expect(diagnostics).toHaveTextContent("503")
    expect(diagnostics).toHaveTextContent("backend down")
    expect(screen.queryByTestId("skills-empty-state")).not.toBeInTheDocument()

    fireEvent.click(within(alert).getByRole("button", { name: "Try again" }))

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalledTimes(2)
    })
  })

  it("clamps stale pagination when totals shrink", async () => {
    const firstPageBeforeShrink = Array.from({ length: 10 }, (_, index) => makeSkill(index + 1))
    const firstPageAfterShrink = Array.from({ length: 5 }, (_, index) => makeSkill(index + 1))
    let firstPageCalls = 0

    tldwClientMock.listSkills.mockImplementation(({ offset }: { offset: number }) => {
      if (offset === 10) {
        return Promise.resolve({
          skills: [],
          count: 0,
          total: 5,
          limit: 10,
          offset: 10
        })
      }

      firstPageCalls += 1
      return Promise.resolve({
        skills: firstPageCalls === 1 ? firstPageBeforeShrink : firstPageAfterShrink,
        count: firstPageCalls === 1 ? 10 : 5,
        total: firstPageCalls === 1 ? 11 : 5,
        limit: 10,
        offset: 0
      })
    })

    renderManager()

    expect(await screen.findByText("11 skills")).toBeInTheDocument()

    const secondPageItem = await screen.findByTitle("2")
    fireEvent.click(within(secondPageItem).getByText("2"))

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalledWith(
        expect.objectContaining({ limit: 10, offset: 10 })
      )
    })
    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({ limit: 10, offset: 0 })
      )
    })
    expect(await screen.findByText("5 skills")).toBeInTheDocument()
    expect(screen.queryByText("No skills yet.")).not.toBeInTheDocument()
  })

  it("distinguishes a non-empty library with an empty current page", async () => {
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [],
      count: 0,
      total: 5,
      limit: 10,
      offset: 10
    })

    renderManager()

    expect(await screen.findByText("5 skills")).toBeInTheDocument()
    expect(screen.getByText("No skills on this page.")).toBeInTheDocument()
    expect(screen.queryByText("No skills yet.")).not.toBeInTheDocument()
    expect(screen.queryByTestId("skills-empty-state")).not.toBeInTheDocument()
  })

  it("uses server-backed search instead of filtering only the loaded page", async () => {
    const firstPage = Array.from({ length: 10 }, (_, index) => makeSkill(index + 1))
    const searchResult = {
      name: "omega-research",
      description: "Needle workflow for longform research synthesis",
      argument_hint: null,
      user_invocable: true,
      disable_model_invocation: false,
      context: "inline" as const
    }

    tldwClientMock.listSkills.mockImplementation(
      (params: { q?: string; limit: number; offset: number }) => {
        if (params.q === "needle") {
          return Promise.resolve({
            skills: [searchResult],
            count: 1,
            total: 1,
            limit: params.limit,
            offset: 0
          })
        }

        return Promise.resolve({
          skills: firstPage,
          count: firstPage.length,
          total: 12,
          limit: params.limit,
          offset: params.offset
        })
      }
    )

    renderManager()

    expect(await screen.findByText("12 skills")).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText("Search skills..."), {
      target: { value: "needle" }
    })

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({
          q: "needle",
          limit: 10,
          offset: 0
        })
      )
    })
    expect(await screen.findByText("omega-research")).toBeInTheDocument()
    expect(screen.getByText("1 skill")).toBeInTheDocument()
  })

  it("debounces server-backed search requests while typing", async () => {
    const firstPage = Array.from({ length: 10 }, (_, index) => makeSkill(index + 1))
    const searchResult = {
      name: "omega-research",
      description: "Needle workflow for longform research synthesis",
      argument_hint: null,
      user_invocable: true,
      disable_model_invocation: false,
      context: "inline" as const
    }

    tldwClientMock.listSkills.mockImplementation(
      (params: { q?: string; limit: number; offset: number }) => {
        if (params.q === "needle") {
          return Promise.resolve({
            skills: [searchResult],
            count: 1,
            total: 1,
            limit: params.limit,
            offset: 0
          })
        }

        return Promise.resolve({
          skills: firstPage,
          count: firstPage.length,
          total: 12,
          limit: params.limit,
          offset: params.offset
        })
      }
    )

    renderManager()

    expect(await screen.findByText("12 skills")).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText("Search skills..."), {
      target: { value: "needle" }
    })

    await new Promise((resolve) => setTimeout(resolve, 120))
    expect(tldwClientMock.listSkills).not.toHaveBeenCalledWith(
      expect.objectContaining({
        q: "needle",
        limit: 10,
        offset: 0
      })
    )

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({
          q: "needle",
          limit: 10,
          offset: 0
        })
      )
    })
    expect(await screen.findByText("omega-research")).toBeInTheDocument()
  })

  it("requests server-backed filter results from filter controls", async () => {
    const visibleSkill = {
      name: "visible-inline",
      description: "Visible inline skill",
      argument_hint: null,
      user_invocable: true,
      disable_model_invocation: false,
      context: "inline" as const
    }
    const hiddenForkSkill = {
      name: "hidden-fork-tools",
      description: "Hidden fork skill with tools",
      argument_hint: null,
      user_invocable: false,
      disable_model_invocation: false,
      context: "fork" as const
    }

    tldwClientMock.listSkills.mockImplementation(
      (params: {
        context?: string
        userInvocable?: boolean
        includeHidden?: boolean
        hasTools?: boolean
        limit: number
        offset: number
      }) => {
        if (
          params.context === "fork"
          && params.userInvocable === false
          && params.includeHidden === true
          && params.hasTools === true
        ) {
          return Promise.resolve({
            skills: [hiddenForkSkill],
            count: 1,
            total: 1,
            limit: params.limit,
            offset: 0
          })
        }

        return Promise.resolve({
          skills: [visibleSkill],
          count: 1,
          total: 2,
          limit: params.limit,
          offset: params.offset
        })
      }
    )

    renderManager()

    expect(await screen.findByText("2 skills")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Fork" }))
    fireEvent.click(screen.getByRole("button", { name: "Hidden" }))
    fireEvent.click(screen.getByRole("button", { name: "Has tools" }))

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({
          context: "fork",
          includeHidden: true,
          userInvocable: false,
          hasTools: true,
          limit: 10,
          offset: 0
        })
      )
    })
    expect(await screen.findByText("hidden-fork-tools")).toBeInTheDocument()
  })

  it("shows a filter empty state instead of onboarding when filters match no skills", async () => {
    const visibleSkill = {
      name: "visible-inline",
      description: "Visible inline skill",
      argument_hint: null,
      user_invocable: true,
      disable_model_invocation: false,
      context: "inline" as const
    }

    tldwClientMock.listSkills.mockImplementation(
      (params: { context?: string; limit: number; offset: number }) => {
        if (params.context === "fork") {
          return Promise.resolve({
            skills: [],
            count: 0,
            total: 0,
            limit: params.limit,
            offset: 0
          })
        }

        return Promise.resolve({
          skills: [visibleSkill],
          count: 1,
          total: 1,
          limit: params.limit,
          offset: params.offset
        })
      }
    )

    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Fork" }))

    expect(await screen.findByText("No skills match these filters.")).toBeInTheDocument()
    expect(screen.queryByTestId("skills-empty-state")).not.toBeInTheDocument()
  })

  it("requests server-backed name sorting from the table header", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [
        {
          name: "alpha",
          description: "Alpha skill",
          argument_hint: null,
          user_invocable: true,
          disable_model_invocation: false,
          context: "inline" as const
        },
        {
          name: "beta",
          description: "Beta skill",
          argument_hint: null,
          user_invocable: true,
          disable_model_invocation: false,
          context: "inline" as const
        }
      ],
      count: 2,
      total: 2,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByText("2 skills")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("columnheader", { name: /Name/ }))

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({
          sort: "name",
          order: "asc",
          limit: 10,
          offset: 0
        })
      )
    })
  })

  it("labels the row execution action as a test run", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    const testRunButton = screen.getByRole("button", { name: "Test run skill-1" })
    expect(screen.queryByRole("button", { name: "Preview skill-1" })).not.toBeInTheDocument()

    fireEvent.click(testRunButton)
    expect(screen.getByTestId("skill-preview-open")).toHaveTextContent("skill-1")
  })

  it("returns focus to the row test-run action after closing the test-run surface", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    const testRunButton = screen.getByRole("button", { name: "Test run skill-1" })
    testRunButton.focus()
    fireEvent.click(testRunButton)

    expect(screen.getByTestId("skill-preview-open")).toHaveTextContent("skill-1")
    fireEvent.click(screen.getByRole("button", { name: "Close test run" }))

    await waitFor(() => {
      expect(testRunButton).toHaveFocus()
    })
  })

  it("passes the row version when deleting a skill", async () => {
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        void config.onOk?.()
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [makeSkill(2)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.deleteSkill.mockResolvedValueOnce(undefined)

    try {
      renderManager()
      await screen.findByText("skill-2")
      fireEvent.click(screen.getByRole("button", { name: "Delete skill-2" }))

      await waitFor(() => {
        expect(tldwClientMock.deleteSkill).toHaveBeenCalledWith("skill-2", 3)
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("keeps delete compatible when a row has no known version", async () => {
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        void config.onOk?.()
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    const legacySkill = { ...makeSkill(4) } as Partial<ReturnType<typeof makeSkill>>
    delete legacySkill.version
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [legacySkill] as any,
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.deleteSkill.mockResolvedValueOnce(undefined)

    try {
      renderManager()
      await screen.findByText("skill-4")
      fireEvent.click(screen.getByRole("button", { name: "Delete skill-4" }))

      await waitFor(() => {
        expect(tldwClientMock.deleteSkill).toHaveBeenCalledWith("skill-4", undefined)
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("shows reload-before-delete guidance on stale delete conflict", async () => {
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")
    let confirmConfig: { onOk?: () => void | Promise<void> } | undefined
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        confirmConfig = config as { onOk?: () => void | Promise<void> }
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    const conflict = Object.assign(new Error("409 version conflict"), { status: 409 })
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.deleteSkill.mockRejectedValueOnce(conflict)

    try {
      renderManager()
      await screen.findByText("skill-1")
      fireEvent.click(screen.getByRole("button", { name: "Delete skill-1" }))

      await waitFor(() => {
        expect(confirmConfig?.onOk).toBeTypeOf("function")
      })
      await confirmConfig?.onOk?.()

      await waitFor(() => {
        expect(notificationMock.error).toHaveBeenCalledWith(
          expect.objectContaining({
            message: "Skill changed elsewhere",
            description: "Reload skills before deleting this version."
          })
        )
      })
      expect(invalidateSpy).toHaveBeenCalledWith(
        expect.objectContaining({ queryKey: ["skills"] })
      )
    } finally {
      confirmSpy.mockRestore()
      invalidateSpy.mockRestore()
    }
  })

  it("downloads exported skills with the server filename and success feedback", async () => {
    const originalCreateObjectURL = URL.createObjectURL
    const originalRevokeObjectURL = URL.revokeObjectURL
    let downloadedFilename = ""
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        downloadedFilename = this.download
      })
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:skills-export")
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn()
    })
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.exportSkill.mockResolvedValueOnce({
      blob: new Blob(["zip"], { type: "application/zip" }),
      filename: "server-skill.zip"
    })

    try {
      renderManager()
      await screen.findByText("skill-1")
      fireEvent.click(screen.getByRole("button", { name: "Export skill-1" }))

      await waitFor(() => {
        expect(tldwClientMock.exportSkill).toHaveBeenCalledWith("skill-1")
      })
      expect(downloadedFilename).toBe("server-skill.zip")
      expect(notificationMock.success).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Export started",
          description: "server-skill.zip download has started."
        })
      )
    } finally {
      clickSpy.mockRestore()
      if (originalCreateObjectURL) {
        Object.defineProperty(URL, "createObjectURL", {
          configurable: true,
          value: originalCreateObjectURL
        })
      } else {
        Reflect.deleteProperty(URL, "createObjectURL")
      }
      if (originalRevokeObjectURL) {
        Object.defineProperty(URL, "revokeObjectURL", {
          configurable: true,
          value: originalRevokeObjectURL
        })
      } else {
        Reflect.deleteProperty(URL, "revokeObjectURL")
      }
    }
  })

  it("sanitizes export failure notifications", async () => {
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.exportSkill.mockRejectedValueOnce(
      new Error(
        "Request failed: GET /api/v1/skills/skill-1/export?token=sk_live_secret from /Users/alice/.tldw with Bearer token_secret_123"
      )
    )

    renderManager()
    await screen.findByText("skill-1")
    fireEvent.click(screen.getByRole("button", { name: "Export skill-1" }))

    await waitFor(() => {
      expect(notificationMock.error).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Failed to export skill",
          description: expect.stringContaining("[server-endpoint]")
        })
      )
    })

    const payload = notificationMock.error.mock.calls.at(-1)?.[0] as {
      description?: string
    }
    expect(payload.description).toContain("[redacted-path]")
    expect(payload.description).toContain("Bearer [redacted-secret]")
    expect(payload.description).not.toContain("/api/v1/skills/skill-1/export")
    expect(payload.description).not.toContain("token=sk_live_secret")
    expect(payload.description).not.toContain("/Users/alice/.tldw")
    expect(payload.description).not.toContain("token_secret_123")
  })

  it("bulk deletes selected rows with their current versions", async () => {
    let confirmConfig: { onOk?: () => void | Promise<void> } | undefined
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        confirmConfig = config as { onOk?: () => void | Promise<void> }
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1), makeSkill(2)],
      count: 2,
      total: 2,
      limit: 10,
      offset: 0
    })
    tldwClientMock.bulkDeleteSkills.mockResolvedValueOnce({
      deleted: ["skill-1", "skill-2"],
      count: 2
    })

    try {
      renderManager()
      await screen.findByText("skill-1")
      selectSkillRow("skill-1")
      selectSkillRow("skill-2")

      expect(screen.getByTestId("skills-selection-actions")).toHaveTextContent("2 selected")
      fireEvent.click(screen.getByRole("button", { name: "Delete selected" }))

      await waitFor(() => {
        expect(confirmConfig?.onOk).toBeTypeOf("function")
      })
      await confirmConfig?.onOk?.()

      await waitFor(() => {
        expect(tldwClientMock.bulkDeleteSkills).toHaveBeenCalledWith([
          { name: "skill-1", version: 2 },
          { name: "skill-2", version: 3 }
        ])
      })
      expect(notificationMock.success).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Skills deleted",
          description: "2 skill(s) deleted."
        })
      )
      await waitFor(() => {
        expect(screen.queryByTestId("skills-selection-actions")).not.toBeInTheDocument()
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("bulk delete stays compatible when selected rows have no known version", async () => {
    let confirmConfig: { onOk?: () => void | Promise<void> } | undefined
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        confirmConfig = config as { onOk?: () => void | Promise<void> }
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    const legacySkill = { ...makeSkill(4) } as Partial<ReturnType<typeof makeSkill>>
    delete legacySkill.version
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [legacySkill] as any,
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.bulkDeleteSkills.mockResolvedValueOnce({
      deleted: ["skill-4"],
      count: 1
    })

    try {
      renderManager()
      await screen.findByText("skill-4")
      selectSkillRow("skill-4")
      fireEvent.click(screen.getByRole("button", { name: "Delete selected" }))

      await waitFor(() => {
        expect(confirmConfig?.onOk).toBeTypeOf("function")
      })
      await confirmConfig?.onOk?.()

      await waitFor(() => {
        expect(tldwClientMock.bulkDeleteSkills).toHaveBeenCalledWith([
          { name: "skill-4" }
        ])
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("keeps selected rows recoverable on stale bulk delete conflict", async () => {
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")
    let confirmConfig: { onOk?: () => void | Promise<void> } | undefined
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        confirmConfig = config as { onOk?: () => void | Promise<void> }
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    const conflict = Object.assign(new Error("409 version conflict"), { status: 409 })
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1), makeSkill(2)],
      count: 2,
      total: 2,
      limit: 10,
      offset: 0
    })
    tldwClientMock.bulkDeleteSkills.mockRejectedValueOnce(conflict)

    try {
      renderManager()
      await screen.findByText("skill-1")
      selectSkillRow("skill-1")
      selectSkillRow("skill-2")
      fireEvent.click(screen.getByRole("button", { name: "Delete selected" }))

      await waitFor(() => {
        expect(confirmConfig?.onOk).toBeTypeOf("function")
      })
      await expect(confirmConfig?.onOk?.()).resolves.toBeUndefined()

      await waitFor(() => {
        expect(notificationMock.error).toHaveBeenCalledWith(
          expect.objectContaining({
            message: "Selected skills changed elsewhere",
            description: "Reload skills before deleting these versions."
          })
        )
      })
      expect(invalidateSpy).toHaveBeenCalledWith(
        expect.objectContaining({ queryKey: ["skills"] })
      )
      expect(screen.getByTestId("skills-selection-actions")).toHaveTextContent("2 selected")
    } finally {
      confirmSpy.mockRestore()
      invalidateSpy.mockRestore()
    }
  })

  it("uses generic delete errors when a non-conflict message mentions 409", async () => {
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")
    let confirmConfig: { onOk?: () => void | Promise<void> } | undefined
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        confirmConfig = config as { onOk?: () => void | Promise<void> }
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    const failure = new Error("Failed after processing 409 skills")
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.deleteSkill.mockRejectedValueOnce(failure)

    try {
      renderManager()
      await screen.findByText("skill-1")
      fireEvent.click(screen.getByRole("button", { name: "Delete skill-1" }))

      await waitFor(() => {
        expect(confirmConfig?.onOk).toBeTypeOf("function")
      })
      await expect(confirmConfig?.onOk?.()).rejects.toThrow(
        "Failed after processing 409 skills"
      )

      await waitFor(() => {
        expect(notificationMock.error).toHaveBeenCalledWith(
          expect.objectContaining({
            message: "Failed to delete skill",
            description: expect.stringContaining("Failed after processing 409 skills")
          })
        )
      })
      expect(notificationMock.error).not.toHaveBeenCalledWith(
        expect.objectContaining({ message: "Skill changed elsewhere" })
      )
      expect(invalidateSpy).not.toHaveBeenCalled()
    } finally {
      confirmSpy.mockRestore()
      invalidateSpy.mockRestore()
    }
  })

  it("clears server-backed mode sorting when the mode column is hidden", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [
        {
          name: "mode-sorted",
          description: "Mode sorted skill",
          argument_hint: null,
          user_invocable: true,
          disable_model_invocation: false,
          context: "inline" as const
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("columnheader", { name: /Mode/ }))

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({
          sort: "context",
          order: "asc",
          limit: 10,
          offset: 0
        })
      )
    })

    fireEvent.click(await getColumnVisibilityOption("Mode"))

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.not.objectContaining({
          sort: "context",
          order: "asc"
        })
      )
    })
    expect(screen.queryByRole("columnheader", { name: /Mode/ })).not.toBeInTheDocument()
  })

  it("resets pagination when server-backed filters change", async () => {
    const pageOneSkills = Array.from({ length: 10 }, (_, index) => makeSkill(index + 1))
    const pageTwoSkills = [makeSkill(11), makeSkill(12)]
    const forkSkill = {
      name: "fork-only",
      description: "Fork skill",
      argument_hint: null,
      user_invocable: true,
      disable_model_invocation: false,
      context: "fork" as const
    }

    tldwClientMock.listSkills.mockImplementation(
      (params: { context?: string; limit: number; offset: number }) => {
        if (params.context === "fork") {
          return Promise.resolve({
            skills: [forkSkill],
            count: 1,
            total: 1,
            limit: params.limit,
            offset: 0
          })
        }

        return Promise.resolve({
          skills: params.offset === 10 ? pageTwoSkills : pageOneSkills,
          count: params.offset === 10 ? pageTwoSkills.length : pageOneSkills.length,
          total: 12,
          limit: params.limit,
          offset: params.offset
        })
      }
    )

    renderManager()

    expect(await screen.findByText("12 skills")).toBeInTheDocument()
    const secondPageItem = await screen.findByTitle("2")
    fireEvent.click(within(secondPageItem).getByText("2"))

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({ limit: 10, offset: 10 })
      )
    })

    fireEvent.click(screen.getByRole("button", { name: "Fork" }))

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({
          context: "fork",
          limit: 10,
          offset: 0
        })
      )
    })
    expect(await screen.findByText("fork-only")).toBeInTheDocument()
  })

  it("debounces model filter requests while typing", async () => {
    const firstPage = Array.from({ length: 10 }, (_, index) => makeSkill(index + 1))
    const modelResult = {
      name: "model-specific",
      description: "Model-specific skill",
      argument_hint: null,
      user_invocable: true,
      disable_model_invocation: false,
      context: "inline" as const
    }

    tldwClientMock.listSkills.mockImplementation(
      (params: { model?: string; limit: number; offset: number }) => {
        if (params.model === "gpt-4o") {
          return Promise.resolve({
            skills: [modelResult],
            count: 1,
            total: 1,
            limit: params.limit,
            offset: 0
          })
        }

        return Promise.resolve({
          skills: firstPage,
          count: firstPage.length,
          total: 12,
          limit: params.limit,
          offset: params.offset
        })
      }
    )

    renderManager()

    expect(await screen.findByText("12 skills")).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("Filter by model"), {
      target: { value: "gpt-4o" }
    })

    await new Promise((resolve) => setTimeout(resolve, 120))
    expect(tldwClientMock.listSkills).not.toHaveBeenCalledWith(
      expect.objectContaining({
        model: "gpt-4o",
        limit: 10,
        offset: 0
      })
    )

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({
          model: "gpt-4o",
          limit: 10,
          offset: 0
        })
      )
    })
    expect(await screen.findByText("model-specific")).toBeInTheDocument()
  })

  it("lets power users switch to compact table density", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    const table = screen.getByTestId("skills-table")
    expect(table).toHaveAttribute("data-density", "comfortable")

    fireEvent.click(screen.getByRole("button", { name: "Compact density" }))

    expect(table).toHaveAttribute("data-density", "compact")
    expect(window.localStorage.getItem("tldw:skills-manager:table-preferences:v1")).toContain(
      "\"density\":\"compact\""
    )
  })

  it("lets power users show and hide optional table columns", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [
        {
          name: "argument-skill",
          description: "Uses a topic",
          argument_hint: "topic",
          user_invocable: false,
          disable_model_invocation: true,
          context: "fork" as const
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByText("argument-skill")).toBeInTheDocument()
    expect(screen.queryByRole("columnheader", { name: "Argument hint" })).not.toBeInTheDocument()
    expect(screen.getByRole("columnheader", { name: /Mode/ })).toBeInTheDocument()

    fireEvent.click(await getColumnVisibilityOption("Argument hint"))

    expect(await screen.findByRole("columnheader", { name: "Argument hint" })).toBeInTheDocument()
    expect(screen.getByText("topic")).toBeInTheDocument()

    fireEvent.click(await getColumnVisibilityOption("Mode"))

    expect(screen.queryByRole("columnheader", { name: /Mode/ })).not.toBeInTheDocument()
    expect(window.localStorage.getItem("tldw:skills-manager:table-preferences:v1")).toContain(
      "argument_hint"
    )
  }, 10000)

  it("lets power users show runtime declarations in the table", async () => {
    const runtime = {
      execution_mode: "fork" as const,
      test_run_may_call_model: true,
      declares_tools: true,
      declared_tool_count: 2,
      model_override: "gpt-4o",
      auto_invocation_enabled: false
    }
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [
        {
          ...makeSkill(5),
          name: "runtime-skill",
          context: "fork" as const,
          allowed_tools: ["Read", "Bash(git *)"],
          model: "gpt-4o",
          runtime
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByText("runtime-skill")).toBeInTheDocument()
    expect(screen.queryByRole("columnheader", { name: "Runtime" })).not.toBeInTheDocument()

    fireEvent.click(await getColumnVisibilityOption("Runtime"))

    expect(await screen.findByRole("columnheader", { name: "Runtime" })).toBeInTheDocument()
    const row = screen.getByText("runtime-skill").closest("tr")
    expect(row).not.toBeNull()
    expect(within(row as HTMLTableRowElement).getByText("Fork")).toBeInTheDocument()
    expect(within(row as HTMLTableRowElement).getByText("Test may call model")).toBeInTheDocument()
    expect(within(row as HTMLTableRowElement).getByText("2 tools declared")).toBeInTheDocument()
    expect(within(row as HTMLTableRowElement).getByText("Model override")).toBeInTheDocument()
    expect(within(row as HTMLTableRowElement).getByText("Auto invocation off")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Test run runtime-skill" }))
    expect(skillPreviewMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        skillName: "runtime-skill",
        runtime
      })
    )
  }, 10000)

  it("falls back when runtime declarations are missing from legacy list responses", async () => {
    window.localStorage.setItem(
      "tldw:skills-manager:table-preferences:v1",
      JSON.stringify({
        density: "comfortable",
        visibleColumns: ["description", "context", "runtime"]
      })
    )
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [
        {
          name: "legacy-runtime",
          description: "Legacy response",
          argument_hint: null,
          user_invocable: true,
          disable_model_invocation: false,
          version: 1
        } as any
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByText("legacy-runtime")).toBeInTheDocument()
    expect(screen.getByRole("columnheader", { name: "Runtime" })).toBeInTheDocument()
    const row = screen.getByText("legacy-runtime").closest("tr")
    expect(row).not.toBeNull()
    expect(within(row as HTMLTableRowElement).getByText("Inline")).toBeInTheDocument()
    expect(
      within(row as HTMLTableRowElement).getByText("Prompt only by default")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Test run legacy-runtime" }))
    expect(skillPreviewMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        skillName: "legacy-runtime",
        runtime: {
          execution_mode: "inline",
          test_run_may_call_model: false,
          declares_tools: false,
          declared_tool_count: 0,
          model_override: null,
          auto_invocation_enabled: true
        }
      })
    )
  })

  it("preserves an explicit runtime execution mode before falling back to skill context", async () => {
    window.localStorage.setItem(
      "tldw:skills-manager:table-preferences:v1",
      JSON.stringify({
        density: "comfortable",
        visibleColumns: ["description", "context", "runtime"]
      })
    )
    const runtime = {
      execution_mode: "inline" as const,
      test_run_may_call_model: false,
      declares_tools: false,
      declared_tool_count: 0,
      model_override: null,
      auto_invocation_enabled: true
    }
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [
        {
          name: "explicit-inline-runtime",
          description: "Runtime contract wins",
          argument_hint: null,
          user_invocable: true,
          disable_model_invocation: false,
          allowed_tools: null,
          model: null,
          context: "fork" as const,
          runtime,
          version: 1
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByText("explicit-inline-runtime")).toBeInTheDocument()
    const row = screen.getByText("explicit-inline-runtime").closest("tr")
    expect(row).not.toBeNull()
    expect(within(row as HTMLTableRowElement).getByText("Inline")).toBeInTheDocument()
    expect(within(row as HTMLTableRowElement).queryByText("Fork")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Test run explicit-inline-runtime" }))
    expect(skillPreviewMock).toHaveBeenLastCalledWith(
      expect.objectContaining({
        skillName: "explicit-inline-runtime",
        runtime
      })
    )
  })

  it("restores persisted density and column visibility preferences", async () => {
    window.localStorage.setItem(
      "tldw:skills-manager:table-preferences:v1",
      JSON.stringify({
        density: "compact",
        visibleColumns: ["description", "argument_hint"]
      })
    )
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [
        {
          name: "restored-skill",
          description: "Restored description",
          argument_hint: "subject",
          user_invocable: true,
          disable_model_invocation: false,
          context: "inline" as const
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByText("restored-skill")).toBeInTheDocument()
    expect(screen.getByTestId("skills-table")).toHaveAttribute("data-density", "compact")
    expect(screen.getByRole("columnheader", { name: "Argument hint" })).toBeInTheDocument()
    expect(screen.queryByRole("columnheader", { name: /Mode/ })).not.toBeInTheDocument()
  })

  it("previews a text import before importing the skill", async () => {
    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))

    const dialog = await screen.findByRole("dialog", {
      name: "Import Skill from Text"
    })

    const contentInput = within(dialog).getByLabelText("SKILL.md Content")
    fireEvent.change(contentInput, {
      target: {
        value: "---\nname: imported-skill\ndescription: imported\n---\n\nBody"
      }
    })

    fireEvent.click(within(dialog).getByRole("button", { name: "Review import" }))

    await waitFor(() => {
      expect(tldwClientMock.previewSkillImport).toHaveBeenCalledWith({
        content: "---\nname: imported-skill\ndescription: imported\n---\n\nBody",
      })
    })
    expect(tldwClientMock.importSkill).not.toHaveBeenCalled()
    expect(await within(dialog).findByText("Import review")).toBeInTheDocument()
    expect(within(dialog).getByText("imported-skill")).toBeInTheDocument()

    fireEvent.click(within(dialog).getByRole("button", { name: "Import skill" }))

    await waitFor(() => {
      expect(tldwClientMock.importSkill).toHaveBeenCalledWith({
        content: "---\nname: imported-skill\ndescription: imported\n---\n\nBody",
        overwrite: false
      })
    })

    const successActions = await screen.findByTestId("skills-success-actions")
    expect(successActions).toHaveAttribute("data-ds-component", "Alert")
    expect(successActions).toHaveTextContent("Skill imported")
    expect(within(successActions).getByRole("button", { name: "Close" })).toBeInTheDocument()
    fireEvent.click(within(successActions).getByRole("button", { name: "View skill" }))

    await waitFor(() => {
      expect(tldwClientMock.getSkill).toHaveBeenCalledWith("imported-skill")
    })
  })

  it("falls back to the validated import name when the API returns an invalid name", async () => {
    tldwClientMock.previewSkillImport.mockResolvedValueOnce({
      valid: true,
      errors: [],
      name: "fallback-skill",
      description: "imported",
      argument_hint: null,
      disable_model_invocation: false,
      user_invocable: true,
      allowed_tools: null,
      model: null,
      context: "inline",
      supporting_file_count: 0,
      conflict: false,
      can_overwrite: false,
      existing_version: null
    })
    tldwClientMock.importSkill.mockResolvedValueOnce({ name: "Imported Skill" })

    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))

    const dialog = await screen.findByRole("dialog", {
      name: "Import Skill from Text"
    })

    fireEvent.change(within(dialog).getByLabelText("Name"), {
      target: { value: "fallback-skill" }
    })
    fireEvent.change(within(dialog).getByLabelText("SKILL.md Content"), {
      target: {
        value: "---\nname: fallback-skill\ndescription: imported\n---\n\nBody"
      }
    })

    fireEvent.click(within(dialog).getByRole("button", { name: "Review import" }))
    expect(await within(dialog).findByText("Import review")).toBeInTheDocument()
    fireEvent.click(within(dialog).getByRole("button", { name: "Import skill" }))

    const successActions = await screen.findByTestId("skills-success-actions")
    fireEvent.click(within(successActions).getByRole("button", { name: "View skill" }))

    await waitFor(() => {
      expect(tldwClientMock.getSkill).toHaveBeenCalledWith("fallback-skill")
    })
  })

  it("requires overwrite confirmation after a conflicting text import preview", async () => {
    tldwClientMock.previewSkillImport.mockResolvedValueOnce({
      valid: true,
      errors: [],
      name: "existing-skill",
      description: "replacement",
      argument_hint: null,
      disable_model_invocation: false,
      user_invocable: true,
      allowed_tools: null,
      model: null,
      context: "inline",
      supporting_file_count: 0,
      conflict: true,
      can_overwrite: true,
      existing_version: 3
    })

    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))

    const dialog = await screen.findByRole("dialog", {
      name: "Import Skill from Text"
    })
    expect(
      within(dialog).queryByRole("switch", { name: "Overwrite existing skill" })
    ).not.toBeInTheDocument()

    fireEvent.change(within(dialog).getByLabelText("SKILL.md Content"), {
      target: {
        value: "---\nname: existing-skill\ndescription: replacement\n---\n\nReplacement"
      }
    })

    fireEvent.click(within(dialog).getByRole("button", { name: "Review import" }))

    expect(await within(dialog).findByText("Existing skill detected")).toBeInTheDocument()
    expect(within(dialog).getByText("Version 3")).toBeInTheDocument()
    const importButton = within(dialog).getByRole("button", { name: "Import skill" })
    expect(importButton).toBeDisabled()

    fireEvent.click(within(dialog).getByRole("switch", { name: "Overwrite existing skill" }))
    await waitFor(() => {
      expect(
        within(dialog).getByRole("button", { name: "Import skill" })
      ).not.toBeDisabled()
    })
    fireEvent.click(within(dialog).getByRole("button", { name: "Import skill" }))

    await waitFor(() => {
      expect(tldwClientMock.importSkill).toHaveBeenCalledWith({
        content: "---\nname: existing-skill\ndescription: replacement\n---\n\nReplacement",
        overwrite: true
      })
    })
  })

  it("previews a file import before importing the uploaded file", async () => {
    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Import" }))

    const input = document.querySelector("input[type='file']") as HTMLInputElement | null
    expect(input).not.toBeNull()

    const file = new File(["# skill"], "my-skill.md", { type: "text/markdown" })
    fireEvent.change(input as HTMLInputElement, { target: { files: [file] } })

    await waitFor(() => {
      expect(tldwClientMock.previewSkillImportFile).toHaveBeenCalledWith(file)
    })
    expect(tldwClientMock.importSkillFile).not.toHaveBeenCalled()

    const dialog = await screen.findByRole("dialog", {
      name: "Review Skill Import"
    })
    expect(within(dialog).getByText("imported-file-skill")).toBeInTheDocument()

    fireEvent.click(within(dialog).getByRole("button", { name: "Import skill" }))

    await waitFor(() => {
      expect(tldwClientMock.importSkillFile).toHaveBeenCalledWith(file, { overwrite: false })
    })
  })

  it("seeds built-in skills via seedSkills action", async () => {
    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Seed Built-ins" }))
    fireEvent.click(await screen.findByText("Seed Missing Only"))

    await waitFor(() => {
      expect(tldwClientMock.seedSkills).toHaveBeenCalledWith({ overwrite: false })
    })

    const successActions = await screen.findByTestId("skills-success-actions")
    expect(successActions).toHaveAttribute("data-ds-component", "Alert")
    expect(successActions).toHaveTextContent("Built-in skills seeded")

    const testRunButton = within(successActions).getByRole("button", { name: "Test summarize" })
    const successActionRow = testRunButton.closest("div")
    expect(successActionRow).not.toBeNull()
    expect(successActionRow as HTMLElement).toHaveClass("mt-2")

    fireEvent.click(testRunButton)
    expect(screen.getByTestId("skill-preview-open")).toHaveTextContent("summarize")

    fireEvent.click(within(successActions).getByRole("button", { name: "Copy /skill summarize" }))
    await waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith("/skill summarize")
    })
  })

  it("opens a destructive confirmation before seeding built-in skills with overwrite", async () => {
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      () =>
        ({
          destroy: vi.fn(),
          update: vi.fn()
        }) as any
    )

    try {
      renderManager()

      await waitFor(() => {
        expect(tldwClientMock.listSkills).toHaveBeenCalled()
      })

      fireEvent.click(screen.getByRole("button", { name: "Seed Built-ins" }))
      fireEvent.click(await screen.findByText("Seed and Overwrite Existing"))

      expect(tldwClientMock.seedSkills).not.toHaveBeenCalled()
      expect(confirmSpy).toHaveBeenCalledTimes(1)
      expect(confirmSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Overwrite existing built-in skills?",
          content:
            "This replaces existing skill copies that match built-in skill names. Custom skills with other names are not changed.",
          okText: "Overwrite built-ins",
          cancelText: "Cancel",
          okButtonProps: expect.objectContaining({ danger: true })
        })
      )
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("does not seed built-in skills with overwrite unless confirmation is accepted", async () => {
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      () =>
        ({
          destroy: vi.fn(),
          update: vi.fn()
        }) as any
    )

    try {
      renderManager()

      await waitFor(() => {
        expect(tldwClientMock.listSkills).toHaveBeenCalled()
      })

      fireEvent.click(screen.getByRole("button", { name: "Seed Built-ins" }))
      fireEvent.click(await screen.findByText("Seed and Overwrite Existing"))

      expect(confirmSpy).toHaveBeenCalledTimes(1)
      expect(tldwClientMock.seedSkills).not.toHaveBeenCalled()
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("seeds built-in skills with overwrite after confirmation", async () => {
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      () =>
        ({
          destroy: vi.fn(),
          update: vi.fn()
        }) as any
    )

    try {
      renderManager()

      await waitFor(() => {
        expect(tldwClientMock.listSkills).toHaveBeenCalled()
      })

      fireEvent.click(screen.getByRole("button", { name: "Seed Built-ins" }))
      fireEvent.click(await screen.findByText("Seed and Overwrite Existing"))

      expect(confirmSpy).toHaveBeenCalledTimes(1)
      const [confirmConfig] = confirmSpy.mock.calls[0] as [
        { onOk?: () => void | Promise<void> }
      ]
      await confirmConfig.onOk?.()

      await waitFor(() => {
        expect(tldwClientMock.seedSkills).toHaveBeenCalledTimes(1)
        expect(tldwClientMock.seedSkills).toHaveBeenCalledWith({ overwrite: true })
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("sanitizes skill action failure notifications", async () => {
    tldwClientMock.seedSkills.mockRejectedValueOnce(
      new Error(
        "Request failed: POST /api/v1/skills/seed?token=sk_live_secret from /Users/alice/.tldw with Bearer token_secret_123"
      )
    )

    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Seed Built-ins" }))
    fireEvent.click(await screen.findByText("Seed Missing Only"))

    await waitFor(() => {
      expect(notificationMock.error).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Failed to seed built-in skills",
          description: expect.stringContaining("[server-endpoint]")
        })
      )
    })

    const payload = notificationMock.error.mock.calls.at(-1)?.[0] as {
      description?: string
    }
    expect(payload.description).toContain("[redacted-path]")
    expect(payload.description).toContain("Bearer [redacted-secret]")
    expect(payload.description).not.toContain("/api/v1/skills")
    expect(payload.description).not.toContain("sk_live_secret")
    expect(payload.description).not.toContain("/Users/alice")
    expect(payload.description).not.toContain("token_secret_123")
  })

  it("offers test-run and copy-invocation actions after creating a skill", async () => {
    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "New Skill" }))
    fireEvent.click(await screen.findByRole("button", { name: "Complete create" }))

    const successActions = await screen.findByTestId("skills-success-actions")
    expect(successActions).toHaveTextContent("Skill created")

    fireEvent.click(within(successActions).getByRole("button", { name: "Test run" }))
    expect(screen.getByTestId("skill-preview-open")).toHaveTextContent("created-skill")

    fireEvent.click(
      within(successActions).getByRole("button", {
        name: "Copy /skill created-skill"
      })
    )

    await waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith("/skill created-skill")
    })
  })

  it("returns focus to New Skill after cancelling the create drawer", async () => {
    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    const newSkillButton = screen.getByRole("button", { name: "New Skill" })
    newSkillButton.focus()
    fireEvent.click(newSkillButton)
    expect(await screen.findByTestId("skill-drawer-open")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Cancel drawer" }))

    await waitFor(() => {
      expect(newSkillButton).toHaveFocus()
    })
  })
})
