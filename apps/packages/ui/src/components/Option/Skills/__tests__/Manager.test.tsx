import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ConfigProvider, message, Modal } from "antd"
import { MemoryRouter, useLocation, useNavigate } from "react-router-dom"
import type { ReactElement, ReactNode } from "react"
import type { SkillImportPreviewResponse } from "@/types/skill"
import { SkillsManager } from "../Manager"

const tldwClientMock = vi.hoisted(() => ({
  getConfig: vi.fn(),
  listSkills: vi.fn(),
  getSkill: vi.fn(),
  deleteSkill: vi.fn(),
  bulkDeleteSkills: vi.fn(),
  listSkillTrash: vi.fn(),
  restoreSkill: vi.fn(),
  purgeSkill: vi.fn(),
  exportSkill: vi.fn(),
  previewSkillImport: vi.fn(),
  previewSkillImportFile: vi.fn(),
  importSkill: vi.fn(),
  importSkillFile: vi.fn(),
  seedSkills: vi.fn()
}))

const tldwAuthMock = vi.hoisted(() => ({
  getCurrentUser: vi.fn()
}))

const notificationMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn()
}))

const skillDrawerMock = vi.hoisted(() => vi.fn())
const skillPreviewMock = vi.hoisted(() => vi.fn())
const skillDetailsMock = vi.hoisted(() => vi.fn())
const setSelectedQuickPromptMock = vi.hoisted(() => vi.fn())
const closeLifecycleMockState = vi.hoisted(() => ({
  drawerWasOpen: false,
  previewWasOpen: false
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: tldwAuthMock
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({ setSelectedQuickPrompt: setSelectedQuickPromptMock })
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
    onAfterClose?: () => void
    onSaved: (skillName?: string) => void
  }) => {
    skillDrawerMock(props)
    if (closeLifecycleMockState.drawerWasOpen && !props.open) {
      window.setTimeout(() => props.onAfterClose?.(), 0)
    }
    closeLifecycleMockState.drawerWasOpen = props.open

    return props.open ? (
      <div data-testid="skill-drawer-open">
        Skill drawer open
        <button
          type="button"
          onClick={() => {
            props.onClose()
          }}
        >
          Cancel drawer
        </button>
        <button
          type="button"
          onClick={() => {
            props.onSaved("created-skill")
          }}
        >
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
    onAfterClose?: () => void
  }) => {
    skillPreviewMock(props)
    const isOpen = Boolean(props.skillName)
    if (closeLifecycleMockState.previewWasOpen && !isOpen) {
      window.setTimeout(() => props.onAfterClose?.(), 0)
    }
    closeLifecycleMockState.previewWasOpen = isOpen

    return props.skillName ? (
      <div data-testid="skill-preview-open">
        Test run: {props.skillName}
        <button
          type="button"
          onClick={() => {
            props.onClose()
          }}
        >
          Close test run
        </button>
      </div>
    ) : null
  }
}))

vi.mock("../SkillDetailsDrawer", () => ({
  SkillDetailsDrawer: (props: {
    skillName: string | null
    onClose: () => void
    onTest: (skillName: string) => void
  }) => {
    skillDetailsMock(props)
    return props.skillName ? (
      <div data-testid="skill-details-open">
        Skill details: {props.skillName}
        <button type="button" onClick={() => props.onTest(props.skillName!)}>
          Test details {props.skillName}
        </button>
        <button type="button" onClick={props.onClose}>Close details</button>
      </div>
    ) : null
  }
}))

const LocationProbe = () => {
  const location = useLocation()
  return <span data-testid="location-probe">{`${location.pathname}${location.search}`}</span>
}

const HistoryBackButton = () => {
  const navigate = useNavigate()
  return <button type="button" onClick={() => navigate(-1)}>Go back</button>
}

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
    closeLifecycleMockState.drawerWasOpen = false
    closeLifecycleMockState.previewWasOpen = false
    window.localStorage.clear()
    window.sessionStorage.clear()
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
    tldwClientMock.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    })
    tldwAuthMock.getCurrentUser.mockResolvedValue({
      id: 1,
      username: "skills-user",
      is_active: true
    })
    tldwClientMock.listSkillTrash.mockResolvedValue({
      skills: [],
      count: 0,
      total: 0,
      limit: 10,
      offset: 0
    })
    tldwClientMock.restoreSkill.mockResolvedValue({ name: "restored-skill" })
    tldwClientMock.purgeSkill.mockResolvedValue(undefined)
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
    vi.restoreAllMocks()
    if (originalClipboard) {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: originalClipboard
      })
    } else {
      Reflect.deleteProperty(navigator, "clipboard")
    }
  })

  const renderManager = (initialEntry = "/skills") =>
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={[initialEntry]}>
            <SkillsManager />
            <LocationProbe />
            <HistoryBackButton />
          </MemoryRouter>
        </QueryClientProvider>
      </ConfigProvider>
    )

  const openFilters = async () => {
    if (!screen.queryByRole("combobox", { name: "Skill mode filter" })) {
      fireEvent.click(screen.getByRole("button", { name: /^Filters/ }))
    }
    return screen.findByRole("combobox", { name: "Skill mode filter" })
  }

  const chooseFilter = async (label: string, option: string) => {
    await openFilters()
    const combobox = screen.getByRole("combobox", { name: label })
    fireEvent.mouseDown(combobox)
    fireEvent.click(await screen.findByText(option))
  }

  const openViewOptions = async () => {
    if (!screen.queryByRole("checkbox", { name: "Description" })) {
      fireEvent.click(screen.getByRole("button", { name: "View options" }))
    }
    return screen.findByRole("checkbox", { name: "Description" })
  }

  const getColumnVisibilityOption = async (name: string) => {
    await openViewOptions()
    return screen.getByRole("checkbox", { name })
  }

  const chooseRowMoreAction = async (skillName: string, action: string) => {
    fireEvent.click(screen.getByRole("button", { name: `More actions for ${skillName}` }))
    fireEvent.click(await screen.findByRole("menuitem", { name: action }))
  }

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
    expect(screen.getByText("Search skills")).toBeInTheDocument()
    expect(screen.getByRole("searchbox", { name: "Search skills" })).toBeInTheDocument()
  })

  it("keeps the Skills list loading live region mounted after loading finishes", async () => {
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

    const loadingStatusRegion = await screen.findByRole("status")
    expect(loadingStatusRegion).toHaveTextContent("Loading skills")
    await waitFor(() => expect(tldwClientMock.listSkills).toHaveBeenCalledTimes(1))

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
    expect(screen.getByRole("status")).toHaveTextContent("")
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
      expect(tldwClientMock.seedSkills).toHaveBeenCalledWith(
        { overwrite: false },
        expect.objectContaining({ signal: expect.anything() })
      )
    })

    const createFromTemplateButton = within(emptyState).getByRole("button", {
      name: "Create from template"
    })
    expect(createFromTemplateButton).toHaveAttribute("data-skill-action", "new")

    fireEvent.click(createFromTemplateButton)
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
    await chooseFilter("Skill mode filter", "Fork")
    await chooseFilter("Skill visibility filter", "Hidden")
    await chooseFilter("Skill tools filter", "Has tools")

    expect(screen.getByText("Mode: Fork")).toBeInTheDocument()
    expect(screen.getByText("Visibility: Hidden")).toBeInTheDocument()
    expect(screen.getByText("Tools: Has tools")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Remove Mode: Fork filter" })
    ).toBeInTheDocument()

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
  }, 10_000)

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
    await chooseFilter("Skill mode filter", "Fork")

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

  it("keeps the primary skill actions visible and moves destructive actions into More", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    expect(await screen.findByRole("button", { name: "View skill-1" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Use skill-1 in chat" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Copy invocation for skill-1" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Test run skill-1" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Delete skill-1" })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "More actions for skill-1" }))
    const menu = await screen.findByRole("menu")
    expect(within(menu).getByRole("menuitem", { name: "Edit" })).toBeInTheDocument()
    expect(within(menu).getByRole("menuitem", { name: "Duplicate" })).toBeInTheDocument()
    expect(within(menu).getByRole("menuitem", { name: "Export" })).toBeInTheDocument()
    expect(within(menu).getByRole("menuitem", { name: "Delete" })).toBeInTheDocument()
  })

  it("prefills the chat composer when using a skill", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    fireEvent.click(await screen.findByRole("button", { name: "Use skill-1 in chat" }))

    expect(setSelectedQuickPromptMock).toHaveBeenCalledWith("/skill skill-1")
    expect(screen.getByTestId("location-probe")).toHaveTextContent("/chat")
  })

  it("opens a read-only details workflow before editing", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    fireEvent.click(await screen.findByRole("button", { name: "View skill-1" }))

    expect(screen.getByTestId("skill-details-open")).toHaveTextContent("skill-1")
  })

  it("returns focus to the row view action after testing from details", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    const viewButton = await screen.findByRole("button", { name: "View skill-1" })
    viewButton.focus()
    fireEvent.click(viewButton)
    const detailsTestButton = screen.getByRole("button", { name: "Test details skill-1" })
    detailsTestButton.focus()
    fireEvent.click(detailsTestButton)

    expect(screen.getByTestId("skill-preview-open")).toHaveTextContent("skill-1")
    fireEvent.click(screen.getByRole("button", { name: "Close test run" }))

    await waitFor(() => expect(viewButton).toHaveFocus())
  })

  it("starts a duplicate from the selected skill without entering edit mode", async () => {
    const source = {
      ...makeSkill(1),
      id: "skill-1",
      allowed_tools: null,
      model: null,
      content: "Body",
      raw_content: null,
      supporting_files: null,
      directory_path: "/tmp/skill-1",
      created_at: "2026-07-14T00:00:00Z",
      last_modified: "2026-07-14T00:00:00Z"
    }
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)], count: 1, total: 1, limit: 10, offset: 0
    })
    tldwClientMock.getSkill.mockResolvedValue(source)
    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    fireEvent.click(await screen.findByRole("button", { name: "More actions for skill-1" }))
    fireEvent.click(await screen.findByRole("menuitem", { name: "Duplicate" }))

    await waitFor(() => expect(tldwClientMock.getSkill).toHaveBeenCalledWith(
      "skill-1",
      expect.objectContaining({ signal: expect.anything() })
    ))
    await waitFor(() => {
      expect(skillDrawerMock).toHaveBeenLastCalledWith(
        expect.objectContaining({ open: true, skill: null, duplicateFrom: source })
      )
    })
  })

  it("restores a shareable view from the URL and writes filter changes back", async () => {
    renderManager(
      "/skills?q=research&mode=fork&visibility=hidden&tools=with-tools&model=gpt&sort=name&order=desc&page=2&pageSize=20"
    )

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalledWith(
        expect.objectContaining({
          q: "research",
          context: "fork",
          includeHidden: true,
          userInvocable: false,
          hasTools: true,
          model: "gpt",
          sort: "name",
          order: "desc",
          limit: 20,
          offset: 20
        })
      )
    })
    await waitFor(() => {
      expect(screen.getByTestId("location-probe")).toHaveTextContent("mode=fork")
      expect(screen.getByTestId("location-probe")).not.toHaveTextContent("page=2")
    })

    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }))
    await waitFor(() => {
      expect(screen.getByTestId("location-probe")).toHaveTextContent("/skills?q=research")
      expect(screen.getByTestId("location-probe")).toHaveTextContent("sort=name&order=desc")
      expect(screen.getByTestId("location-probe")).not.toHaveTextContent("mode=fork")
    })

    fireEvent.click(screen.getByRole("button", { name: "Go back" }))
    await waitFor(() => {
      expect(screen.getByTestId("location-probe")).toHaveTextContent("mode=fork")
      expect(screen.getByTestId("location-probe")).toHaveTextContent("visibility=hidden")
      expect(screen.getByTestId("location-probe")).toHaveTextContent("tools=with-tools")
    })
    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({
          context: "fork",
          userInvocable: false,
          hasTools: true
        })
      )
    })
  })

  it("supports keyboard shortcuts without hijacking editable fields", async () => {
    renderManager()
    await waitFor(() => expect(tldwClientMock.listSkills).toHaveBeenCalled())
    const search = screen.getByRole("searchbox", { name: "Search skills" })

    fireEvent.keyDown(document, { key: "/" })
    expect(search).toHaveFocus()

    search.blur()
    fireEvent.keyDown(document, { key: "n" })
    expect(await screen.findByTestId("skill-drawer-open")).toBeInTheDocument()
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

  it("returns focus to the stable row action after editing from the More menu", async () => {
    const skill = makeSkill(1)
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [skill],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.getSkill.mockResolvedValue(skill)

    renderManager()

    expect(await screen.findByText("1 skill")).toBeInTheDocument()
    const moreButton = screen.getByRole("button", { name: "More actions for skill-1" })
    fireEvent.click(moreButton)
    fireEvent.click(await screen.findByRole("menuitem", { name: "Edit" }))
    expect(await screen.findByTestId("skill-drawer-open")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Cancel drawer" }))

    await waitFor(() => expect(moreButton).toHaveFocus())
  })

  it("does not restore focus to an identically labelled button outside Skills when the trigger is gone", async () => {
    const outsideButton = document.createElement("button")
    outsideButton.type = "button"
    outsideButton.setAttribute("aria-label", "Test run skill-1")
    outsideButton.textContent = "Outside duplicate"
    document.body.prepend(outsideButton)

    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    try {
      renderManager()

      expect(await screen.findByText("1 skill")).toBeInTheDocument()
      const testRunButton = screen
        .getAllByRole("button", { name: "Test run skill-1" })
        .find((button) => button !== outsideButton)
      expect(testRunButton).toBeDefined()
      fireEvent.click(testRunButton as HTMLElement)
      expect(screen.getByTestId("skill-preview-open")).toHaveTextContent("skill-1")

      vi.useFakeTimers()
      ;(testRunButton as HTMLElement).remove()
      fireEvent.click(screen.getByRole("button", { name: "Close test run" }))
      await vi.runAllTimersAsync()

      expect(outsideButton).not.toHaveFocus()
    } finally {
      outsideButton.remove()
      vi.useRealTimers()
    }
  })

  it("does not steal focus back when the user moves focus before restore runs", async () => {
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    try {
      renderManager()

      expect(await screen.findByText("1 skill")).toBeInTheDocument()
      const testRunButton = screen.getByRole("button", { name: "Test run skill-1" })
      fireEvent.click(testRunButton)
      expect(screen.getByTestId("skill-preview-open")).toHaveTextContent("skill-1")

      vi.useFakeTimers()
      fireEvent.click(screen.getByRole("button", { name: "Close test run" }))
      const newSkillButton = screen.getByRole("button", { name: "New Skill" })
      newSkillButton.focus()
      await vi.runAllTimersAsync()

      expect(newSkillButton).toHaveFocus()
    } finally {
      vi.useRealTimers()
    }
  })

  it("passes the row version when deleting a skill", async () => {
    let confirmTitle: ReactNode
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        confirmTitle = config.title
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
      await chooseRowMoreAction("skill-2", "Delete")

      await waitFor(() => {
        expect(tldwClientMock.deleteSkill).toHaveBeenCalledWith(
          "skill-2",
          3,
          expect.objectContaining({ signal: expect.anything() })
        )
      })
      expect(confirmTitle).toBe("Delete skill-2?")
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("aborts an in-flight delete and ignores its result after a scope change", async () => {
    let confirmConfig: { onOk?: () => void | Promise<void> } | undefined
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce((config) => {
      confirmConfig = config as { onOk?: () => void | Promise<void> }
      return { destroy: vi.fn(), update: vi.fn() } as any
    })
    let resolveDelete: (() => void) | undefined
    let deleteSignal: AbortSignal | undefined
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [makeSkill(2)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.deleteSkill.mockImplementationOnce(
      (_name: string, _version: number, options?: { signal?: AbortSignal }) => {
        deleteSignal = options?.signal
        return new Promise<void>((resolve) => {
          resolveDelete = resolve
        })
      }
    )

    try {
      renderManager()
      await screen.findByText("skill-2")
      await chooseRowMoreAction("skill-2", "Delete")
      const pendingDelete = confirmConfig?.onOk?.()
      await waitFor(() => expect(tldwClientMock.deleteSkill).toHaveBeenCalledTimes(1))

      act(() => window.dispatchEvent(new Event("tldw:config-updated")))

      expect(deleteSignal?.aborted).toBe(true)
      resolveDelete?.()
      await pendingDelete
      expect(notificationMock.success).not.toHaveBeenCalledWith(
        expect.objectContaining({ message: "Skill moved to Trash" })
      )
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
      await chooseRowMoreAction("skill-4", "Delete")

      await waitFor(() => {
        expect(tldwClientMock.deleteSkill).toHaveBeenCalledWith(
          "skill-4",
          undefined,
          expect.objectContaining({ signal: expect.anything() })
        )
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
      await chooseRowMoreAction("skill-1", "Delete")

      await waitFor(() => {
        expect(confirmConfig?.onOk).toBeTypeOf("function")
      })
      await confirmConfig?.onOk?.()

      await waitFor(() => {
        expect(notificationMock.error).toHaveBeenCalledWith(
          expect.objectContaining({
            message: "Skill changed elsewhere",
            description: "Reload skills before deleting this version.",
            btn: expect.anything()
          })
        )
      })
      expect(invalidateSpy).toHaveBeenCalledWith(
        expect.objectContaining({ queryKey: ["skills"] })
      )
      const conflictNotice = notificationMock.error.mock.calls.at(-1)?.[0] as {
        btn?: ReactElement
      }
      const callsBeforeReload = tldwClientMock.listSkills.mock.calls.length
      render(conflictNotice.btn as ReactElement)
      fireEvent.click(screen.getByRole("button", { name: "Reload skills" }))
      await waitFor(() => {
        expect(tldwClientMock.listSkills.mock.calls.length).toBeGreaterThan(callsBeforeReload)
      })
    } finally {
      confirmSpy.mockRestore()
      invalidateSpy.mockRestore()
    }
  })

  it("offers immediate durable undo after moving a skill to Trash", async () => {
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
      await chooseRowMoreAction("skill-2", "Delete")

      await waitFor(() => expect(tldwClientMock.deleteSkill).toHaveBeenCalledWith(
        "skill-2",
        3,
        expect.objectContaining({ signal: expect.anything() })
      ))
      const deleteNotice = notificationMock.success.mock.calls.at(-1)?.[0] as {
        message?: string
        btn?: ReactElement
      }
      expect(deleteNotice.message).toBe("Skill moved to Trash")
      expect(deleteNotice.btn).toBeDefined()

      render(deleteNotice.btn as ReactElement)
      fireEvent.click(screen.getByRole("button", { name: "Undo delete skill-2" }))

      await waitFor(() => {
        expect(tldwClientMock.restoreSkill).toHaveBeenCalledWith(
          "skill-2",
          4,
          expect.objectContaining({ signal: expect.anything() })
        )
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("lists Trash items with versioned restore actions", async () => {
    tldwClientMock.listSkillTrash.mockResolvedValueOnce({
      skills: [
        {
          ...makeSkill(1),
          name: "trashed-skill",
          deleted_at: "2026-07-14T12:00:00Z",
          restorable: true,
          restore_unavailable_reason: null,
          version: 4
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager("/skills?view=trash")

    expect(await screen.findByText("trashed-skill")).toBeInTheDocument()
    expect(tldwClientMock.listSkillTrash).toHaveBeenCalledWith(
      expect.objectContaining({ limit: 10, offset: 0 })
    )
    fireEvent.click(screen.getByRole("button", { name: "Restore trashed-skill" }))
    await waitFor(() => {
      expect(tldwClientMock.restoreSkill).toHaveBeenCalledWith(
        "trashed-skill",
        4,
        expect.objectContaining({ signal: expect.anything() })
      )
    })
  })

  it("permanently deletes Trash items only after confirmation", async () => {
    tldwClientMock.listSkillTrash.mockResolvedValueOnce({
      skills: [
        {
          ...makeSkill(1),
          name: "purge-skill",
          deleted_at: "2026-07-14T12:00:00Z",
          restorable: true,
          restore_unavailable_reason: null,
          version: 5
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    let confirmTitle: ReactNode
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        confirmTitle = config.title
        void config.onOk?.()
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    try {
      renderManager("/skills?view=trash")

      expect(await screen.findByText("purge-skill")).toBeInTheDocument()
      fireEvent.click(screen.getByRole("button", { name: "Permanently delete purge-skill" }))
      await waitFor(() => {
        expect(tldwClientMock.purgeSkill).toHaveBeenCalledWith(
          "purge-skill",
          5,
          expect.objectContaining({ signal: expect.anything() })
        )
      })
      expect(confirmTitle).toBe("Permanently delete purge-skill?")
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("explains and disables restore when archived files are missing", async () => {
    tldwClientMock.listSkillTrash.mockResolvedValueOnce({
      skills: [
        {
          ...makeSkill(1),
          name: "broken-archive",
          deleted_at: "2026-07-14T12:00:00Z",
          restorable: false,
          restore_unavailable_reason: "Archived skill files are missing.",
          version: 2
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager("/skills?view=trash")

    expect(await screen.findByText("Archived skill files are missing.")).toBeInTheDocument()
    const restoreButton = screen.getByRole("button", { name: "Restore broken-archive" })
    expect(restoreButton).toBeDisabled()
    const reasonId = restoreButton.getAttribute("aria-describedby")
    expect(reasonId).toBeTruthy()
    expect(document.getElementById(reasonId!)).toHaveTextContent(
      "Archived skill files are missing."
    )
  })

  it("disables competing Trash actions while a restore is pending", async () => {
    let finishRestore: ((value: { name: string }) => void) | undefined
    tldwClientMock.restoreSkill.mockImplementationOnce(
      () => new Promise((resolve) => {
        finishRestore = resolve
      })
    )
    tldwClientMock.listSkillTrash.mockResolvedValueOnce({
      skills: [
        {
          ...makeSkill(1),
          name: "pending-restore",
          deleted_at: "2026-07-14T12:00:00Z",
          restorable: true,
          restore_unavailable_reason: null,
          version: 2
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager("/skills?view=trash")

    fireEvent.click(await screen.findByRole("button", { name: "Restore pending-restore" }))
    await waitFor(() => expect(tldwClientMock.restoreSkill).toHaveBeenCalled())
    expect(
      screen.getByRole("button", { name: "Permanently delete pending-restore" })
    ).toBeDisabled()

    await act(async () => finishRestore?.({ name: "pending-restore" }))
  })

  it("offers Reload Trash after a stale restore conflict", async () => {
    const conflict = Object.assign(new Error("409 version conflict"), { status: 409 })
    tldwClientMock.listSkillTrash.mockResolvedValue({
      skills: [
        {
          ...makeSkill(1),
          name: "restore-conflict",
          deleted_at: "2026-07-14T12:00:00Z",
          restorable: true,
          restore_unavailable_reason: null,
          version: 4
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.restoreSkill.mockRejectedValueOnce(conflict)

    renderManager("/skills?view=trash")
    expect(await screen.findByText("restore-conflict")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Restore restore-conflict" }))

    await waitFor(() => {
      expect(notificationMock.error).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Trash item changed elsewhere",
          description: "Reload Trash before restoring this version.",
          btn: expect.anything()
        })
      )
    })
    const conflictNotice = notificationMock.error.mock.calls.at(-1)?.[0] as {
      btn?: ReactElement
    }
    const callsBeforeReload = tldwClientMock.listSkillTrash.mock.calls.length
    render(conflictNotice.btn as ReactElement)
    fireEvent.click(screen.getByRole("button", { name: "Reload Trash" }))
    await waitFor(() => {
      expect(tldwClientMock.listSkillTrash.mock.calls.length).toBeGreaterThan(callsBeforeReload)
    })
  })

  it("offers Reload Trash after a stale permanent-delete conflict", async () => {
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        void config.onOk?.()
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    const conflict = Object.assign(new Error("409 version conflict"), { status: 409 })
    tldwClientMock.listSkillTrash.mockResolvedValue({
      skills: [
        {
          ...makeSkill(1),
          name: "purge-conflict",
          deleted_at: "2026-07-14T12:00:00Z",
          restorable: true,
          restore_unavailable_reason: null,
          version: 5
        }
      ],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.purgeSkill.mockRejectedValueOnce(conflict)

    try {
      renderManager("/skills?view=trash")
      expect(await screen.findByText("purge-conflict")).toBeInTheDocument()
      fireEvent.click(screen.getByRole("button", { name: "Permanently delete purge-conflict" }))

      await waitFor(() => {
        expect(notificationMock.error).toHaveBeenCalledWith(
          expect.objectContaining({
            message: "Trash item changed elsewhere",
            description: "Reload Trash before permanently deleting this version.",
            btn: expect.anything()
          })
        )
      })
    } finally {
      confirmSpy.mockRestore()
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
      await chooseRowMoreAction("skill-1", "Export")

      await waitFor(() => {
        expect(tldwClientMock.exportSkill).toHaveBeenCalledWith(
          "skill-1",
          expect.objectContaining({ signal: expect.anything() })
        )
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
    await chooseRowMoreAction("skill-1", "Export")

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

  it("keeps selections across pages and downloads one archive for bulk export", async () => {
    const originalCreateObjectURL = URL.createObjectURL
    const originalRevokeObjectURL = URL.revokeObjectURL
    let downloadCount = 0
    let downloadedFilename = ""
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        downloadCount += 1
        downloadedFilename = this.download
      })
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:skills-bulk-export")
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn()
    })
    const firstPage = Array.from({ length: 10 }, (_, index) => makeSkill(index + 1))
    const secondPage = [makeSkill(11)]
    tldwClientMock.listSkills.mockImplementation(
      (params: { limit: number; offset: number }) => Promise.resolve({
        skills: params.offset === 10 ? secondPage : firstPage,
        count: params.offset === 10 ? secondPage.length : firstPage.length,
        total: 11,
        limit: params.limit,
        offset: params.offset
      })
    )
    tldwClientMock.exportSkill.mockImplementation((name: string) => Promise.resolve({
      blob: new Blob([name], { type: "application/zip" }),
      filename: `${name}.zip`
    }))

    try {
      renderManager()
      await screen.findByText("skill-1")
      selectSkillRow("skill-1")

      const secondPageItem = await screen.findByTitle("2")
      fireEvent.click(within(secondPageItem).getByText("2"))
      expect(await screen.findByText("skill-11")).toBeInTheDocument()
      expect(screen.getByTestId("skills-selection-actions")).toHaveTextContent("1 selected")

      selectSkillRow("skill-11")
      expect(screen.getByTestId("skills-selection-actions")).toHaveTextContent("2 selected")
      fireEvent.click(screen.getByRole("button", { name: "Export selected" }))

      await waitFor(() => {
        expect(tldwClientMock.exportSkill).toHaveBeenCalledTimes(2)
      })
      await waitFor(() => expect(downloadCount).toBe(1))
      expect(tldwClientMock.exportSkill).toHaveBeenCalledWith(
        "skill-1",
        expect.objectContaining({ signal: expect.anything() })
      )
      expect(tldwClientMock.exportSkill).toHaveBeenCalledWith(
        "skill-11",
        expect.objectContaining({ signal: expect.anything() })
      )
      expect(downloadedFilename).toMatch(/^skills-export-\d{4}-\d{2}-\d{2}\.zip$/)
      expect(notificationMock.success).toHaveBeenCalledWith(
        expect.objectContaining({ message: "Skills exported" })
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
  }, 10000)

  it("limits concurrent requests during bulk export", async () => {
    const originalCreateObjectURL = URL.createObjectURL
    const originalRevokeObjectURL = URL.revokeObjectURL
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:skills-bounded-export")
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn()
    })
    const skills = Array.from({ length: 6 }, (_, index) => makeSkill(index + 1))
    let inFlight = 0
    let maxInFlight = 0
    tldwClientMock.listSkills.mockResolvedValue({
      skills,
      count: skills.length,
      total: skills.length,
      limit: 10,
      offset: 0
    })
    tldwClientMock.exportSkill.mockImplementation(async (name: string) => {
      inFlight += 1
      maxInFlight = Math.max(maxInFlight, inFlight)
      await new Promise((resolve) => window.setTimeout(resolve, 10))
      inFlight -= 1
      return {
        blob: new Blob([name], { type: "application/zip" }),
        filename: `${name}.zip`
      }
    })

    try {
      renderManager()
      await screen.findByText("skill-1")
      for (const skill of skills) selectSkillRow(skill.name)

      fireEvent.click(screen.getByRole("button", { name: "Export selected" }))

      await waitFor(() => {
        expect(notificationMock.success).toHaveBeenCalledWith(
          expect.objectContaining({ message: "Skills exported" })
        )
      })
      expect(maxInFlight).toBeLessThanOrEqual(4)
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
  }, 10000)

  it("cancels an in-flight bulk export when the server scope changes", async () => {
    const originalCreateObjectURL = URL.createObjectURL
    const originalRevokeObjectURL = URL.revokeObjectURL
    const createObjectURL = vi.fn(() => "blob:stale-skills-export")
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: createObjectURL
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn()
    })
    let resolveExport: ((result: { blob: Blob; filename: string }) => void) | undefined
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.exportSkill.mockImplementationOnce(
      () => new Promise((resolve) => {
        resolveExport = resolve
      })
    )

    try {
      renderManager()
      await screen.findByText("skill-1")
      selectSkillRow("skill-1")
      fireEvent.click(screen.getByRole("button", { name: "Export selected" }))
      await waitFor(() => expect(tldwClientMock.exportSkill).toHaveBeenCalledTimes(1))

      act(() => window.dispatchEvent(new Event("tldw:config-updated")))
      await act(async () => {
        resolveExport?.({
          blob: new Blob(["skill-1"], { type: "application/zip" }),
          filename: "skill-1.zip"
        })
        await new Promise((resolve) => window.setTimeout(resolve, 25))
      })

      expect(createObjectURL).not.toHaveBeenCalled()
      expect(notificationMock.success).not.toHaveBeenCalledWith(
        expect.objectContaining({ message: "Skills exported" })
      )
      expect(notificationMock.error).not.toHaveBeenCalledWith(
        expect.objectContaining({ message: "Failed to export selected skills" })
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
        expect(tldwClientMock.bulkDeleteSkills).toHaveBeenCalledWith(
          [
            { name: "skill-1", version: 2 },
            { name: "skill-2", version: 3 }
          ],
          expect.objectContaining({ signal: expect.anything() })
        )
      })
      expect(notificationMock.success).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Skills moved to Trash",
          description: "2 skill(s) can be restored from Trash."
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
        expect(tldwClientMock.bulkDeleteSkills).toHaveBeenCalledWith(
          [{ name: "skill-4" }],
          expect.objectContaining({ signal: expect.anything() })
        )
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("clears stale cross-page selections after a bulk delete conflict", async () => {
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")
    let confirmConfig: { onOk?: () => void | Promise<void> } | undefined
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        confirmConfig = config as { onOk?: () => void | Promise<void> }
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    const conflict = Object.assign(new Error("409 version conflict"), { status: 409 })
    const firstPage = Array.from({ length: 10 }, (_, index) => makeSkill(index + 1))
    const secondPage = [makeSkill(11)]
    tldwClientMock.listSkills.mockImplementation(
      (params: { limit: number; offset: number }) => Promise.resolve({
        skills: params.offset === 10 ? secondPage : firstPage,
        count: params.offset === 10 ? secondPage.length : firstPage.length,
        total: 11,
        limit: params.limit,
        offset: params.offset
      })
    )
    tldwClientMock.bulkDeleteSkills.mockRejectedValueOnce(conflict)

    try {
      renderManager()
      await screen.findByText("skill-1")
      selectSkillRow("skill-1")
      const secondPageItem = await screen.findByTitle("2")
      fireEvent.click(within(secondPageItem).getByText("2"))
      expect(await screen.findByText("skill-11")).toBeInTheDocument()
      selectSkillRow("skill-11")
      fireEvent.click(screen.getByRole("button", { name: "Delete selected" }))

      await waitFor(() => {
        expect(confirmConfig?.onOk).toBeTypeOf("function")
      })
      await expect(confirmConfig?.onOk?.()).resolves.toBeUndefined()

      await waitFor(() => {
        expect(notificationMock.error).toHaveBeenCalledWith(
          expect.objectContaining({
            message: "Selected skills changed elsewhere",
            description: "The stale selection was cleared. Select current versions and try again."
          })
        )
      })
      expect(invalidateSpy).toHaveBeenCalledWith(
        expect.objectContaining({ queryKey: ["skills"] })
      )
      await waitFor(() => {
        expect(screen.queryByTestId("skills-selection-actions")).not.toBeInTheDocument()
      })
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
      await chooseRowMoreAction("skill-1", "Delete")

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

    await chooseFilter("Skill mode filter", "Fork")

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
    await openFilters()
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

    await openViewOptions()
    fireEvent.click(screen.getByRole("radio", { name: "Compact" }))

    expect(table).toHaveAttribute("data-density", "compact")
    expect(window.localStorage.getItem("tldw:skills-manager:table-preferences:v1")).toContain(
      "\"density\":\"compact\""
    )
  })

  it("renders equivalent skill workflows with touch-sized actions on mobile", async () => {
    vi.spyOn(window, "matchMedia").mockImplementation((query: string) => ({
      matches: query === "(max-width: 767px)",
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }) as MediaQueryList)
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByTestId("skills-mobile-list")).toBeInTheDocument()
    expect(await screen.findByText("skill-1")).toBeInTheDocument()
    expect(screen.queryByTestId("skills-table")).not.toBeInTheDocument()
    const actionNames = [
      "View skill-1",
      "Use skill-1 in chat",
      "Copy invocation for skill-1",
      "Test run skill-1",
      "More actions for skill-1"
    ]
    for (const name of actionNames) {
      expect(screen.getByRole("button", { name })).toHaveClass("min-h-11", "min-w-11")
    }
    expect(
      screen.getByRole("checkbox", { name: "Select skill-1" }).closest(".ant-checkbox-wrapper")
    ).toHaveClass(
      "min-h-11",
      "min-w-11"
    )
    expect(screen.getByRole("button", { name: "skill-1" })).toHaveClass("min-h-11")

    fireEvent.click(screen.getByRole("button", { name: "View skill-1" }))
    expect(screen.getByTestId("skill-details-open")).toHaveTextContent("skill-1")
  })

  it("offers server-backed sorting without desktop table headers on mobile", async () => {
    vi.spyOn(window, "matchMedia").mockImplementation((query: string) => ({
      matches: query === "(max-width: 767px)",
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }) as MediaQueryList)
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByTestId("skills-mobile-list")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /View options/ }))
    const sortSelect = await screen.findByRole("combobox", { name: "Sort by" })
    expect(screen.queryByText("Table density")).not.toBeInTheDocument()
    expect(screen.queryByRole("checkbox", { name: "Description" })).not.toBeInTheDocument()

    fireEvent.mouseDown(sortSelect)
    fireEvent.click(await screen.findByText("Modified (newest)"))

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({ sort: "last_modified", order: "desc" })
      )
    })
  })

  it.each([
    ["created_at", "asc", "Created (oldest)"],
    ["created_at", "desc", "Created (newest)"],
    ["last_modified", "asc", "Modified (oldest)"],
    ["last_modified", "desc", "Modified (newest)"]
  ] as const)(
    "shows the %s %s URL sort in the visible selector",
    async (sort, order, label) => {
      tldwClientMock.listSkills.mockResolvedValue({
        skills: [makeSkill(1)],
        count: 1,
        total: 1,
        limit: 10,
        offset: 0
      })

      renderManager(`/skills?sort=${sort}&order=${order}`)

      await screen.findByText("skill-1")
      fireEvent.click(screen.getByRole("button", { name: "View options" }))
      const sortSelect = await screen.findByRole("combobox", { name: "Sort by" })
      expect(sortSelect.closest(".ant-select")).toHaveTextContent(label)
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith(
        expect.objectContaining({ sort, order })
      )
    }
  )

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
      expect(tldwClientMock.previewSkillImport).toHaveBeenCalledWith(
        {
          content: "---\nname: imported-skill\ndescription: imported\n---\n\nBody"
        },
        { signal: expect.any(AbortSignal) }
      )
    })
    expect(tldwClientMock.importSkill).not.toHaveBeenCalled()
    expect(await within(dialog).findByText("Import review")).toBeInTheDocument()
    expect(within(dialog).getByText("imported-skill")).toBeInTheDocument()

    fireEvent.click(within(dialog).getByRole("button", { name: "Import skill" }))

    await waitFor(() => {
      expect(tldwClientMock.importSkill).toHaveBeenCalledWith(
        {
          content: "---\nname: imported-skill\ndescription: imported\n---\n\nBody",
          overwrite: false
        },
        expect.objectContaining({ signal: expect.anything() })
      )
    })
    expect(
      window.sessionStorage.getItem("tldw:skills:import-text-draft:v1")
    ).toBeNull()

    const successActions = await screen.findByTestId("skills-success-actions")
    expect(successActions).toHaveAttribute("data-ds-component", "Alert")
    expect(successActions).toHaveTextContent("Skill imported")
    expect(within(successActions).getByRole("button", { name: "Close" })).toBeInTheDocument()
    fireEvent.click(within(successActions).getByRole("button", { name: "View skill" }))
    expect(await screen.findByTestId("skill-details-open")).toHaveTextContent(
      "Skill details: imported-skill"
    )
    expect(tldwClientMock.getSkill).not.toHaveBeenCalled()
  })

  it("starts a replacement text preview while the stale request remains unresolved", async () => {
    let staleSignal: AbortSignal | undefined
    let resolveStalePreview: ((value: SkillImportPreviewResponse) => void) | undefined
    tldwClientMock.previewSkillImport
      .mockImplementationOnce(
        (_payload: unknown, options?: { signal?: AbortSignal }) => {
          staleSignal = options?.signal
          return new Promise((resolve) => {
            resolveStalePreview = resolve
          })
        }
      )
      .mockResolvedValueOnce({
        valid: true,
        errors: [],
        name: "new-skill",
        description: "new",
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
    renderManager()
    await waitFor(() => expect(tldwClientMock.listSkills).toHaveBeenCalled())

    fireEvent.click(screen.getByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))
    const dialog = await screen.findByRole("dialog", { name: "Import Skill from Text" })
    const contentInput = within(dialog).getByLabelText("SKILL.md Content")
    fireEvent.change(contentInput, {
      target: { value: "---\nname: old-skill\n---\n\nOld body" }
    })
    fireEvent.click(within(dialog).getByRole("button", { name: "Review import" }))
    await waitFor(() => expect(tldwClientMock.previewSkillImport).toHaveBeenCalledTimes(1))

    fireEvent.change(contentInput, {
      target: { value: "---\nname: new-skill\n---\n\nNew body" }
    })
    await waitFor(() => expect(staleSignal?.aborted).toBe(true))
    const reviewButton = within(dialog).getByRole("button", { name: "Review import" })
    await waitFor(() => {
      expect(reviewButton).not.toHaveClass("ant-btn-loading")
    })
    fireEvent.click(reviewButton)

    await waitFor(() => expect(tldwClientMock.previewSkillImport).toHaveBeenCalledTimes(2))
    expect(await within(dialog).findByText("new-skill")).toBeInTheDocument()
    expect(within(dialog).queryByText("Existing skill detected")).not.toBeInTheDocument()
    expect(
      within(dialog).queryByRole("switch", { name: "Overwrite existing skill" })
    ).not.toBeInTheDocument()
    expect(tldwClientMock.importSkill).not.toHaveBeenCalled()

    await act(async () => {
      resolveStalePreview?.({
        valid: true,
        errors: [],
        name: "old-skill",
        description: "old",
        argument_hint: null,
        disable_model_invocation: false,
        user_invocable: true,
        allowed_tools: null,
        model: null,
        context: "inline",
        supporting_file_count: 0,
        conflict: true,
        can_overwrite: true,
        existing_version: 1
      })
      await Promise.resolve()
    })
    expect(within(dialog).getByText("new-skill")).toBeInTheDocument()
    expect(within(dialog).queryByText("Existing skill detected")).not.toBeInTheDocument()
  })

  it("recovers an unfinished text import within the browser session", async () => {
    const first = renderManager()
    await waitFor(() => expect(tldwClientMock.listSkills).toHaveBeenCalled())

    fireEvent.click(screen.getByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))
    fireEvent.change(screen.getByLabelText("SKILL.md Content"), {
      target: { value: "Recovered import content" }
    })
    first.unmount()

    renderManager()
    fireEvent.click(await screen.findByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))

    const dialog = await screen.findByRole("dialog", { name: "Import Skill from Text" })
    expect(within(dialog).getByRole("status")).toHaveTextContent(
      "Recovered your unfinished import from this session."
    )
    expect(within(dialog).getByLabelText("SKILL.md Content")).toHaveValue(
      "Recovered import content"
    )
  })

  it("does not recover an import draft after switching servers", async () => {
    tldwClientMock.getConfig.mockResolvedValue({
      serverUrl: "https://server-one.example",
      authMode: "single-user"
    })
    const first = renderManager()
    await waitFor(() => expect(tldwClientMock.getConfig).toHaveBeenCalled())
    fireEvent.click(await screen.findByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))
    fireEvent.change(screen.getByLabelText("SKILL.md Content"), {
      target: { value: "Private server-one import" }
    })
    first.unmount()

    tldwClientMock.getConfig.mockResolvedValue({
      serverUrl: "https://server-two.example",
      authMode: "single-user"
    })
    renderManager()
    await waitFor(() => expect(tldwClientMock.getConfig).toHaveBeenCalledTimes(2))
    fireEvent.click(await screen.findByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))

    const dialog = await screen.findByRole("dialog", { name: "Import Skill from Text" })
    expect(within(dialog).getByLabelText("SKILL.md Content")).toHaveValue("")
    expect(within(dialog).queryByRole("status")).not.toBeInTheDocument()
  })

  it("reloads skills and clears server-scoped UI state after a live server switch", async () => {
    const serverOneSkill = { ...makeSkill(1), name: "shared-skill", description: "Server one" }
    const serverTwoSkill = { ...makeSkill(2), name: "server-two-skill", description: "Server two" }
    const serverTwoConfig = {
      serverUrl: "https://server-two.example",
      authMode: "single-user"
    }
    let resolveServerTwoConfig: ((config: typeof serverTwoConfig) => void) | undefined
    let serverTwoActive = false
    tldwClientMock.getConfig
      .mockResolvedValueOnce({
        serverUrl: "https://server-one.example",
        authMode: "single-user"
      })
      .mockImplementation(() => new Promise((resolve) => {
        resolveServerTwoConfig = resolve
      }))
    tldwClientMock.listSkills.mockImplementation(async () => (
      serverTwoActive
        ? {
          skills: [serverTwoSkill],
          count: 1,
          total: 1,
          limit: 10,
          offset: 0
        }
        : {
          skills: [serverOneSkill],
          count: 1,
          total: 1,
          limit: 10,
          offset: 0
        }
    ))
    renderManager()

    fireEvent.click(await screen.findByRole("checkbox", { name: "Select shared-skill" }))
    fireEvent.click(screen.getByRole("button", { name: "View shared-skill" }))
    expect(screen.getByTestId("skills-selection-actions")).toHaveTextContent("1 selected")
    expect(screen.getByTestId("skill-details-open")).toHaveTextContent("shared-skill")

    const listCallsBeforeSwitch = tldwClientMock.listSkills.mock.calls.length
    serverTwoActive = true
    window.dispatchEvent(new Event("tldw:config-updated"))

    await waitFor(() => expect(tldwClientMock.getConfig).toHaveBeenCalledTimes(2))
    expect(screen.queryByTestId("skills-selection-actions")).not.toBeInTheDocument()
    expect(screen.queryByTestId("skill-details-open")).not.toBeInTheDocument()
    resolveServerTwoConfig?.(serverTwoConfig)
    await waitFor(() => {
      expect(tldwClientMock.listSkills.mock.calls.length).toBeGreaterThan(listCallsBeforeSwitch)
    })
    expect(await screen.findByRole("button", { name: "View server-two-skill" }))
      .toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "View shared-skill" })).not.toBeInTheDocument()
    expect(screen.queryByTestId("skills-selection-actions")).not.toBeInTheDocument()
    expect(screen.queryByTestId("skill-details-open")).not.toBeInTheDocument()
  })

  it("clears scope-bound import state when config changes before initial identity resolution", async () => {
    const initialConfig = {
      serverUrl: "https://server-one.example",
      authMode: "single-user"
    }
    let resolveInitialConfig: ((config: typeof initialConfig) => void) | undefined
    tldwClientMock.getConfig
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveInitialConfig = resolve
      }))
      .mockResolvedValue({
        serverUrl: "https://server-two.example",
        authMode: "single-user"
      })

    renderManager()
    fireEvent.click(await screen.findByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))
    expect(await screen.findByRole("dialog", { name: "Import Skill from Text" }))
      .toBeInTheDocument()

    act(() => window.dispatchEvent(new Event("tldw:config-updated")))

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Import Skill from Text" }))
        .not.toBeInTheDocument()
    })
    act(() => resolveInitialConfig?.(initialConfig))
  })

  it("destroys and invalidates skill confirmations when the server scope changes", async () => {
    const destroy = vi.fn()
    let confirmConfig: { onOk?: () => void | Promise<void> } | undefined
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce((config) => {
      confirmConfig = config as { onOk?: () => void | Promise<void> }
      return { destroy, update: vi.fn() } as any
    })
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    try {
      renderManager()
      await screen.findByText("skill-1")
      await chooseRowMoreAction("skill-1", "Delete")
      expect(confirmConfig).toBeDefined()

      act(() => window.dispatchEvent(new Event("tldw:config-updated")))

      expect(destroy).toHaveBeenCalledTimes(1)
      await confirmConfig?.onOk?.()
      expect(tldwClientMock.deleteSkill).not.toHaveBeenCalled()
    } finally {
      confirmSpy.mockRestore()
    }
  })

  it("loads skills with an isolated query scope when identity resolution fails", async () => {
    tldwClientMock.getConfig.mockRejectedValue(new Error("config unavailable"))
    tldwClientMock.listSkills.mockResolvedValue({
      skills: [{ ...makeSkill(1), name: "fallback-scope-skill" }],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })

    renderManager()

    expect(await screen.findByRole("button", { name: "View fallback-scope-skill" }))
      .toBeInTheDocument()
    expect(tldwClientMock.listSkills).toHaveBeenCalled()
  })

  it("does not recover an import draft after switching users", async () => {
    tldwClientMock.getConfig.mockResolvedValue({
      serverUrl: "https://shared.example",
      authMode: "multi-user",
      accessToken: "opaque-token"
    })
    tldwAuthMock.getCurrentUser
      .mockResolvedValueOnce({ id: 1, username: "first", is_active: true })
      .mockResolvedValue({ id: 2, username: "second", is_active: true })
    const first = renderManager()
    await waitFor(() => expect(tldwAuthMock.getCurrentUser).toHaveBeenCalledTimes(1))
    fireEvent.click(await screen.findByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))
    fireEvent.change(screen.getByLabelText("SKILL.md Content"), {
      target: { value: "Private user-one import" }
    })
    first.unmount()

    renderManager()
    await waitFor(() => expect(tldwAuthMock.getCurrentUser).toHaveBeenCalledTimes(2))
    fireEvent.click(await screen.findByRole("button", { name: "Import" }))
    fireEvent.click(await screen.findByText("Import Text"))

    const dialog = await screen.findByRole("dialog", { name: "Import Skill from Text" })
    expect(within(dialog).getByLabelText("SKILL.md Content")).toHaveValue("")
    expect(within(dialog).queryByRole("status")).not.toBeInTheDocument()
  })

  it("guards a dirty text import before discarding it", async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementation(vi.fn() as never)
    renderManager()
    await waitFor(() => expect(tldwClientMock.listSkills).toHaveBeenCalled())

    await user.click(screen.getByRole("button", { name: "Import" }))
    await user.click(await screen.findByText("Import Text"))
    const dialog = await screen.findByRole("dialog", { name: "Import Skill from Text" })
    await user.type(within(dialog).getByLabelText("SKILL.md Content"), "Unsaved import")
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))

    expect(confirmSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Discard unfinished import?",
        cancelText: "Keep editing",
        okText: "Discard import"
      })
    )
    expect(dialog).toBeInTheDocument()

    const config = confirmSpy.mock.calls[0][0] as { onOk?: () => void }
    act(() => config.onOk?.())
    expect(dialog).not.toBeInTheDocument()
    expect(window.sessionStorage.getItem("tldw:skills:import-text-draft:v1")).toBeNull()
    confirmSpy.mockRestore()
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
    expect(await screen.findByTestId("skill-details-open")).toHaveTextContent(
      "Skill details: fallback-skill"
    )
    expect(tldwClientMock.getSkill).not.toHaveBeenCalled()
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
      expect(tldwClientMock.importSkill).toHaveBeenCalledWith(
        {
          content: "---\nname: existing-skill\ndescription: replacement\n---\n\nReplacement",
          overwrite: true,
          expected_version: 3
        },
        expect.objectContaining({ signal: expect.anything() })
      )
    })
  })

  it("previews a file import before importing the uploaded file", async () => {
    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Import" }))

    const input = screen.getByLabelText("Import skill file") as HTMLInputElement
    expect(document.querySelectorAll("input[type='file']")).toHaveLength(1)

    const file = new File(["# skill"], "my-skill.md", { type: "text/markdown" })
    fireEvent.change(input, { target: { files: [file] } })

    await waitFor(() => {
      expect(tldwClientMock.previewSkillImportFile).toHaveBeenCalledWith(
        file,
        expect.objectContaining({ signal: expect.anything() })
      )
    })
    expect(tldwClientMock.importSkillFile).not.toHaveBeenCalled()

    const dialog = await screen.findByRole("dialog", {
      name: "Review Skill Import"
    })
    expect(within(dialog).getByText("imported-file-skill")).toBeInTheDocument()

    fireEvent.click(within(dialog).getByRole("button", { name: "Import skill" }))

    await waitFor(() => {
      expect(tldwClientMock.importSkillFile).toHaveBeenCalledWith(
        file,
        expect.objectContaining({
          overwrite: false,
          signal: expect.anything()
        })
      )
    })
  })

  it("submits the previewed version when overwriting from a file", async () => {
    tldwClientMock.previewSkillImportFile.mockResolvedValueOnce({
      valid: true,
      errors: [],
      name: "existing-file-skill",
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
      existing_version: 7
    })
    renderManager()
    await waitFor(() => expect(tldwClientMock.listSkills).toHaveBeenCalled())
    const file = new File(["Replacement"], "existing-file-skill.md", {
      type: "text/markdown"
    })

    fireEvent.change(screen.getByLabelText("Import skill file"), {
      target: { files: [file] }
    })
    const dialog = await screen.findByRole("dialog", { name: "Review Skill Import" })
    fireEvent.click(within(dialog).getByRole("switch", { name: "Overwrite existing skill" }))
    fireEvent.click(within(dialog).getByRole("button", { name: "Import skill" }))

    await waitFor(() => {
      expect(tldwClientMock.importSkillFile).toHaveBeenCalledWith(file, {
        overwrite: true,
        expectedVersion: 7,
        signal: expect.anything()
      })
    })
  })

  it("ignores an older file preview that finishes after the latest selection", async () => {
    let resolveOldPreview: ((value: SkillImportPreviewResponse) => void) | undefined
    tldwClientMock.previewSkillImportFile
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveOldPreview = resolve
      }))
      .mockResolvedValueOnce({
        valid: true,
        errors: [],
        name: "new-file-skill",
        description: "new file",
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
    renderManager()
    await waitFor(() => expect(tldwClientMock.listSkills).toHaveBeenCalled())
    const input = screen.getByLabelText("Import skill file") as HTMLInputElement
    const oldFile = new File(["old"], "old.md", { type: "text/markdown" })
    const newFile = new File(["new"], "new.md", { type: "text/markdown" })

    fireEvent.change(input, { target: { files: [oldFile] } })
    await waitFor(() => expect(tldwClientMock.previewSkillImportFile).toHaveBeenCalledTimes(1))
    fireEvent.change(input, { target: { files: [newFile] } })

    const dialog = await screen.findByRole("dialog", { name: "Review Skill Import" })
    expect(within(dialog).getByText("new-file-skill")).toBeInTheDocument()
    await act(async () => {
      resolveOldPreview?.({
        valid: true,
        errors: [],
        name: "old-file-skill",
        description: "old file",
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
    })

    expect(within(dialog).getByText("new-file-skill")).toBeInTheDocument()
    expect(within(dialog).queryByText("old-file-skill")).not.toBeInTheDocument()
  })

  it("guards a reviewed file before discarding the import", async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementation(vi.fn() as never)
    renderManager()
    await waitFor(() => expect(tldwClientMock.listSkills).toHaveBeenCalled())

    const input = screen.getByLabelText("Import skill file") as HTMLInputElement
    const file = new File(["# skill"], "guarded-skill.md", { type: "text/markdown" })
    await user.upload(input, file)
    const dialog = await screen.findByRole("dialog", { name: "Review Skill Import" })

    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    expect(confirmSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Discard reviewed file import?" })
    )
    expect(dialog).toBeInTheDocument()

    const config = confirmSpy.mock.calls[0][0] as { onOk?: () => void }
    act(() => config.onOk?.())
    expect(dialog).not.toBeInTheDocument()
    confirmSpy.mockRestore()
  })

  it("seeds built-in skills via seedSkills action", async () => {
    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Seed Built-ins" }))
    fireEvent.click(await screen.findByText("Seed Missing Only"))

    await waitFor(() => {
      expect(tldwClientMock.seedSkills).toHaveBeenCalledWith(
        { overwrite: false },
        expect.objectContaining({ signal: expect.anything() })
      )
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
        expect(tldwClientMock.seedSkills).toHaveBeenCalledWith(
          { overwrite: true },
          expect.objectContaining({ signal: expect.anything() })
        )
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
