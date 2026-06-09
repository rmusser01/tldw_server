import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { message, Modal } from "antd"
import { SkillsManager } from "../Manager"

const tldwClientMock = vi.hoisted(() => ({
  listSkills: vi.fn(),
  getSkill: vi.fn(),
  deleteSkill: vi.fn(),
  exportSkill: vi.fn(),
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
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

vi.mock("../SkillDrawer", () => ({
  SkillDrawer: (props: { open: boolean; onSaved: (skillName?: string) => void }) => {
    skillDrawerMock(props)
    return props.open ? (
      <div data-testid="skill-drawer-open">
        Skill drawer open
        <button type="button" onClick={() => props.onSaved("created-skill")}>
          Complete create
        </button>
      </div>
    ) : null
  }
}))

vi.mock("../SkillPreview", () => ({
  SkillPreview: (props: { skillName: string | null }) => {
    skillPreviewMock(props)
    return props.skillName ? (
      <div data-testid="skill-preview-open">Test run: {props.skillName}</div>
    ) : null
  }
}))

const makeSkill = (index: number) => ({
  name: `skill-${index}`,
  description: `Skill ${index}`,
  argument_hint: null,
  user_invocable: true,
  disable_model_invocation: false,
  context: "inline" as const
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

  it("shows an explicit load error instead of the beginner empty state", async () => {
    tldwClientMock.listSkills.mockRejectedValueOnce(new Error("backend down"))

    renderManager()

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Failed to load skills")
    expect(alert).toHaveTextContent("backend down")
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
      expect(tldwClientMock.listSkills).toHaveBeenCalledWith({ limit: 10, offset: 10 })
    })
    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith({ limit: 10, offset: 0 })
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
      expect(tldwClientMock.listSkills).toHaveBeenLastCalledWith({
        q: "needle",
        limit: 10,
        offset: 0
      })
    })
    expect(await screen.findByText("omega-research")).toBeInTheDocument()
    expect(screen.getByText("1 skill")).toBeInTheDocument()
  })

  it("imports a skill from text via importSkill", async () => {
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

    fireEvent.click(within(dialog).getByRole("button", { name: "Import" }))

    await waitFor(() => {
      expect(tldwClientMock.importSkill).toHaveBeenCalledWith({
        content: "---\nname: imported-skill\ndescription: imported\n---\n\nBody",
        overwrite: false
      })
    })

    const successActions = await screen.findByTestId("skills-success-actions")
    expect(successActions).toHaveTextContent("Skill imported")
    fireEvent.click(within(successActions).getByRole("button", { name: "View skill" }))

    await waitFor(() => {
      expect(tldwClientMock.getSkill).toHaveBeenCalledWith("imported-skill")
    })
  })

  it("falls back to the validated import name when the API returns an invalid name", async () => {
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

    fireEvent.click(within(dialog).getByRole("button", { name: "Import" }))

    const successActions = await screen.findByTestId("skills-success-actions")
    fireEvent.click(within(successActions).getByRole("button", { name: "View skill" }))

    await waitFor(() => {
      expect(tldwClientMock.getSkill).toHaveBeenCalledWith("fallback-skill")
    })
  })

  it("keeps file import flow functional via importSkillFile", async () => {
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
      expect(tldwClientMock.importSkillFile).toHaveBeenCalledWith(file)
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
    expect(successActions).toHaveTextContent("Built-in skills seeded")

    fireEvent.click(within(successActions).getByRole("button", { name: "Test summarize" }))
    expect(screen.getByTestId("skill-preview-open")).toHaveTextContent("summarize")

    fireEvent.click(within(successActions).getByRole("button", { name: "Copy /skill summarize" }))
    await waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith("/skill summarize")
    })
  })

  it("seeds built-in skills with overwrite via seedSkills action", async () => {
    renderManager()

    await waitFor(() => {
      expect(tldwClientMock.listSkills).toHaveBeenCalled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Seed Built-ins" }))
    fireEvent.click(await screen.findByText("Seed and Overwrite Existing"))

    await waitFor(() => {
      expect(tldwClientMock.seedSkills).toHaveBeenCalledWith({ overwrite: true })
    })
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
})
