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
  SkillDrawer: (props: { open: boolean }) => {
    skillDrawerMock(props)
    return props.open ? <div data-testid="skill-drawer-open">Skill drawer open</div> : null
  }
}))

vi.mock("../SkillPreview", () => ({
  SkillPreview: () => null
}))

describe("SkillsManager imports", () => {
  let queryClient: QueryClient

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })
    vi.clearAllMocks()
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
      seeded: ["summarize", "code-review", "feynman-technique"],
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

    fireEvent.click(within(emptyState).getByRole("button", { name: "Import" }))
    expect(
      await screen.findByRole("dialog", {
        name: "Import Skill from Text"
      })
    ).toBeInTheDocument()
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
})
