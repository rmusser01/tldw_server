import { readdirSync, readFileSync, statSync } from "node:fs"
import path from "node:path"
import type { Page } from "@playwright/test"
import { afterEach, describe, expect, it, vi } from "vitest"
import { WorldBooksPage } from "../e2e/utils/page-objects/WorldBooksPage"

vi.mock("@playwright/test", () => ({
  expect: () => ({ toBeVisible: vi.fn().mockResolvedValue(undefined) }),
}))
vi.mock("../e2e/utils/helpers", () => ({
  waitForAppShell: vi.fn(),
  waitForConnection: vi.fn(),
}))

const pageObjectsDir = path.resolve(__dirname, "../e2e/utils/page-objects")
const fixedSleepGuardFiles = [
  "ChatPage.ts",
  "CharactersPage.ts",
  "EvaluationsPage.ts",
  "FlashcardsPage.ts",
  "NotesPage.ts",
  "PromptsWorkspacePage.ts",
  "SearchPage.ts",
  "WorldBooksPage.ts",
  "WritingPlaygroundPage.ts",
]

const listPageObjectFiles = (dir: string): string[] => {
  const entries = readdirSync(dir)
  const files: string[] = []

  for (const entry of entries) {
    const fullPath = path.join(dir, entry)
    const stats = statSync(fullPath)
    if (stats.isDirectory()) {
      files.push(...listPageObjectFiles(fullPath))
      continue
    }
    if (entry.endsWith("Page.ts")) {
      files.push(fullPath)
    }
  }

  return files.sort()
}

describe("e2e page object readiness contracts", () => {
  it("keeps page objects off direct networkidle waits", () => {
    const files = listPageObjectFiles(pageObjectsDir)

    for (const file of files) {
      const source = readFileSync(file, "utf8")
      expect(source).not.toContain('waitForLoadState("networkidle"')
    }
  })

  it("keeps high-traffic page objects off readiness sleeps while allowing bounded rate-limit backoff", () => {
    const files = listPageObjectFiles(pageObjectsDir).filter((file) =>
      fixedSleepGuardFiles.includes(path.basename(file))
    )

    for (const file of files) {
      let source = readFileSync(file, "utf8")
      if (path.basename(file) === "WorldBooksPage.ts") {
        // f6fabc345e added one increasing delay only after a create HTTP 429.
        // The behavior tests below prove this exception cannot delay successful
        // readiness, retry other errors, or retry indefinitely.
        source = source.replace("await this.page.waitForTimeout(1_000 * attempt)", "")
      }
      expect(source, path.basename(file)).not.toContain("waitForTimeout(")
    }
  })
})

describe("WorldBooks create rate-limit backoff", () => {
  afterEach(() => vi.restoreAllMocks())

  const setup = (statuses: number[]) => {
    const dialogHidden = vi.fn().mockResolvedValue(undefined)
    const waitForTimeout = vi.fn().mockResolvedValue(undefined)
    const locator = { waitFor: dialogHidden, filter: vi.fn() }
    locator.filter.mockReturnValue(locator)
    const page = { getByRole: vi.fn(() => locator), waitForTimeout }
    const worldBooks = new WorldBooksPage(page as unknown as Page)
    vi.spyOn(worldBooks, "clickNewWorldBook").mockResolvedValue(undefined)
    vi.spyOn(worldBooks, "fillWorldBookForm").mockResolvedValue(undefined)
    vi.spyOn(worldBooks, "searchWorldBooks").mockResolvedValue(undefined)
    const submit = vi.spyOn(worldBooks, "submitWorldBookForm").mockResolvedValue(undefined)
    const response = vi.spyOn(worldBooks, "waitForApiCall")
    for (const status of statuses) response.mockResolvedValueOnce({ status, body: null })
    return { worldBooks, waitForTimeout, submit, response, dialogHidden }
  }

  it("completes a successful create without sleeping", async () => {
    const { worldBooks, waitForTimeout, submit, response, dialogHidden } = setup([201])
    await worldBooks.createWorldBook("Travel notes")

    expect(submit).toHaveBeenCalledTimes(1)
    expect(response).toHaveBeenCalledWith(/\/api\/v1\/characters\/world-books\/?$/, "POST")
    expect(dialogHidden).toHaveBeenCalledWith({ state: "hidden", timeout: 15_000 })
    expect(worldBooks.searchWorldBooks).toHaveBeenCalledWith("Travel notes")
    expect(waitForTimeout).not.toHaveBeenCalled()
  })

  it("backs off after rate limiting and stops when creation succeeds", async () => {
    const { worldBooks, waitForTimeout, submit } = setup([429, 201])
    await worldBooks.createWorldBook("Travel notes")

    expect(submit).toHaveBeenCalledTimes(2)
    expect(waitForTimeout.mock.calls).toEqual([[1000]])
  })

  it.each([400, 401, 403, 500])("fails immediately on HTTP %i without sleeping", async (status) => {
    const { worldBooks, waitForTimeout, submit } = setup([status])
    await expect(worldBooks.createWorldBook("Travel notes")).rejects.toThrow(`status ${status}`)

    expect(submit).toHaveBeenCalledTimes(1)
    expect(waitForTimeout).not.toHaveBeenCalled()
  })

  it("fails after four rate-limited attempts without a final sleep", async () => {
    const { worldBooks, waitForTimeout, submit } = setup([429, 429, 429, 429])
    await expect(worldBooks.createWorldBook("Travel notes")).rejects.toThrow("status 429")

    expect(submit).toHaveBeenCalledTimes(4)
    expect(waitForTimeout.mock.calls).toEqual([[1000], [2000], [3000]])
  })
})
