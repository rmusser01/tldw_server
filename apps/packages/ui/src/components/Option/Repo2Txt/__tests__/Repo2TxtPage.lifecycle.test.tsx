import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { Repo2TxtPage } from "../Repo2TxtPage"

const mocks = vi.hoisted(() => ({ tree: vi.fn(), file: vi.fn() }))
vi.mock("../providers/GitHubProvider", () => ({
  GitHubProvider: class {
    validateUrl() {
      return true
    }
    fetchTree(...args: unknown[]) {
      return mocks.tree(...args)
    }
    fetchFile(...args: unknown[]) {
      return mocks.file(...args)
    }
  }
}))
vi.mock("../formatter/Formatter", () => ({
  Formatter: {
    formatAsync: async (_: unknown, files: Array<{ text: string }>) => ({
      directoryTree: "Directory Structure:",
      fileContents: files.map((f) => f.text).join("\n"),
      tokenCount: 2
    })
  }
}))

const nodes = [
  { path: "README.md", type: "blob" },
  { path: "guide.py", type: "blob" }
]
async function load() {
  fireEvent.click(screen.getByRole("button", { name: "GitHub", exact: true }))
  fireEvent.change(
    screen.getByPlaceholderText("https://github.com/owner/repo"),
    { target: { value: "https://github.com/test/public/tree/fixed" } }
  )
  fireEvent.click(
    screen.getByRole("button", { name: "Load Source", exact: true })
  )
  await screen.findByRole("checkbox", { name: "Select README.md" })
}
async function generate() {
  fireEvent.click(
    screen.getByRole("button", { name: "Generate Output", exact: true })
  )
  await waitFor(() =>
    expect(
      (
        screen.getByRole("textbox", {
          name: "Repo2txt output preview"
        }) as HTMLTextAreaElement
      ).value
    ).toContain("readable")
  )
}
beforeEach(() => {
  mocks.tree.mockReset().mockResolvedValue(nodes)
  mocks.file
    .mockReset()
    .mockImplementation(async (node) => ({
      path: node.path,
      text: "readable",
      lineCount: 1
    }))
})

describe("Repo2Txt operation lifecycle", () => {
  it("invalidates output immediately when selection changes", async () => {
    render(<Repo2TxtPage />)
    await load()
    await generate()
    fireEvent.click(screen.getByRole("checkbox", { name: "Select guide.py" }))
    expect(
      screen.getByRole("button", { name: "Download", exact: true })
    ).toBeDisabled()
    expect(
      screen.getByRole("textbox", { name: "Repo2txt output preview" })
    ).toHaveValue("")
  })
  it("clears old output and old source on a failed reload", async () => {
    render(<Repo2TxtPage />)
    await load()
    await generate()
    mocks.tree.mockRejectedValueOnce(new Error("Repository inaccessible"))
    fireEvent.click(
      screen.getByRole("button", { name: "Load Source", exact: true })
    )
    await screen.findByText("Repository inaccessible")
    expect(
      screen.getByRole("button", { name: "Download", exact: true })
    ).toBeDisabled()
    expect(
      screen.queryByRole("checkbox", { name: "Select README.md" })
    ).not.toBeInTheDocument()
  })
  it("cancels generation and ignores a late file result", async () => {
    const finishes: Array<(value: unknown) => void> = []
    mocks.file.mockImplementation(
      () =>
        new Promise((resolve) => {
          finishes.push(resolve)
        })
    )
    render(<Repo2TxtPage />)
    await load()
    fireEvent.click(
      screen.getByRole("button", { name: "Generate Output", exact: true })
    )
    await waitFor(() => expect(mocks.file).toHaveBeenCalled())
    fireEvent.click(screen.getByRole("button", { name: "Cancel", exact: true }))
    await act(async () => {
      finishes.forEach((finish) =>
        finish({ path: "guide.py", text: "late output", lineCount: 1 })
      )
    })
    expect(
      screen.getByRole("textbox", { name: "Repo2txt output preview" })
    ).toHaveValue("")
    expect(
      screen.getByRole("button", { name: "Generate Output", exact: true })
    ).toBeEnabled()
    expect(screen.getByRole("status")).toHaveTextContent(/cancelled/i)
  })
  it("cancels source loading and ignores a late tree", async () => {
    let finish!: (value: unknown) => void
    mocks.tree.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    render(<Repo2TxtPage />)
    fireEvent.click(screen.getByRole("button", { name: "GitHub", exact: true }))
    fireEvent.change(
      screen.getByPlaceholderText("https://github.com/owner/repo"),
      { target: { value: "https://github.com/test/public" } }
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Load Source", exact: true })
    )
    await waitFor(() => expect(mocks.tree).toHaveBeenCalled())
    fireEvent.click(screen.getByRole("button", { name: "Cancel", exact: true }))
    await act(async () => {
      finish(nodes)
    })
    expect(
      screen.queryByRole("checkbox", { name: "Select README.md" })
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Generate Output", exact: true })
    ).toBeDisabled()
  })
})
