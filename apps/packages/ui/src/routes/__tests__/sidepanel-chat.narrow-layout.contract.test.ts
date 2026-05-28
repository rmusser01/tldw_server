import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const resolveSourcePath = (relativePath: string) => {
  const candidates = [
    path.resolve(process.cwd(), relativePath),
    path.resolve(process.cwd(), "../packages/ui", relativePath),
    path.resolve(process.cwd(), "apps/packages/ui", relativePath),
  ]

  const resolved = candidates.find((candidate) => existsSync(candidate))
  if (!resolved) {
    throw new Error(`Unable to locate ${relativePath}`)
  }

  return resolved
}

const sidepanelChatSource = readFileSync(
  resolveSourcePath("src/routes/sidepanel-chat.tsx"),
  "utf8",
)
const sidepanelFormSource = readFileSync(
  resolveSourcePath("src/components/Sidepanel/Chat/form.tsx"),
  "utf8",
)
const controlRowSource = readFileSync(
  resolveSourcePath("src/components/Sidepanel/Chat/ControlRow.tsx"),
  "utf8",
)

describe("sidepanel chat narrow layout contract", () => {
  it("keeps the sidepanel chat shell from exporting horizontal overflow at 390px", () => {
    expect(sidepanelChatSource).toContain(
      'className="flex h-dvh w-full min-w-0 overflow-x-hidden"',
    )
    expect(sidepanelChatSource).toContain(
      'className="relative flex h-dvh min-w-0 flex-1 flex-col overflow-x-hidden bg-bg"',
    )
    expect(sidepanelChatSource).toContain(
      "relative flex min-h-0 min-w-0 flex-1 flex-col items-center overflow-x-hidden bg-bg",
    )
    expect(sidepanelChatSource).toContain(
      "relative z-10 flex min-w-0 flex-1 w-full flex-col items-center overflow-x-hidden overflow-y-auto",
    )
  })

  it("allows the sticky composer and control rows to wrap within a 390px sidepanel", () => {
    expect(sidepanelChatSource).toContain(
      'className="absolute bottom-0 left-0 right-0 z-10 w-full min-w-0"',
    )
    expect(sidepanelFormSource).toContain(
      "flex w-full min-w-0 flex-col items-center",
    )
    expect(sidepanelFormSource).toContain(
      "relative z-10 flex w-full min-w-0 flex-col items-center justify-center",
    )
    expect(sidepanelFormSource).toContain(
      "relative w-full min-w-0 max-w-[64rem]",
    )
    expect(sidepanelFormSource).toContain(
      'className="min-w-0 flex-1 flex flex-col items-center"',
    )
    expect(sidepanelFormSource).toContain("flex w-full min-w-0 flex-col px-1")
    expect(sidepanelFormSource).toContain(
      'className="mt-2 flex min-w-0 flex-col gap-2"',
    )
    expect(sidepanelFormSource).toContain(
      'className="flex w-full min-w-0 flex-row flex-wrap items-center justify-between gap-1.5"',
    )
    expect(sidepanelFormSource).toContain(
      'className="flex min-w-0 flex-wrap items-center justify-end gap-2"',
    )
    expect(sidepanelFormSource).toContain(
      'className="flex min-w-0 flex-wrap items-center gap-2"',
    )
    expect(controlRowSource).toContain(
      'data-testid="control-row" className="flex min-w-0 flex-1 flex-wrap items-center gap-2"',
    )
  })
})
