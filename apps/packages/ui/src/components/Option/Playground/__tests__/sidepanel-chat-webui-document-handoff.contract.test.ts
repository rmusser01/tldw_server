import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const resolvePlaygroundPath = () => {
  const candidates = [
    path.resolve(__dirname, "../Playground.tsx"),
    path.resolve(process.cwd(), "src/components/Option/Playground/Playground.tsx"),
    path.resolve(
      process.cwd(),
      "../packages/ui/src/components/Option/Playground/Playground.tsx"
    ),
    path.resolve(
      process.cwd(),
      "apps/packages/ui/src/components/Option/Playground/Playground.tsx"
    )
  ]

  const resolved = candidates.find((candidate) => existsSync(candidate))
  if (!resolved) {
    throw new Error("Unable to locate Playground source")
  }

  return resolved
}

const playgroundSource = readFileSync(resolvePlaygroundPath(), "utf8")

describe("sidepanel WebUI document handoff contract", () => {
  it("imports server-backed sidepanel document drafts into chat attachments", () => {
    expect(playgroundSource).toContain("getDocumentUploadDraft")
    expect(playgroundSource).toContain("deleteDocumentUploadDraft")
    expect(playgroundSource).toContain("chatDocumentDraftId")
    expect(playgroundSource).toContain("setUploadedFiles")
    expect(playgroundSource).toContain("setContextFiles")
  })
})
