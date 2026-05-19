import { existsSync, readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const loadSource = (...candidates: string[]) => {
  const path = candidates.find((candidate) => existsSync(candidate))
  if (!path) {
    throw new Error(`Missing extension route source: ${candidates.join(" | ")}`)
  }
  return readFileSync(path, "utf8")
}

describe("extension option flashcards shared workspace parity", () => {
  it("wraps the flashcards workspace in the shared layout and error boundary", () => {
    const source = loadSource(
      "apps/tldw-frontend/extension/routes/option-flashcards.tsx",
      "tldw-frontend/extension/routes/option-flashcards.tsx",
      "extension/routes/option-flashcards.tsx"
    )

    expect(source).toContain("FlashcardsWorkspace")
    expect(source).toContain("RouteErrorBoundary")
    expect(source).toContain("OptionLayout")
    expect(source).toContain('routeId="flashcards"')
    expect(source).toContain('routeLabel="Flashcards"')
  })
})
