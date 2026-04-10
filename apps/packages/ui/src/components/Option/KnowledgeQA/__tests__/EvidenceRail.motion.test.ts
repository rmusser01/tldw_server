import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const sourcePath = path.resolve(
  process.cwd(),
  "src/components/Option/KnowledgeQA/evidence/EvidenceRail.tsx"
)

describe("EvidenceRail motion preferences", () => {
  it("applies reduced-motion guards to both desktop and mobile entry animations", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain("motion-reduce:animate-none")
    expect(source).toContain("motion-reduce:transition-none")
    expect(source).toContain("motion-safe:slide-in-from-right")
    expect(source).toMatch(
      /<aside className="absolute right-0 top-0 h-full w-\[88vw\] max-w-md border-l border-border bg-surface shadow-xl[^"]*motion-reduce:animate-none/
    )
  })
})
