import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"

const readSource = (relativePath: string) =>
  readFileSync(resolve(process.cwd(), relativePath), "utf8")

describe("CodeQL source contracts", () => {
  it("keeps the document job identifier out of the console format argument", () => {
    const source = readSource(
      "src/components/Common/Playground/DocumentGeneratorDrawer.tsx"
    )

    expect(source).toContain(
      'console.debug("Failed to refresh job", job.job_id, err)'
    )
  })

  it("keeps timeline identifiers out of console format arguments", () => {
    const source = readSource("src/services/timeline/api.ts")

    expect(source).toContain(
      'console.error("Failed to get conversation", conversationId, error)'
    )
    expect(source).toContain(
      'console.error("Failed to list conversations by root", rootId, error)'
    )
  })

  it("uses predicate rules without a RegExp-like match member", () => {
    const source = readSource("src/utils/provider-registry.ts")

    expect(source).toContain("matches: (value)")
    expect(source).toContain("rule.matches(value)")
    expect(source).not.toContain("rule.match(value)")
  })
})
