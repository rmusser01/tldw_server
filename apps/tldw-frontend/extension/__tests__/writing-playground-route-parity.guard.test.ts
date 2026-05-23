import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("writing playground route parity", () => {
  it("tracks the actual shared route contract", () => {
    const webRoute = fs.readFileSync(
      path.resolve(__dirname, "../../../packages/ui/src/routes/option-writing-playground.tsx"),
      "utf8"
    )
    const extRoute = fs.readFileSync(
      path.resolve(__dirname, "../routes/option-writing-playground.tsx"),
      "utf8"
    )
    const sharedWritingPlayground = fs.readFileSync(
      path.resolve(
        __dirname,
        "../../../packages/ui/src/components/Option/WritingPlayground/index.tsx"
      ),
      "utf8"
    )

    expect(webRoute).toContain('data-testid="writing-playground-route-shell"')
    expect(extRoute).toContain('PageShell className="py-6" maxWidthClassName="max-w-7xl"')
    expect(webRoute).toContain("<WritingPlayground />")
    expect(extRoute).toContain("<WritingPlayground />")
    expect(sharedWritingPlayground).toContain("<WritingActionBar")
    expect(sharedWritingPlayground).toContain("<WritingRevisionQueue")
    expect(sharedWritingPlayground).toContain('data-testid="writing-revision-pending-count"')
    expect(sharedWritingPlayground).toContain('data-testid="writing-status-word-count"')
    expect(sharedWritingPlayground).toContain('data-testid="writing-status-selected-word-count"')
  })
})
