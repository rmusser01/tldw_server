import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { SharedWorkspaceSafeMarkdown } from "../SharedResearchWorkspace/SharedWorkspaceSafeMarkdown"

describe("SharedWorkspaceSafeMarkdown", () => {
  it("keeps its static dependency tree isolated from executable and local mutation paths", () => {
    const rendererSource = readFileSync(
      resolve(
        process.cwd(),
        "src/components/Option/ResearchWorkspace/SharedResearchWorkspace/SharedWorkspaceSafeMarkdown.tsx"
      ),
      "utf8"
    )
    const imports = Array.from(
      rendererSource.matchAll(/from\s+["']([^"']+)["']/g),
      (match) => match[1]
    )

    expect(imports).toEqual(["react", "react-markdown", "remark-gfm"])
    expect(rendererSource).not.toMatch(
      /CodeBlock|ArtifactsStore|useArtifacts|useUiModeStore|zustand|Mermaid|iframe|sandbox|storage|allowExternalImages|dangerouslySetInnerHTML/i
    )
    for (const caller of ["SharedWorkspaceChatPane.tsx", "SharedWorkspacePreview.tsx"]) {
      const callerSource = readFileSync(
        resolve(
          process.cwd(),
          `src/components/Option/ResearchWorkspace/SharedResearchWorkspace/${caller}`
        ),
        "utf8"
      )
      expect(callerSource).toContain(
        'from "./SharedWorkspaceSafeMarkdown"'
      )
      expect(callerSource).not.toMatch(/components\/Common\/Markdown|CodeBlock/)
    }
  })

  it("renders fenced code as inert preformatted text without executable controls or external images", () => {
    render(
      <SharedWorkspaceSafeMarkdown
        content={[
          "```html",
          '<script>alert("no")</script>',
          "```",
          "![remote](https://example.test/remote.png)",
          "",
          "[unsafe](javascript:alert(1))",
          "",
          '<iframe src="https://example.test/embed"></iframe>',
        ].join("\n")}
      />
    )

    expect(screen.getByText('<script>alert("no")</script>')).toBeInTheDocument()
    expect(document.querySelector("pre > code")).toBeInTheDocument()
    expect(document.querySelector("button")).not.toBeInTheDocument()
    expect(document.querySelector("iframe")).not.toBeInTheDocument()
    expect(document.querySelector("img")).not.toBeInTheDocument()
    expect(screen.getByText("unsafe").closest("a")).toBeNull()
  })
})
