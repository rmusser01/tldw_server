import { afterEach, describe, expect, it, vi } from "vitest"
import { GitHubProvider } from "../GitHubProvider"

describe("GitHubProvider", () => {
  it("parses owner/repo from github URL", () => {
    const provider = new GitHubProvider()
    const parsed = provider.parseUrl("https://github.com/facebook/react")
    expect(parsed.isValid).toBe(true)
    expect(parsed.owner).toBe("facebook")
    expect(parsed.repo).toBe("react")
  })
})

afterEach(() => vi.unstubAllGlobals())

describe("GitHubProvider complete export", () => {
  it("decodes UTF8 content exactly", async () => {
    const text = "Selene — café 星\n"
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({
              content: Buffer.from(text).toString("base64"),
              encoding: "base64"
            })
          )
        )
    )
    const result = await new GitHubProvider().fetchFile({
      path: "README.md",
      type: "blob",
      url: "https://api.github.com/repos/test/public/git/blobs/123"
    })
    expect(result.text).toBe(text)
  })

  it("refuses a truncated tree rather than silently omitting files", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({
              truncated: true,
              tree: [{ path: "README.md", type: "blob" }]
            })
          )
        )
    )
    await expect(
      new GitHubProvider().fetchTree(
        "https://github.com/test/public/tree/abc123"
      )
    ).rejects.toThrow(/truncat|incomplete/i)
  })

  it("refuses symbolic links in the repository tree", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(
            JSON.stringify({
              tree: [{ path: "link.md", type: "blob", mode: "120000" }]
            })
          )
        )
    )
    await expect(
      new GitHubProvider().fetchTree("https://github.com/test/public")
    ).rejects.toThrow(/symlink|symbolic/i)
  })
})
