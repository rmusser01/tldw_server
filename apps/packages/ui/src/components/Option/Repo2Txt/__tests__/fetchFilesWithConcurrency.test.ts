import { describe, expect, it, vi } from "vitest"
import type { RepoFileContent, RepoTreeNode } from "../providers/types"
import { fetchFilesWithConcurrency } from "../fetchFilesWithConcurrency"

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

describe("fetchFilesWithConcurrency", () => {
  it("limits concurrent fetches", async () => {
    const nodes: RepoTreeNode[] = Array.from({ length: 7 }, (_, index) => ({
      path: `src/file-${index}.ts`,
      type: "blob",
      url: `src/file-${index}.ts`
    }))

    let inFlight = 0
    let maxInFlight = 0

    const provider = {
      fetchFile: async (node: RepoTreeNode): Promise<RepoFileContent> => {
        inFlight += 1
        maxInFlight = Math.max(maxInFlight, inFlight)
        await sleep(5)
        inFlight -= 1
        return {
          path: node.path,
          text: `content-${node.path}`,
          lineCount: 1
        }
      }
    }

    const progress: number[] = []
    const result = await fetchFilesWithConcurrency({
      nodes,
      provider,
      limit: 3,
      onProgress: (completed) => progress.push(completed)
    })

    expect(result).toHaveLength(nodes.length)
    expect(maxInFlight).toBeLessThanOrEqual(3)
    expect(progress.at(-1)).toBe(nodes.length)
  })
})

describe("fetchFilesWithConcurrency export boundaries", () => {
  it("does no reads after pre-cancellation", async () => {
    const controller = new AbortController()
    controller.abort()
    const provider = { fetchFile: vi.fn() }
    await expect(
      fetchFilesWithConcurrency({
        nodes: [{ path: "README.md", type: "blob" }],
        provider,
        signal: controller.signal
      })
    ).rejects.toMatchObject({ name: "AbortError" })
    expect(provider.fetchFile).not.toHaveBeenCalled()
  })
  it("rejects a total export exceeding 10 MiB", async () => {
    const nodes: RepoTreeNode[] = Array.from({ length: 11 }, (_, i) => ({
      path: `file${i}.md`,
      type: "blob"
    }))
    const provider = {
      fetchFile: async (node: RepoTreeNode) => ({
        path: node.path,
        text: "x".repeat(1024 * 1024),
        lineCount: 1
      })
    }
    await expect(
      fetchFilesWithConcurrency({ nodes, provider })
    ).rejects.toThrow(/10 MiB/)
  })
})
