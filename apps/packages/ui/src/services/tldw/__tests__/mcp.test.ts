import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequestClient: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequestClient: (...args: unknown[]) => mocks.bgRequestClient(...args)
}))

import {
  fetchMcpModulesViaDiscovery,
  fetchMcpToolCatalogsViaDiscovery,
  fetchMcpToolsViaDiscovery
} from "../mcp"

const contentJsonResult = (json: Record<string, unknown>) => ({
  result: {
    content: [
      {
        type: "json",
        json
      }
    ]
  }
})

describe("mcp service client", () => {
  beforeEach(() => {
    mocks.bgRequestClient.mockReset()
  })

  it("normalizes discovery tools returned as MCP content JSON", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce(
      contentJsonResult({
        tools: [{ name: "media.search" }]
      })
    )

    await expect(fetchMcpToolsViaDiscovery()).resolves.toEqual([
      { name: "media.search" }
    ])
  })

  it("normalizes discovery catalogs returned as MCP content JSON", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce(
      contentJsonResult({
        catalogs: {
          global: [{ id: 1, name: "Research" }],
          user: [{ id: 2, name: "Personal" }]
        }
      })
    )

    await expect(fetchMcpToolCatalogsViaDiscovery()).resolves.toEqual([
      { id: 1, name: "Research" },
      { id: 2, name: "Personal" }
    ])
  })

  it("normalizes discovery modules returned as MCP content JSON", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce(
      contentJsonResult({
        modules: [{ module_id: "media" }]
      })
    )

    await expect(fetchMcpModulesViaDiscovery()).resolves.toEqual(["media"])
  })

  it("normalizes discovery tools returned as plain result payloads", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce({
      result: {
        tools: [{ name: "media.search" }]
      }
    })

    await expect(fetchMcpToolsViaDiscovery()).resolves.toEqual([
      { name: "media.search" }
    ])
  })

  it("normalizes discovery catalogs returned as plain result payloads", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce({
      result: {
        catalogs: {
          global: [{ id: 1, name: "Research" }],
          user: [{ id: 2, name: "Personal" }]
        }
      }
    })

    await expect(fetchMcpToolCatalogsViaDiscovery()).resolves.toEqual([
      { id: 1, name: "Research" },
      { id: 2, name: "Personal" }
    ])
  })

  it("normalizes discovery modules returned as plain result payloads", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce({
      result: {
        modules: [{ module_id: "media" }]
      }
    })

    await expect(fetchMcpModulesViaDiscovery()).resolves.toEqual(["media"])
  })
})
