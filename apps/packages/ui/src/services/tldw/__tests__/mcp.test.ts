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

describe("mcp service client", () => {
  beforeEach(() => {
    mocks.bgRequestClient.mockReset()
  })

  it("normalizes discovery tool results returned as MCP content JSON", async () => {
    mocks.bgRequestClient
      .mockResolvedValueOnce({
        result: {
          content: [
            {
              type: "json",
              json: {
                tools: [{ name: "media.search" }]
              }
            }
          ]
        }
      })
      .mockResolvedValueOnce({
        result: {
          content: [
            {
              type: "json",
              json: {
                catalogs: {
                  global: [{ id: 1, name: "Research" }],
                  user: [{ id: 2, name: "Personal" }]
                }
              }
            }
          ]
        }
      })
      .mockResolvedValueOnce({
        result: {
          content: [
            {
              type: "json",
              json: {
                modules: [{ module_id: "media" }]
              }
            }
          ]
        }
      })

    await expect(fetchMcpToolsViaDiscovery()).resolves.toEqual([
      { name: "media.search" }
    ])
    await expect(fetchMcpToolCatalogsViaDiscovery()).resolves.toEqual([
      { id: 1, name: "Research" },
      { id: 2, name: "Personal" }
    ])
    await expect(fetchMcpModulesViaDiscovery()).resolves.toEqual(["media"])
  })
})
