import fs from "node:fs"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it, vi } from "vitest"

import { resolveExtensionId } from "./extension-id"

const TEST_MANIFEST_KEY =
  "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAjI1q+ZCGeQEsFkXz8Jcx9BHxpWcxr4egilGW2LKpyDcxbd+2id2k0WtauiWSS+eBfvJWRnonnIjZQ/6jkNbN41z+G6Wp5HzHJaGHB609GO4LWW5kVkPo0h+KkSSEVjoXTyRQZO3ViwDbne3gqHVJmnKGWV+Tz6X2se3GwCah3I0AG2290/E4aweSV6OG/SRD15MCiDTImSCNa7WXhMQtqN61o+b8MGr3t5eN3E2UCKMFYAFH017EuRQ46vn8q29O7ATaEwHnB0U/7g9zyi3OKhCU5bI9XhZNoRH/iZqOajz5vVu4Pbq6Wq0Vu2Y1nHIjOQi4XADuUrd4ZFyQWkDFcwIDAQAB"
const TEST_EXTENSION_ID = "biemafbfmdmpamjhjbbnjimmgkkgefca"

let tempRoots: string[] = []

afterEach(() => {
  for (const tempRoot of tempRoots) {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
  tempRoots = []
  vi.restoreAllMocks()
})

const makeContextWithoutExtensionTargets = () => {
  const page = {}
  const send = vi.fn().mockResolvedValue({ targetInfos: [] })

  return {
    backgroundPages: vi.fn(() => []),
    serviceWorkers: vi.fn(() => []),
    pages: vi.fn(() => []),
    newPage: vi.fn().mockResolvedValue(page),
    newCDPSession: vi.fn().mockResolvedValue({ send })
  } as any
}

describe("resolveExtensionId", () => {
  it("cleans up a fallback probe page and CDP session after finding an extension target", async () => {
    const probePage = { close: vi.fn().mockResolvedValue(undefined) }
    const detach = vi.fn().mockResolvedValue(undefined)
    const context = {
      backgroundPages: vi.fn(() => []),
      serviceWorkers: vi.fn(() => []),
      pages: vi.fn(() => []),
      newPage: vi.fn().mockResolvedValue(probePage),
      newCDPSession: vi.fn().mockResolvedValue({
        detach,
        send: vi.fn().mockResolvedValue({
          targetInfos: [
            {
              type: "service_worker",
              url: `chrome-extension://${TEST_EXTENSION_ID}/background.js`,
            },
          ],
        }),
      }),
    } as any

    await expect(resolveExtensionId(context)).resolves.toBe(TEST_EXTENSION_ID)

    expect(detach).toHaveBeenCalledTimes(1)
    expect(probePage.close).toHaveBeenCalledTimes(1)
  })

  it("falls back to a staged manifest key when no extension target is active", async () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-extension-id-"))
    tempRoots.push(tempRoot)
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(extensionDir, { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({
        manifest_version: 3,
        name: "Test Extension",
        version: "1.0.0",
        key: TEST_MANIFEST_KEY
      }),
      "utf8"
    )

    await expect(
      resolveExtensionId(makeContextWithoutExtensionTargets(), {
        extensionPath: extensionDir
      })
    ).resolves.toBe(TEST_EXTENSION_ID)
  })
})
