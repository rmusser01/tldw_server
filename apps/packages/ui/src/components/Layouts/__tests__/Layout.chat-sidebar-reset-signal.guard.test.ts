import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testModulePath = import.meta.url.startsWith("file:")
  ? fileURLToPath(import.meta.url)
  : import.meta.url
const layoutSourcePath = resolve(dirname(testModulePath), "../Layout.tsx")

describe("Layout chat sidebar reset signal", () => {
  it("passes openResetKey to desktop and mobile ChatSidebar mounts", () => {
    const source = readFileSync(layoutSourcePath, "utf8")

    expect(source).toContain("chatSidebarOpenResetKey")
    expect(source.match(/openResetKey=\{chatSidebarOpenResetKey\}/g)).toHaveLength(2)
  })

  it("increments the reset key only on explicit open paths", () => {
    const source = readFileSync(layoutSourcePath, "utf8")

    expect(source).toContain("signalChatSidebarOpen")
    expect(source).toContain("setChatSidebarOpenResetKey((value) => value + 1)")
    expect(source).toContain("if (!sidebarOpen) signalChatSidebarOpen()")
    expect(source).toContain("if (chatSidebarCollapsed) signalChatSidebarOpen()")
    expect(source).toContain('window.addEventListener("tldw:open-chat-sidebar", handler)')
  })
})
