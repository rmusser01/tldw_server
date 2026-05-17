import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

describe("Layout chat sidebar reset signal", () => {
  it("passes openResetKey to desktop and mobile ChatSidebar mounts", () => {
    const source = readFileSync("src/components/Layouts/Layout.tsx", "utf8")

    expect(source).toContain("chatSidebarOpenResetKey")
    expect(source.match(/openResetKey=\{chatSidebarOpenResetKey\}/g)).toHaveLength(2)
  })

  it("increments the reset key only on explicit open paths", () => {
    const source = readFileSync("src/components/Layouts/Layout.tsx", "utf8")

    expect(source).toContain("signalChatSidebarOpen")
    expect(source).toContain("setChatSidebarOpenResetKey((value) => value + 1)")
    expect(source).toContain("if (!sidebarOpen) signalChatSidebarOpen()")
    expect(source).toContain("if (chatSidebarCollapsed) signalChatSidebarOpen()")
    expect(source).toContain('window.addEventListener("tldw:open-chat-sidebar", handler)')
  })
})
