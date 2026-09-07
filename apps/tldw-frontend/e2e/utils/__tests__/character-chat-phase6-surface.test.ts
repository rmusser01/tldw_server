import { describe, expect, it } from "vitest"

import {
  revealCharacterChatSessions,
  type CharacterChatSessionSurface,
} from "../character-chat-phase6-surface"

class CharacterChatSessionSurfaceHarness
  implements CharacterChatSessionSurface
{
  readonly actions: string[] = []

  constructor(
    private layoutMode: "cockpit" | "focus",
    private readonly viewportWidth: number,
    private sessionVisible = false,
    private contextRailVisible = false,
  ) {}

  async isSessionVisible(): Promise<boolean> {
    return this.sessionVisible
  }

  async isFocusMode(): Promise<boolean> {
    return this.layoutMode === "focus"
  }

  async exitFocusMode(): Promise<void> {
    this.actions.push("exit-focus-mode")
    this.layoutMode = "cockpit"
    this.sessionVisible = this.contextRailVisible
  }

  async restoreDesktopContextRail(): Promise<void> {
    this.actions.push("restore-desktop-context-rail")
    if (this.layoutMode === "cockpit") {
      this.contextRailVisible = true
      this.sessionVisible = true
    }
  }

  async selectCompactContextTab(): Promise<void> {
    this.actions.push("select-compact-context-tab")
    if (this.layoutMode === "cockpit") {
      this.contextRailVisible = true
      this.sessionVisible = true
    }
  }

  async getViewportWidth(): Promise<number> {
    return this.viewportWidth
  }

  getLayoutMode(): "cockpit" | "focus" {
    return this.layoutMode
  }
}

describe("Phase 6 Character Chat session reachability", () => {
  it("restores a collapsed desktop context rail without entering focus", async () => {
    const surface = new CharacterChatSessionSurfaceHarness("cockpit", 1440)

    await revealCharacterChatSessions(surface)

    expect(surface.getLayoutMode()).toBe("cockpit")
    expect(await surface.isSessionVisible()).toBe(true)
  })

  it("uses Exit focus before selecting the compact Context rail", async () => {
    const surface = new CharacterChatSessionSurfaceHarness("focus", 390)

    await revealCharacterChatSessions(surface)

    expect(surface.actions).toEqual([
      "exit-focus-mode",
      "select-compact-context-tab",
    ])
    expect(await surface.isSessionVisible()).toBe(true)
  })
})
