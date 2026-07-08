import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { normalizeSettingValue } from "@/services/settings/registry"
import {
  CHAT_CHARACTER_IMAGE_OPACITY_SETTING,
  CHAT_MESSAGE_OPACITY_SETTING,
  CHAT_WINDOW_OPACITY_SETTING
} from "@/services/settings/ui-settings"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const sidepanelSource = readFileSync(
  path.resolve(testDir, "../sidepanel-chat.tsx"),
  "utf8"
)
const extensionSidepanelSource = readFileSync(
  path.resolve(
    testDir,
    "../../../../../tldw-frontend/extension/routes/sidepanel-chat.tsx"
  ),
  "utf8"
)
const playgroundSource = readFileSync(
  path.resolve(
    testDir,
    "../../components/Option/Playground/Playground.tsx"
  ),
  "utf8"
)
const cockpitShellSource = readFileSync(
  path.resolve(
    testDir,
    "../../components/Option/Playground/PlaygroundCockpitShell.tsx"
  ),
  "utf8"
)
const uiSettingsSource = readFileSync(
  path.resolve(testDir, "../../services/settings/ui-settings.ts"),
  "utf8"
)
const chatSettingsSource = readFileSync(
  path.resolve(
    testDir,
    "../../components/Option/Settings/ChatSettings.tsx"
  ),
  "utf8"
)
const messageSource = readFileSync(
  path.resolve(testDir, "../../components/Common/Playground/Message.tsx"),
  "utf8"
)
const playgroundUserMessageSource = readFileSync(
  path.resolve(
    testDir,
    "../../components/Common/Playground/PlaygroundUserMessage.tsx"
  ),
  "utf8"
)

describe("chat background image translucency", () => {
  it.each([
    ["sidepanel chat", sidepanelSource],
    ["extension sidepanel chat", extensionSidepanelSource],
    ["playground chat", playgroundSource]
  ])("%s keeps background images visible behind the chat wash", (_name, source) => {
    expect(source).toContain("chatWindowOpacity")
    expect(source).toContain("backgroundColor: `rgb(var(--color-bg) / ${")
    expect(source).not.toContain("style={{ opacity: 0.9, pointerEvents: \"none\" }}")
  })

  it("lets the playground cockpit shell reveal themed backgrounds", () => {
    expect(playgroundSource).toContain("themedBackdrop={Boolean(chatBackgroundImage)}")
    expect(playgroundSource).toContain("themedBackdropOpacity={chatWindowOpacityAlpha}")
    expect(cockpitShellSource).toContain("themedBackdrop?: boolean")
    expect(cockpitShellSource).toContain("themedBackdropOpacity?: number")
    expect(cockpitShellSource).toContain("backgroundColor: `rgb(var(--color-bg) / ${clampedThemedBackdropOpacity})`")
    expect(cockpitShellSource).toContain("themedBackdrop ? \"bg-transparent\" : \"bg-bg\"")
  })

  it("wires adjustable transparency settings into chat theming surfaces", () => {
    expect(uiSettingsSource).toContain("CHAT_WINDOW_OPACITY_SETTING")
    expect(uiSettingsSource).toContain("CHAT_MESSAGE_OPACITY_SETTING")
    expect(uiSettingsSource).toContain("CHAT_CHARACTER_IMAGE_OPACITY_SETTING")

    expect(chatSettingsSource).toContain("chatWindowOpacity")
    expect(chatSettingsSource).toContain("chatMessageOpacity")
    expect(chatSettingsSource).toContain("chatCharacterImageOpacity")

    expect(playgroundSource).toContain("chatWindowOpacity")
    expect(sidepanelSource).toContain("chatWindowOpacity")
    expect(extensionSidepanelSource).toContain("chatWindowOpacity")
    expect(cockpitShellSource).toContain("themedBackdropOpacity")

    expect(messageSource).toContain("chatMessageOpacity")
    expect(messageSource).toContain("chatCharacterImageOpacity")
    expect(playgroundUserMessageSource).toContain("chatMessageOpacity")
  })

  it("clamps chat transparency settings to usable percentages", () => {
    expect(normalizeSettingValue(CHAT_WINDOW_OPACITY_SETTING, "115")).toBe(100)
    expect(normalizeSettingValue(CHAT_MESSAGE_OPACITY_SETTING, -15)).toBe(0)
    expect(normalizeSettingValue(CHAT_CHARACTER_IMAGE_OPACITY_SETTING, 48.6)).toBe(49)
  })
})
