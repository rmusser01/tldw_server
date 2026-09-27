import { readFile } from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { beforeAll, describe, expect, it } from "vitest"
import { normalizeSettingValue } from "@/services/settings/registry"
import {
  CHAT_CHARACTER_IMAGE_OPACITY_SETTING,
  CHAT_MESSAGE_OPACITY_SETTING,
  CHAT_WINDOW_OPACITY_SETTING
} from "@/services/settings/ui-settings"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const sourcePaths = {
  sidepanel: path.resolve(testDir, "../sidepanel-chat.tsx"),
  extensionSidepanel: path.resolve(
    testDir,
    "../../../../../tldw-frontend/extension/routes/sidepanel-chat.tsx"
  ),
  playground: path.resolve(
    testDir,
    "../../components/Option/Playground/Playground.tsx"
  ),
  cockpitShell: path.resolve(
    testDir,
    "../../components/Option/Playground/PlaygroundCockpitShell.tsx"
  ),
  uiSettings: path.resolve(testDir, "../../services/settings/ui-settings.ts"),
  chatSettings: path.resolve(
    testDir,
    "../../components/Option/Settings/ChatSettings.tsx"
  ),
  message: path.resolve(
    testDir,
    "../../components/Common/Playground/Message.tsx"
  ),
  playgroundUserMessage: path.resolve(
    testDir,
    "../../components/Common/Playground/PlaygroundUserMessage.tsx"
  ),
  chatOpacityCssVars: path.resolve(
    testDir,
    "../../services/settings/chat-opacity-css-vars.ts"
  )
} as const
type SourceKey = keyof typeof sourcePaths
const sources = {} as Record<SourceKey, string>

const readSource = async (key: SourceKey): Promise<string> => {
  try {
    return await readFile(sourcePaths[key], "utf8")
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(
      `Unable to read ${key} source at ${sourcePaths[key]}. This guard expects a full monorepo checkout. ${message}`
    )
  }
}

describe("chat background image translucency", () => {
  beforeAll(async () => {
    await Promise.all(
      (Object.keys(sourcePaths) as SourceKey[]).map(async (key) => {
        sources[key] = await readSource(key)
      })
    )
  })

  it.each([
    ["sidepanel chat", "sidepanel"],
    ["extension sidepanel chat", "extensionSidepanel"],
    ["playground chat", "playground"]
  ] as const)("%s keeps background images visible behind the chat wash", (_name, key) => {
    const source = sources[key]
    expect(source).toContain("chatWindowOpacity")
    expect(source).toContain("backgroundColor: `rgb(var(--color-bg) / ${")
    expect(source).not.toContain("style={{ opacity: 0.9, pointerEvents: \"none\" }}")
  })

  it("lets the playground cockpit shell reveal themed backgrounds", () => {
    const playgroundSource = sources.playground
    const cockpitShellSource = sources.cockpitShell

    expect(playgroundSource).toContain("themedBackdrop={Boolean(chatBackgroundImage)}")
    expect(playgroundSource).not.toContain("themedBackdropOpacity")
    expect(cockpitShellSource).toContain("themedBackdrop?: boolean")
    expect(cockpitShellSource).not.toContain("themedBackdropOpacity")
    expect(cockpitShellSource).not.toContain("backgroundColor: `rgb(var(--color-bg)")
    expect(cockpitShellSource).toContain("themedBackdrop ? \"bg-transparent\" : \"bg-bg\"")
  })

  it("keeps exactly one playground chat window wash layer", () => {
    const playgroundSource = sources.playground
    const playgroundWashMatches =
      playgroundSource.match(
        /backgroundColor: `rgb\(var\(--color-bg\) \/ \$\{chatWindowOpacityAlpha\}\)`/g
      ) ?? []

    expect(playgroundWashMatches).toHaveLength(1)
  })

  it("wires adjustable transparency settings into chat theming surfaces", () => {
    const chatSettingsSource = sources.chatSettings
    const extensionSidepanelSource = sources.extensionSidepanel
    const messageSource = sources.message
    const playgroundSource = sources.playground
    const playgroundUserMessageSource = sources.playgroundUserMessage
    const sidepanelSource = sources.sidepanel
    const uiSettingsSource = sources.uiSettings

    expect(uiSettingsSource).toContain("CHAT_WINDOW_OPACITY_SETTING")
    expect(uiSettingsSource).toContain("CHAT_MESSAGE_OPACITY_SETTING")
    expect(uiSettingsSource).toContain("CHAT_CHARACTER_IMAGE_OPACITY_SETTING")
    expect(uiSettingsSource).toContain("resolveOpacityAlpha")

    expect(chatSettingsSource).toContain("chatWindowOpacity")
    expect(chatSettingsSource).toContain("chatMessageOpacity")
    expect(chatSettingsSource).toContain("chatCharacterImageOpacity")

    expect(playgroundSource).toContain("chatWindowOpacity")
    expect(sidepanelSource).toContain("chatWindowOpacity")
    expect(extensionSidepanelSource).toContain("chatWindowOpacity")
    expect(playgroundSource).toContain("CHAT_MESSAGE_OPACITY_SETTING")
    expect(sidepanelSource).toContain("CHAT_MESSAGE_OPACITY_SETTING")
    expect(extensionSidepanelSource).toContain("CHAT_MESSAGE_OPACITY_SETTING")
    expect(playgroundSource).toContain("--chat-message-opacity")
    expect(sidepanelSource).toContain("--chat-message-opacity")
    expect(extensionSidepanelSource).toContain("--chat-message-opacity")
    expect(playgroundSource).toContain("--chat-character-image-opacity")
    expect(sidepanelSource).toContain("--chat-character-image-opacity")
    expect(extensionSidepanelSource).toContain("--chat-character-image-opacity")

    // Message components consume the adjustable CSS vars through the shared
    // chat-opacity-css-vars constants so the var names stay defined in one place.
    const chatOpacityCssVarsSource = sources.chatOpacityCssVars
    expect(chatOpacityCssVarsSource).toMatch(
      /CHAT_MESSAGE_OPACITY_ALPHA =\s*"var\(--chat-message-opacity/
    )
    expect(chatOpacityCssVarsSource).toMatch(
      /CHAT_CHARACTER_IMAGE_OPACITY_ALPHA =\s*"var\(--chat-character-image-opacity/
    )

    expect(messageSource).toContain("CHAT_MESSAGE_OPACITY_ALPHA")
    expect(messageSource).toContain("CHAT_CHARACTER_IMAGE_OPACITY_ALPHA")
    expect(messageSource).toContain(
      '} from "@/services/settings/chat-opacity-css-vars"'
    )
    expect(messageSource).toContain("--color-surface2")
    expect(messageSource).not.toContain("--color-surface-2")
    expect(messageSource).not.toContain("CHAT_MESSAGE_OPACITY_SETTING")
    expect(playgroundUserMessageSource).toContain("CHAT_MESSAGE_OPACITY_ALPHA")
    expect(playgroundUserMessageSource).toContain(
      'from "@/services/settings/chat-opacity-css-vars"'
    )
    expect(playgroundUserMessageSource).toContain("--color-surface2")
    expect(playgroundUserMessageSource).not.toContain("--color-surface-2")
    expect(playgroundUserMessageSource).not.toContain(
      "CHAT_MESSAGE_OPACITY_SETTING"
    )
  })

  it("clamps chat transparency settings to usable percentages", () => {
    expect(normalizeSettingValue(CHAT_WINDOW_OPACITY_SETTING, "115")).toBe(100)
    expect(normalizeSettingValue(CHAT_MESSAGE_OPACITY_SETTING, -15)).toBe(0)
    expect(normalizeSettingValue(CHAT_CHARACTER_IMAGE_OPACITY_SETTING, 48.6)).toBe(49)
  })
})
