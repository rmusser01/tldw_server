import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ChatSettings } from "../ChatSettings"
import { CHAT_BACKGROUND_IMAGE_MAX_BASE64_LENGTH } from "@/services/settings/ui-settings"

const {
  useChatSettingsMock,
  useStorageMock,
  useSettingMock,
  setChatBackgroundImageMock,
  setQuickChatStrictDocsOnlyMock,
  setQuickChatDocsNamespaceMock,
  setQuickChatDocsMediaIdsMock,
  setQuickChatWorkflowGuidesMock,
  notificationErrorMock,
  notificationSuccessMock,
  notificationInfoMock,
  toBase64Mock
} = vi.hoisted(() => ({
  useChatSettingsMock: vi.fn(),
  useStorageMock: vi.fn(),
  useSettingMock: vi.fn(),
  setChatBackgroundImageMock: vi.fn(),
  setQuickChatStrictDocsOnlyMock: vi.fn(),
  setQuickChatDocsNamespaceMock: vi.fn(),
  setQuickChatDocsMediaIdsMock: vi.fn(),
  setQuickChatWorkflowGuidesMock: vi.fn(),
  notificationErrorMock: vi.fn(),
  notificationSuccessMock: vi.fn(),
  notificationInfoMock: vi.fn(),
  toBase64Mock: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/hooks/useChatSettings", () => ({
  useChatSettings: useChatSettingsMock
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: useStorageMock
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: useSettingMock
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: notificationErrorMock,
    success: notificationSuccessMock,
    info: notificationInfoMock
  })
}))

vi.mock("@/libs/to-base64", () => ({
  toBase64: toBase64Mock
}))

vi.mock("../DiscoSkillsSettings", () => ({
  DiscoSkillsSettings: () => <div>Disco Skills</div>
}))

const buildChatSettingsState = () => ({
  copilotResumeLastChat: false,
  setCopilotResumeLastChat: vi.fn(),
  defaultChatWithWebsite: false,
  setDefaultChatWithWebsite: vi.fn(),
  webUIResumeLastChat: false,
  setWebUIResumeLastChat: vi.fn(),
  hideCurrentChatModelSettings: false,
  setHideCurrentChatModelSettings: vi.fn(),
  hideQuickChatHelper: false,
  setHideQuickChatHelper: vi.fn(),
  restoreLastChatModel: false,
  setRestoreLastChatModel: vi.fn(),
  generateTitle: false,
  setGenerateTitle: vi.fn(),
  checkWideMode: false,
  setCheckWideMode: vi.fn(),
  stickyChatInput: false,
  setStickyChatInput: vi.fn(),
  menuDensity: "comfortable" as const,
  setMenuDensity: vi.fn(),
  openReasoning: false,
  setOpenReasoning: vi.fn(),
  userChatBubble: true,
  setUserChatBubble: vi.fn(),
  autoCopyResponseToClipboard: false,
  setAutoCopyResponseToClipboard: vi.fn(),
  useMarkdownForUserMessage: false,
  setUseMarkdownForUserMessage: vi.fn(),
  chatRichTextMode: "safe_markdown" as const,
  setChatRichTextMode: vi.fn(),
  chatRichTextStylePreset: "default" as const,
  setChatRichTextStylePreset: vi.fn(),
  chatRichItalicColor: "default" as const,
  setChatRichItalicColor: vi.fn(),
  chatRichItalicFont: "default" as const,
  setChatRichItalicFont: vi.fn(),
  chatRichBoldColor: "default" as const,
  setChatRichBoldColor: vi.fn(),
  chatRichBoldFont: "default" as const,
  setChatRichBoldFont: vi.fn(),
  chatRichQuoteTextColor: "default" as const,
  setChatRichQuoteTextColor: vi.fn(),
  chatRichQuoteFont: "default" as const,
  setChatRichQuoteFont: vi.fn(),
  chatRichQuoteBorderColor: "default" as const,
  setChatRichQuoteBorderColor: vi.fn(),
  chatRichQuoteBackgroundColor: "default" as const,
  setChatRichQuoteBackgroundColor: vi.fn(),
  copyAsFormattedText: false,
  setCopyAsFormattedText: vi.fn(),
  allowExternalImages: false,
  setAllowExternalImages: vi.fn(),
  tabMentionsEnabled: false,
  setTabMentionsEnabled: vi.fn(),
  pasteLargeTextAsFile: false,
  setPasteLargeTextAsFile: vi.fn(),
  sidepanelTemporaryChat: false,
  setSidepanelTemporaryChat: vi.fn(),
  removeReasoningTagFromCopy: false,
  setRemoveReasoningTagFromCopy: vi.fn(),
  promptSearchIncludeServer: false,
  setPromptSearchIncludeServer: vi.fn(),
  userTextColor: "default" as const,
  setUserTextColor: vi.fn(),
  assistantTextColor: "default" as const,
  setAssistantTextColor: vi.fn(),
  userTextFont: "default" as const,
  setUserTextFont: vi.fn(),
  assistantTextFont: "default" as const,
  setAssistantTextFont: vi.fn(),
  userTextSize: "md" as const,
  setUserTextSize: vi.fn(),
  assistantTextSize: "md" as const,
  setAssistantTextSize: vi.fn()
})

describe("ChatSettings background image controls", () => {
  beforeEach(() => {
    useChatSettingsMock.mockReturnValue(buildChatSettingsState())
    useStorageMock.mockImplementation((key: string, defaultValue: unknown) => {
      if (key === "quickChatStrictDocsOnly") {
        return [true, setQuickChatStrictDocsOnlyMock, { isLoading: false }] as const
      }
      if (key === "quickChatDocsIndexNamespace") {
        return ["project_docs", setQuickChatDocsNamespaceMock, { isLoading: false }] as const
      }
      if (key === "quickChatDocsProjectMediaIds") {
        return [[], setQuickChatDocsMediaIdsMock, { isLoading: false }] as const
      }
      if (key === "quickChatWorkflowGuidesV1") {
        return [
          [
            {
              id: "guide-1",
              title: "Guide 1",
              question: "Where do I start?",
              answer: "Start in Research Studio.",
              route: "/research-studio",
              routeLabel: "Research Studio",
              tags: ["workflow"]
            }
          ],
          setQuickChatWorkflowGuidesMock,
          { isLoading: false }
        ] as const
      }
      return [defaultValue, vi.fn(), { isLoading: false }] as const
    })
    useSettingMock.mockReturnValue([
      undefined,
      setChatBackgroundImageMock,
      { isLoading: false }
    ])
    setChatBackgroundImageMock.mockReset()
    setChatBackgroundImageMock.mockResolvedValue(undefined)
    setQuickChatStrictDocsOnlyMock.mockReset()
    setQuickChatDocsNamespaceMock.mockReset()
    setQuickChatDocsMediaIdsMock.mockReset()
    setQuickChatWorkflowGuidesMock.mockReset()
    notificationErrorMock.mockReset()
    notificationSuccessMock.mockReset()
    notificationInfoMock.mockReset()
    toBase64Mock.mockReset()
    toBase64Mock.mockResolvedValue("data:image/png;base64,abc123")
  })

  it("rejects non-image uploads", async () => {
    const { container } = render(<ChatSettings />)
    const fileInput = container.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement | null
    expect(fileInput).not.toBeNull()

    const invalidFile = new File(["not image"], "notes.txt", {
      type: "text/plain"
    })
    fireEvent.change(fileInput as HTMLInputElement, {
      target: { files: [invalidFile] }
    })

    await waitFor(() => {
      expect(notificationErrorMock).toHaveBeenCalled()
    })
    expect(toBase64Mock).not.toHaveBeenCalled()
    expect(setChatBackgroundImageMock).not.toHaveBeenCalled()
  })

  it("uploads valid images and persists the setting", async () => {
    const { container } = render(<ChatSettings />)
    const fileInput = container.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement | null
    expect(fileInput).not.toBeNull()

    const imageFile = new File(["fake image"], "chat-bg.png", {
      type: "image/png"
    })
    fireEvent.change(fileInput as HTMLInputElement, {
      target: { files: [imageFile] }
    })

    await waitFor(() => {
      expect(toBase64Mock).toHaveBeenCalledWith(imageFile)
      expect(setChatBackgroundImageMock).toHaveBeenCalledWith(
        "data:image/png;base64,abc123"
      )
    })
  })

  it("accepts background images under the new 15 MB cap", async () => {
    toBase64Mock.mockResolvedValue(
      `data:image/png;base64,${"a".repeat(CHAT_BACKGROUND_IMAGE_MAX_BASE64_LENGTH - 32)}`
    )

    const { container } = render(<ChatSettings />)
    const fileInput = container.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement | null
    expect(fileInput).not.toBeNull()

    const imageFile = new File(["fake image"], "chat-bg-large.png", {
      type: "image/png"
    })
    fireEvent.change(fileInput as HTMLInputElement, {
      target: { files: [imageFile] }
    })

    await waitFor(() => {
      expect(toBase64Mock).toHaveBeenCalledWith(imageFile)
      expect(setChatBackgroundImageMock).toHaveBeenCalledWith(
        expect.stringContaining("data:image/png;base64,")
      )
    })
    expect(notificationErrorMock).not.toHaveBeenCalled()
  })

  it("rejects background images above the 15 MB cap", async () => {
    toBase64Mock.mockResolvedValue(
      `data:image/png;base64,${"a".repeat(CHAT_BACKGROUND_IMAGE_MAX_BASE64_LENGTH + 1)}`
    )

    const { container } = render(<ChatSettings />)
    const fileInput = container.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement | null
    expect(fileInput).not.toBeNull()

    const imageFile = new File(["fake image"], "chat-bg-too-large.png", {
      type: "image/png"
    })
    fireEvent.change(fileInput as HTMLInputElement, {
      target: { files: [imageFile] }
    })

    await waitFor(() => {
      expect(notificationErrorMock).toHaveBeenCalled()
    })
    expect(setChatBackgroundImageMock).not.toHaveBeenCalled()
  })

  it("shows and handles remove button when a background is set", async () => {
    useSettingMock.mockReturnValue([
      "data:image/png;base64,existing",
      setChatBackgroundImageMock,
      { isLoading: false }
    ])

    render(<ChatSettings />)
    fireEvent.click(
      screen.getByRole("button", { name: "Remove background image" })
    )

    await waitFor(() => {
      expect(setChatBackgroundImageMock).toHaveBeenCalledWith(undefined)
    })
  })

  it("renders rich text mode preview cards", () => {
    render(<ChatSettings />)

    expect(screen.getByText("Rendering preview")).toBeInTheDocument()
    expect(screen.getByText("Safe Markdown")).toBeInTheDocument()
    expect(screen.getAllByText("SillyTavern-compatible").length).toBeGreaterThan(0)
    expect(screen.getByText("Rich text element styles")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Reset rich text styles" })).toBeInTheDocument()
  })

  it("updates quick chat docs scope settings", () => {
    render(<ChatSettings />)

    fireEvent.click(
      screen.getByRole("switch", {
        name: "Restrict Quick Chat Docs Q&A to project documentation"
      })
    )
    expect(setQuickChatStrictDocsOnlyMock).toHaveBeenCalledWith(false)

    fireEvent.change(
      screen.getByLabelText("Quick Chat project docs namespace"),
      { target: { value: "official_docs" } }
    )
    expect(setQuickChatDocsNamespaceMock).toHaveBeenCalledWith("official_docs")

    fireEvent.change(
      screen.getByLabelText("Quick Chat project docs media IDs"),
      { target: { value: "101, 205, 309" } }
    )
    expect(setQuickChatDocsMediaIdsMock).toHaveBeenCalledWith("101, 205, 309")
  })

  it("saves quick chat workflow card edits", () => {
    render(<ChatSettings />)

    const workflowEditor = screen.getByLabelText(
      "Quick Chat workflow cards JSON"
    )
    fireEvent.change(workflowEditor, {
      target: {
        value: JSON.stringify(
          [
            {
              id: "custom-setup-guide",
              title: "Custom setup guide",
              question: "How do I configure setup?",
              answer: "Open Health & Diagnostics first.",
              route: "/settings/health",
              routeLabel: "Health & Diagnostics",
              tags: ["setup", "diagnostics"]
            }
          ],
          null,
          2
        )
      }
    })

    fireEvent.click(screen.getByRole("button", { name: "Save workflow cards" }))

    expect(setQuickChatWorkflowGuidesMock).toHaveBeenCalledWith([
      {
        id: "custom-setup-guide",
        title: "Custom setup guide",
        question: "How do I configure setup?",
        answer: "Open Health & Diagnostics first.",
        route: "/settings/health",
        routeLabel: "Health & Diagnostics",
        tags: ["setup", "diagnostics"]
      }
    ])
    expect(notificationSuccessMock).toHaveBeenCalled()
  })
})
