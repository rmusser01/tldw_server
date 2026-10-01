import React from "react"
import { App, ConfigProvider } from "antd"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { TFunction } from "i18next"
import { PlaygroundMessage } from "../Message"
import { MessageContent, type MessageContentProps } from "../MessageContent"

const settings = vi.hoisted(() => new Map<string, unknown>())
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) => [settings.get(key) ?? fallback, vi.fn()]
}))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, fallback?: string) => fallback || key }) }))
vi.mock("@/hooks/useTTS", () => ({ useTTS: () => ({ cancel: vi.fn(), speak: vi.fn(), isSpeaking: false }) }))
vi.mock("@/hooks/useFeedback", () => ({ useFeedback: () => ({ thumb: null, detail: "", sourceFeedback: {}, canSubmit: false, isSubmitting: false, showThanks: false }) }))
vi.mock("@/hooks/useImplicitFeedback", () => ({ useImplicitFeedback: () => ({ trackDwellTime: vi.fn() }) }))
vi.mock("@/hooks/useServerCapabilities", () => ({ useServerCapabilities: () => ({ capabilities: {} }) }))
vi.mock("@/hooks/useTldwAudioStatus", () => ({ useTldwAudioStatus: () => ({ healthState: "ready", voicesAvailable: false }) }))
vi.mock("@/hooks/useDiscoSkills", () => ({ useDiscoSkills: () => ({ enabled: false, stats: null, triggerProbabilityBase: 0, persistComments: false }) }))
vi.mock("../MessageActionsBar", () => ({ MessageActionsBar: () => null }))
vi.mock("@/components/Sidepanel/Chat/FeedbackModal", () => ({ FeedbackModal: () => null }))

const messageProps: React.ComponentProps<typeof PlaygroundMessage> = {
  message: "# Answer", isBot: true, role: "assistant", name: "Assistant", currentMessageIndex: 0, totalMessages: 1,
  onRegenerate: vi.fn(), onContinue: vi.fn(), onEditFormSubmit: vi.fn(), isProcessing: false, isStreaming: false,
  conversationInstanceId: "heading-test"
}
const contentProps: MessageContentProps = {
  t: ((key: string, fallback?: string) => fallback || key) as TFunction,
  message: "# Answer", isBot: true, isStreaming: false, editMode: false, onEditFormSubmit: vi.fn(), onCloseEdit: vi.fn(),
  errorPayload: null, shouldRenderStreamingPlainText: false, renderGreetingMarkdown: false, assistantTextClass: "", chatTextClass: "",
  errorRecoveryActions: [], showInlineImageActions: false, canRegenerateImage: false, imageGenerationMetadata: null
}

describe.each(["safe_markdown", "st_compat"] as const)("Transcript heading propagation in %s", mode => {
  beforeEach(() => {
    settings.clear()
    settings.set("chatRichTextMode", mode)
  })

  for (const component of ["Message", "MessageContent"] as const) {
    const view = (message: string, headingOffset?: number, greeting = false) => render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <App>
          <h1>Chat Workspace</h1>
          {component === "Message"
            ? <PlaygroundMessage {...messageProps} message={message} headingOffset={headingOffset} message_type={greeting ? "greeting" : undefined} openReasoning />
            : <MessageContent {...contentProps} message={message} headingOffset={headingOffset} renderGreetingMarkdown={greeting} openReasoning />}
        </App>
      </ConfigProvider>
    )

    it(`${component} keeps default transcript H1 elsewhere`, async () => {
      view("# Answer")
      await screen.findByRole("heading", { level: 1, name: "Answer" })
      expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(2)
    })

    it(`${component} reserves H1 for the workspace page on answers`, async () => {
      view("# Answer", 1)
      await screen.findByRole("heading", { level: 2, name: "Answer" })
      expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1)
    })

    it(`${component} offsets greeting headings`, async () => {
      view("# Welcome", 1, true)
      await screen.findByRole("heading", { level: 2, name: "Welcome" })
      expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1)
    })

    it(`${component} offsets expanded reasoning as well as the final answer`, async () => {
      view("<think>\n# Reasoning\n</think>\n# Answer", 1)
      await screen.findByRole("heading", { level: 2, name: "Answer" })
      await screen.findByRole("heading", { level: 2, name: "Reasoning" })
      expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1)
    })
  }
})
