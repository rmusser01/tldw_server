import React from "react"
import { App, ConfigProvider } from "antd"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { PlaygroundMessage } from "../Message"

const feedback = vi.hoisted(() => ({ submitSourceThumb: vi.fn() }))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback, vi.fn()]
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback || key })
}))
vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({ cancel: vi.fn(), speak: vi.fn(), isSpeaking: false })
}))
vi.mock("@/hooks/useFeedback", () => ({
  useFeedback: () => ({
    thumb: null, detail: "", sourceFeedback: {}, canSubmit: true,
    isSubmitting: false, showThanks: false, submitThumb: vi.fn(),
    submitDetail: vi.fn(), submitSourceThumb: feedback.submitSourceThumb
  })
}))
vi.mock("@/hooks/useImplicitFeedback", () => ({
  useImplicitFeedback: () => ({
    trackCopy: vi.fn(), trackSourcesExpanded: vi.fn(), trackSourceClick: vi.fn(),
    trackCitationUsed: vi.fn(), trackDwellTime: vi.fn()
  })
}))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasFeedbackExplicit: true } })
}))
vi.mock("@/hooks/useTldwAudioStatus", () => ({
  useTldwAudioStatus: () => ({ healthState: "ready", voicesAvailable: false })
}))
vi.mock("@/hooks/useDiscoSkills", () => ({
  useDiscoSkills: () => ({ enabled: false, stats: null, triggerProbabilityBase: 0, persistComments: false })
}))
vi.mock("../MessageActionsBar", () => ({ MessageActionsBar: () => null }))
vi.mock("@/components/Sidepanel/Chat/FeedbackModal", () => ({ FeedbackModal: () => null }))

const source = {
  name: "Operator Notes",
  content: "The verified evidence is VIOLET7429.",
  url: "https://example.com/operator-notes",
  mode: "rag",
  type: "txt",
  score: 0.91,
  metadata: {
    chunk_id: "chunk-1", retrieval_strategy: "hybrid",
    source_type: "media_db", reason: "High lexical overlap",
    page: 2, loc: { lines: { from: 3, to: 8 } }
  }
}

const baseProps: React.ComponentProps<typeof PlaygroundMessage> = {
  message: "Grounded answer [1]", isBot: true, role: "assistant", name: "Assistant",
  currentMessageIndex: 0, totalMessages: 1, onRegenerate: vi.fn(),
  onEditFormSubmit: vi.fn(), isProcessing: false, isStreaming: false,
  conversationInstanceId: "source-actions-test", sources: [source]
}

const messageView = (props: Partial<React.ComponentProps<typeof PlaygroundMessage>> = {}) => (
  <ConfigProvider theme={{ token: { motion: false } }}>
    <App><PlaygroundMessage {...baseProps} {...props} /></App>
  </ConfigProvider>
)

describe("PlaygroundMessage source workflow visibility", () => {
  beforeEach(() => vi.clearAllMocks())

  it("hides unsupported workflows while retaining real citation evidence, safe links and feedback", async () => {
    const user = userEvent.setup()
    render(messageView({ hideSourceActions: true }))
    await user.click(screen.getByText("citations"))
    await user.click(await screen.findByText("Operator Notes"))

    expect(screen.getByText("Operator Notes").closest("details")).toHaveAttribute("open")
    expect(screen.getByText(source.content)).toBeVisible()
    expect(screen.getByText("Why this source")).toBeVisible()
    expect(screen.getByText("91%")).toBeVisible()
    expect(screen.getByText("chunk-1")).toBeVisible()
    expect(screen.getByText("High lexical overlap")).toBeVisible()
    expect(screen.getByText("Page 2")).toBeVisible()
    expect(screen.getByText("Line 3 - 8")).toBeVisible()
    const link = screen.getByRole("link", { name: "Open source" })
    expect(link).toHaveAttribute("href", "https://example.com/operator-notes")
    expect(link).toHaveAttribute("target", "_blank")
    expect(link).toHaveAttribute("rel", "noopener noreferrer")
    expect(screen.queryAllByRole("button", { name: /Ask with/ })).toHaveLength(0)
    expect(screen.queryAllByRole("button", { name: "Open Search & Context" })).toHaveLength(0)

    await user.click(screen.getByRole("button", { name: "Helpful source" }))
    expect(feedback.submitSourceThumb).toHaveBeenCalledWith({
      sourceKey: expect.any(String), source, thumb: "up"
    })
    expect(screen.getByRole("button", { name: "Unhelpful source" })).toBeEnabled()
  })

  it("keeps unsafe source URLs non-navigable when workflows are hidden", async () => {
    const user = userEvent.setup()
    render(messageView({
      hideSourceActions: true,
      sources: [{ ...source, url: "java\tscript:alert(1)" }]
    }))
    await user.click(screen.getByText("citations"))
    await user.click(await screen.findByText("Operator Notes"))
    expect(screen.getByText(source.content)).toBeVisible()
    expect(screen.queryByRole("link", { name: "Open source" })).not.toBeInTheDocument()
  })

  it.each([undefined, false])("retains legacy source commands and event payloads with hideSourceActions=%s", async (hideSourceActions) => {
    const user = userEvent.setup()
    const composerEvent = vi.fn()
    const knowledgeEvent = vi.fn()
    const focusEvent = vi.fn()
    window.addEventListener("tldw:set-composer-message", composerEvent)
    window.addEventListener("tldw:open-knowledge-panel", knowledgeEvent)
    window.addEventListener("tldw:focus-composer", focusEvent)
    try {
      render(messageView({ hideSourceActions }))
      await user.click(screen.getByText("citations"))
      await user.click(await screen.findByText("Operator Notes"))
      await user.click(screen.getByRole("button", { name: "Ask with these sources" }))
      await user.click(screen.getByRole("button", { name: "Ask with this source" }))
      expect(composerEvent.mock.calls.map(([event]) => event.detail)).toEqual([
        { message: "Use these sources in your next answer:\n[1] Operator Notes\nQuestion:" },
        { message: "Use these sources in your next answer:\n[1] Operator Notes\nQuestion:" }
      ])
      expect(focusEvent).toHaveBeenCalledTimes(2)
      const openButtons = screen.getAllByRole("button", { name: "Open Search & Context" })
      expect(openButtons).toHaveLength(2)
      for (const button of openButtons) await user.click(button)
      expect(knowledgeEvent.mock.calls.map(([event]) => event.detail)).toEqual([
        { tab: "search" }, { tab: "search" }, { tab: "search" }, { tab: "search" }
      ])
    } finally {
      window.removeEventListener("tldw:set-composer-message", composerEvent)
      window.removeEventListener("tldw:open-knowledge-panel", knowledgeEvent)
      window.removeEventListener("tldw:focus-composer", focusEvent)
    }
  })

  it("withdraws already-expanded workflow controls when the surface hides source actions", async () => {
    const user = userEvent.setup()
    const view = render(messageView())
    await user.click(screen.getByText("citations"))
    await user.click(await screen.findByText("Operator Notes"))
    expect(screen.getByRole("button", { name: "Ask with this source" })).toBeVisible()
    view.rerender(messageView({ hideSourceActions: true }))
    expect(screen.getByText(source.content)).toBeVisible()
    expect(screen.getByRole("link", { name: "Open source" })).toBeVisible()
    expect(screen.queryAllByRole("button", { name: /Ask with/ })).toHaveLength(0)
    expect(screen.queryAllByRole("button", { name: "Open Search & Context" })).toHaveLength(0)
  })
})
