import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("react-router-dom", () => ({
  useLocation: () => ({ pathname: "/audio-studio", search: "", hash: "", state: null }),
  useNavigate: () => vi.fn()
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <main>{children}</main>
  )
}))

vi.mock("@/components/Option/AudiobookStudio/ContentInput/TextEditor", () => ({
  TextEditor: () => <div>Paste or type your content</div>
}))

vi.mock("@/components/Option/AudiobookStudio/ChapterEditor/ChapterList", () => ({
  ChapterList: () => <div>Chapter List</div>
}))

vi.mock("@/components/Option/AudiobookStudio/Generation/GenerationPanel", () => ({
  GenerationPanel: () => <div>Voice Settings</div>
}))

vi.mock("@/components/Option/AudiobookStudio/Output/OutputPanel", () => ({
  OutputPanel: () => <div>Audiobook Player</div>
}))

import { AudioStudioPage } from "../AudioStudioPage"
import { useAudioStudioStore } from "@/store/audio-studio"

describe("AudioStudioPage", () => {
  beforeEach(() => {
    useAudioStudioStore.getState().resetAudioStudio()
  })

  it("renders all workflow labels as first-class choices", () => {
    render(<AudioStudioPage />)

    expect(screen.getByRole("heading", { name: "Audio Studio" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Narration/ })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Podcast/ })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Briefing/ })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /Music/ })).toBeInTheDocument()
  })

  it("shows imported audiobook controls in Narration without the old top heading", () => {
    useAudioStudioStore.getState().setActiveWorkflow("narration")

    render(<AudioStudioPage />)

    expect(screen.getByText("Paste or type your content")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("tab", { name: "Chapters" }))
    expect(screen.getByText("Chapter List")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("tab", { name: "Voice" }))
    expect(screen.getByText("Voice Settings")).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: "Audiobook Studio" })
    ).not.toBeInTheDocument()
  })

  it("surfaces Podcast and Briefing as production workflows", () => {
    useAudioStudioStore.getState().setActiveWorkflow("podcast")
    const { rerender } = render(<AudioStudioPage />)

    expect(screen.getByText("Podcast script")).toBeInTheDocument()
    expect(screen.getByText("Speakers")).toBeInTheDocument()

    useAudioStudioStore.getState().setActiveWorkflow("briefing")
    rerender(<AudioStudioPage />)

    expect(screen.getByText("Briefing outline")).toBeInTheDocument()
    expect(screen.getByText("Source notes")).toBeInTheDocument()
  })

  it("shows Music prompt, lyrics, style, and provider controls", () => {
    useAudioStudioStore.getState().setActiveWorkflow("music")

    render(<AudioStudioPage />)

    expect(screen.getByLabelText("Prompt")).toBeInTheDocument()
    expect(screen.getByLabelText("Lyrics")).toBeInTheDocument()
    expect(screen.getByLabelText("Style")).toBeInTheDocument()
    expect(screen.getByLabelText("Provider")).toBeInTheDocument()
  })
})
