// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { TtsClip } from "@/db/dexie/types"

const testState = vi.hoisted(() => ({ clips: [] as TtsClip[] }))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | Record<string, unknown>
    ) => {
      const options =
        typeof fallbackOrOptions === "object" ? fallbackOrOptions : undefined
      const template =
        typeof fallbackOrOptions === "string"
          ? fallbackOrOptions
          : String(options?.defaultValue || key)
      return template.replace(/\{\{(\w+)\}\}/g, (_match, name) =>
        String(options?.[name] ?? "")
      )
    }
  })
}))

vi.mock("dexie-react-hooks", () => ({
  useLiveQuery: () => testState.clips
}))

vi.mock("@/db/dexie/schema", () => ({ db: { ttsClips: {} } }))
vi.mock("@/db/dexie/tts-clips", () => ({
  clearTtsClips: vi.fn(),
  deleteTtsClip: vi.fn()
}))
vi.mock("@/utils/download-blob", () => ({ downloadBlob: vi.fn() }))
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn() })
}))

vi.mock("antd", () => {
  const Modal = Object.assign(
    ({ children }: { children?: React.ReactNode }) => <>{children}</>,
    { confirm: vi.fn() }
  )
  return {
    Button: ({ children }: { children?: React.ReactNode }) => (
      <button type="button">{children}</button>
    ),
    Drawer: ({ children, open }: { children?: React.ReactNode; open?: boolean }) =>
      open ? <div>{children}</div> : null,
    Empty: ({ description }: { description?: React.ReactNode }) => (
      <div>{description}</div>
    ),
    Modal,
    Tag: ({ children }: { children?: React.ReactNode }) => <span>{children}</span>,
    Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>
  }
})

import { TtsClipsDrawer } from "../TtsClipsDrawer"

const makeClip = (overrides: Partial<TtsClip> = {}): TtsClip => ({
  id: "clip-1",
  createdAt: Date.parse("2026-07-16T12:00:00Z"),
  provider: "tldw",
  model: "Vendor/Model",
  voice: "Narrator",
  format: "mp3",
  mimeType: "audio/mpeg",
  playbackSpeed: 1,
  utterance: "Hello",
  textPreview: "Hello",
  totalBytes: 3,
  segments: [
    {
      id: "clip-1:0",
      index: 0,
      text: "Hello",
      format: "mp3",
      mimeType: "audio/mpeg",
      blob: new Blob(["abc"], { type: "audio/mpeg" }),
      sizeBytes: 3
    }
  ],
  ...overrides
})

describe("TtsClipsDrawer gateway metadata", () => {
  beforeEach(() => {
    testState.clips = []
  })

  it("shows compact requested, actual, and fallback provenance", () => {
    testState.clips = [
      makeClip({
        requestedBackend: "gateway:company-proxy",
        actualBackends: ["gateway:company-proxy", "openrouter"],
        fallbackUsed: true
      })
    ]

    render(<TtsClipsDrawer open onClose={vi.fn()} />)

    expect(screen.getByText("Requested gateway:company-proxy")).toBeInTheDocument()
    expect(
      screen.getByText("Actual gateway:company-proxy, openrouter")
    ).toBeInTheDocument()
    expect(screen.getByText("Fallback used")).toBeInTheDocument()
  })

  it("keeps old records readable without empty provenance labels", () => {
    testState.clips = [makeClip()]

    render(<TtsClipsDrawer open onClose={vi.fn()} />)

    expect(screen.getByText("Hello")).toBeInTheDocument()
    expect(screen.queryByText(/^Requested /)).not.toBeInTheDocument()
    expect(screen.queryByText(/^Actual /)).not.toBeInTheDocument()
    expect(screen.queryByText("Fallback used")).not.toBeInTheDocument()
  })
})
