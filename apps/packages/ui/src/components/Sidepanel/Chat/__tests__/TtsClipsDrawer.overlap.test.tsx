import React from "react"
import { act, fireEvent, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { TtsClipsDrawer } from "@/components/Sidepanel/Chat/TtsClipsDrawer"

const h = vi.hoisted(() => ({ clips: [] as any[] }))

vi.mock("dexie-react-hooks", () => ({
  useLiveQuery: () => h.clips
}))

vi.mock("@/db/dexie/schema", () => ({ db: { ttsClips: {} } }))

vi.mock("@/db/dexie/tts-clips", () => ({
  clearTtsClips: vi.fn(),
  deleteTtsClip: vi.fn()
}))

vi.mock("@/utils/download-blob", () => ({ downloadBlob: vi.fn() }))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn(), warning: vi.fn() })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: unknown) =>
      typeof fallback === "string" ? fallback : key
  })
}))

// Audio whose play() never settles so a clip stays "playing" until aborted,
// letting us reproduce the overlapping-playback scenario.
class HangingAudio {
  src: string
  playbackRate = 1
  currentTime = 0
  onended: (() => void) | null = null
  onerror: (() => void) | null = null
  error: unknown = null
  constructor(src?: string) {
    this.src = src ?? ""
  }
  play() {
    return new Promise<void>(() => {})
  }
  pause() {}
}

const makeClip = (id: string, createdAt: number) => ({
  id,
  createdAt,
  provider: "tldw",
  voice: "Bella",
  playbackSpeed: 1,
  utterance: `Clip ${id}`,
  textPreview: `Clip ${id}`,
  segments: [
    {
      id: `${id}:0`,
      index: 0,
      text: id,
      format: "mp3",
      mimeType: "audio/mpeg",
      blob: new Blob([id]),
      sizeBytes: 1
    }
  ]
})

describe("TtsClipsDrawer overlapping playback", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    h.clips = []
  })

  it("stops/aborts the currently-playing clip before starting a different one", async () => {
    h.clips = [makeClip("A", 2), makeClip("B", 1)]

    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:clip")
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})
    vi.stubGlobal("Audio", HangingAudio)
    const abortSpy = vi.spyOn(AbortController.prototype, "abort")

    render(<TtsClipsDrawer open onClose={() => {}} />)

    const rows = Array.from(
      document.querySelectorAll<HTMLDivElement>(".rounded-xl")
    )
    expect(rows.length).toBe(2)

    const playA = rows[0].querySelector("button") as HTMLButtonElement
    const playB = rows[1].querySelector("button") as HTMLButtonElement
    expect(playA).toBeTruthy()
    expect(playB).toBeTruthy()

    // Start clip A. It stays "playing" because play() never settles.
    await act(async () => {
      fireEvent.click(playA)
    })
    expect(abortSpy).not.toHaveBeenCalled()

    // Start clip B. The previously playing clip A must be aborted first.
    await act(async () => {
      fireEvent.click(playB)
    })

    expect(abortSpy).toHaveBeenCalled()
  })
})
