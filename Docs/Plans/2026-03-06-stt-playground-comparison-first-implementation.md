# STT Playground Comparison-First Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign SttPlaygroundPage so users record once and compare transcription results across multiple models side-by-side.

**Architecture:** Three-zone layout (Recording Strip, Comparison Panel, History Panel). Audio blobs retained in IndexedDB via Dexie. New custom hooks (`useAudioRecorder`, `useComparisonTranscribe`) encapsulate recording and parallel-transcription logic. Existing `tldwClient.transcribeAudio()` and Dexie patterns (from audiobook-projects.ts) are reused.

**Tech Stack:** React, Vitest + React Testing Library, Ant Design, Dexie (IndexedDB), @plasmohq/storage/hook, TldwApiClient, Lucide icons

**Design Doc:** `Docs/Plans/2026-03-06-stt-playground-comparison-first-redesign.md`

---

## Task 1: Add Dexie table for STT recordings

**Files:**
- Modify: `apps/packages/ui/src/db/dexie/schema.ts`
- Create: `apps/packages/ui/src/db/dexie/stt-recordings.ts`
- Test: `apps/packages/ui/src/db/dexie/__tests__/stt-recordings.test.ts`

**Context:** The codebase already uses Dexie for audiobook chapter assets (see `audiobook-projects.ts`). Follow the same pattern: store blobs directly in IndexedDB with a storage cap.

**Current dependency note:** This historical plan references `fake-indexeddb/auto`
for the initial failing test. As of TASK-141, `fake-indexeddb` is no longer a
direct WebUI dev dependency; revive this plan with the current jsdom/browser
test shims, or re-add a narrowly justified test dependency in the implementing
PR.

**Step 1: Write failing tests**

```typescript
// apps/packages/ui/src/db/dexie/__tests__/stt-recordings.test.ts
import { describe, it, expect, beforeEach } from "vitest"
import "fake-indexeddb/auto"
import {
  saveSttRecording,
  getSttRecording,
  deleteSttRecording,
  listSttRecordings,
  type SttRecording
} from "../stt-recordings"

describe("stt-recordings Dexie store", () => {
  beforeEach(async () => {
    const all = await listSttRecordings()
    for (const r of all) await deleteSttRecording(r.id)
  })

  it("saves and retrieves a recording by id", async () => {
    const blob = new Blob(["audio-data"], { type: "audio/webm" })
    const id = await saveSttRecording({
      blob,
      durationMs: 5000,
      mimeType: "audio/webm"
    })
    expect(typeof id).toBe("string")
    const rec = await getSttRecording(id)
    expect(rec).not.toBeNull()
    expect(rec!.durationMs).toBe(5000)
    expect(rec!.blob.size).toBe(blob.size)
  })

  it("lists recordings sorted by createdAt desc", async () => {
    const blob = new Blob(["x"], { type: "audio/webm" })
    await saveSttRecording({ blob, durationMs: 100, mimeType: "audio/webm" })
    await saveSttRecording({ blob, durationMs: 200, mimeType: "audio/webm" })
    const all = await listSttRecordings()
    expect(all).toHaveLength(2)
    expect(all[0].createdAt).toBeGreaterThanOrEqual(all[1].createdAt)
  })

  it("deletes a recording", async () => {
    const blob = new Blob(["x"], { type: "audio/webm" })
    const id = await saveSttRecording({ blob, durationMs: 100, mimeType: "audio/webm" })
    await deleteSttRecording(id)
    const rec = await getSttRecording(id)
    expect(rec).toBeUndefined()
  })

  it("enforces max 20 recordings by evicting oldest", async () => {
    const blob = new Blob(["x"], { type: "audio/webm" })
    for (let i = 0; i < 21; i++) {
      await saveSttRecording({ blob, durationMs: i * 100, mimeType: "audio/webm" })
    }
    const all = await listSttRecordings()
    expect(all.length).toBeLessThanOrEqual(20)
  })
})
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/packages/ui && npx vitest run src/db/dexie/__tests__/stt-recordings.test.ts`
Expected: FAIL — module `../stt-recordings` not found

**Step 3: Add Dexie schema version**

In `apps/packages/ui/src/db/dexie/schema.ts`, add a new version bump with the `sttRecordings` table:

```typescript
// Add to the class:
sttRecordings!: Table<SttRecordingRow>

// Add version (increment from current latest):
this.version(N).stores({
  // ... keep all existing stores ...
  sttRecordings: "id, createdAt"
})
```

Type definition:
```typescript
export interface SttRecordingRow {
  id: string
  blob: Blob
  mimeType: string
  durationMs: number
  createdAt: number // Date.now()
}
```

**Step 4: Implement stt-recordings.ts**

```typescript
// apps/packages/ui/src/db/dexie/stt-recordings.ts
import { db, type SttRecordingRow } from "./schema"

const MAX_RECORDINGS = 20

export type SttRecording = SttRecordingRow

export async function saveSttRecording(input: {
  blob: Blob
  durationMs: number
  mimeType: string
}): Promise<string> {
  const id = crypto.randomUUID()
  const row: SttRecordingRow = {
    id,
    blob: input.blob,
    mimeType: input.mimeType,
    durationMs: input.durationMs,
    createdAt: Date.now()
  }
  await db.sttRecordings.put(row)
  // Evict oldest if over cap
  const count = await db.sttRecordings.count()
  if (count > MAX_RECORDINGS) {
    const oldest = await db.sttRecordings
      .orderBy("createdAt")
      .limit(count - MAX_RECORDINGS)
      .toArray()
    await db.sttRecordings.bulkDelete(oldest.map((r) => r.id))
  }
  return id
}

export async function getSttRecording(id: string): Promise<SttRecording | undefined> {
  return db.sttRecordings.get(id)
}

export async function deleteSttRecording(id: string): Promise<void> {
  await db.sttRecordings.delete(id)
}

export async function listSttRecordings(): Promise<SttRecording[]> {
  return db.sttRecordings.orderBy("createdAt").reverse().toArray()
}
```

**Step 5: Run tests to verify they pass**

Run: `cd apps/packages/ui && npx vitest run src/db/dexie/__tests__/stt-recordings.test.ts`
Expected: 4 tests PASS

**Step 6: Commit**

```bash
git add apps/packages/ui/src/db/dexie/schema.ts \
       apps/packages/ui/src/db/dexie/stt-recordings.ts \
       apps/packages/ui/src/db/dexie/__tests__/stt-recordings.test.ts
git commit -m "feat(stt): add Dexie table for STT recording blob persistence"
```

---

## Task 2: Create useAudioRecorder hook

**Files:**
- Create: `apps/packages/ui/src/hooks/useAudioRecorder.ts`
- Test: `apps/packages/ui/src/hooks/__tests__/useAudioRecorder.test.ts`

**Context:** This hook encapsulates MediaRecorder lifecycle, duration timer, and blob retention. It replaces the inline recording logic in SttPlaygroundPage. The existing `useMicStream` hook uses WebAudio API for PCM — this hook uses MediaRecorder for webm/opus blobs (compatible with the transcription endpoint).

**Step 1: Write failing tests**

```typescript
// apps/packages/ui/src/hooks/__tests__/useAudioRecorder.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest"
import { renderHook, act } from "@testing-library/react"
import { useAudioRecorder } from "../useAudioRecorder"

// Mock MediaRecorder
const mockStop = vi.fn()
const mockStart = vi.fn()
let mockOndataavailable: ((e: any) => void) | null = null
let mockOnstop: (() => void) | null = null

class MockMediaRecorder {
  ondataavailable: ((e: any) => void) | null = null
  onstop: (() => void) | null = null
  onerror: ((e: any) => void) | null = null
  mimeType = "audio/webm"
  state = "inactive"
  start = vi.fn(() => {
    this.state = "recording"
    mockStart()
  })
  stop = vi.fn(() => {
    this.state = "inactive"
    mockStop()
    // Simulate data + stop events
    if (this.ondataavailable) {
      this.ondataavailable({ data: new Blob(["audio"], { type: "audio/webm" }) })
    }
    if (this.onstop) this.onstop()
  })
}

const mockGetUserMedia = vi.fn().mockResolvedValue({
  getTracks: () => [{ stop: vi.fn() }]
})

beforeEach(() => {
  vi.stubGlobal("MediaRecorder", MockMediaRecorder)
  vi.stubGlobal("navigator", {
    mediaDevices: { getUserMedia: mockGetUserMedia }
  })
  vi.clearAllMocks()
})

describe("useAudioRecorder", () => {
  it("starts in idle state", () => {
    const { result } = renderHook(() => useAudioRecorder())
    expect(result.current.status).toBe("idle")
    expect(result.current.blob).toBeNull()
    expect(result.current.durationMs).toBe(0)
  })

  it("transitions to recording on startRecording", async () => {
    const { result } = renderHook(() => useAudioRecorder())
    await act(async () => {
      await result.current.startRecording()
    })
    expect(result.current.status).toBe("recording")
    expect(mockGetUserMedia).toHaveBeenCalledWith({ audio: true })
  })

  it("produces a blob on stopRecording", async () => {
    const { result } = renderHook(() => useAudioRecorder())
    await act(async () => {
      await result.current.startRecording()
    })
    await act(async () => {
      result.current.stopRecording()
    })
    expect(result.current.status).toBe("idle")
    expect(result.current.blob).toBeInstanceOf(Blob)
  })

  it("clears blob on clearRecording", async () => {
    const { result } = renderHook(() => useAudioRecorder())
    await act(async () => {
      await result.current.startRecording()
    })
    await act(async () => {
      result.current.stopRecording()
    })
    expect(result.current.blob).not.toBeNull()
    act(() => {
      result.current.clearRecording()
    })
    expect(result.current.blob).toBeNull()
    expect(result.current.durationMs).toBe(0)
  })

  it("accepts a blob via loadBlob for re-compare", () => {
    const { result } = renderHook(() => useAudioRecorder())
    const blob = new Blob(["loaded"], { type: "audio/webm" })
    act(() => {
      result.current.loadBlob(blob, 3000)
    })
    expect(result.current.blob).toBe(blob)
    expect(result.current.durationMs).toBe(3000)
  })
})
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/packages/ui && npx vitest run src/hooks/__tests__/useAudioRecorder.test.ts`
Expected: FAIL — module `../useAudioRecorder` not found

**Step 3: Implement the hook**

```typescript
// apps/packages/ui/src/hooks/useAudioRecorder.ts
import { useCallback, useRef, useState } from "react"

export type RecorderStatus = "idle" | "recording"

export function useAudioRecorder() {
  const [status, setStatus] = useState<RecorderStatus>("idle")
  const [blob, setBlob] = useState<Blob | null>(null)
  const [durationMs, setDurationMs] = useState(0)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const startTimeRef = useRef<number>(0)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const startRecording = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    streamRef.current = stream
    const recorder = new MediaRecorder(stream)
    recorderRef.current = recorder
    chunksRef.current = []
    startTimeRef.current = Date.now()
    setDurationMs(0)
    setBlob(null)

    recorder.ondataavailable = (e: BlobEvent) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
    }

    recorder.onstop = () => {
      const finalBlob = new Blob(chunksRef.current, {
        type: recorder.mimeType || "audio/webm"
      })
      const elapsed = Date.now() - startTimeRef.current
      setBlob(finalBlob)
      setDurationMs(elapsed)
      setStatus("idle")
      if (timerRef.current) clearInterval(timerRef.current)
      try {
        stream.getTracks().forEach((t) => t.stop())
      } catch {}
    }

    recorder.start()
    setStatus("recording")
    timerRef.current = setInterval(() => {
      setDurationMs(Date.now() - startTimeRef.current)
    }, 200)
  }, [])

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current
    if (recorder && recorder.state !== "inactive") {
      recorder.stop()
    }
  }, [])

  const clearRecording = useCallback(() => {
    setBlob(null)
    setDurationMs(0)
    chunksRef.current = []
  }, [])

  const loadBlob = useCallback((loadedBlob: Blob, loadedDurationMs: number) => {
    setBlob(loadedBlob)
    setDurationMs(loadedDurationMs)
  }, [])

  return {
    status,
    blob,
    durationMs,
    startRecording,
    stopRecording,
    clearRecording,
    loadBlob
  }
}
```

**Step 4: Run tests to verify they pass**

Run: `cd apps/packages/ui && npx vitest run src/hooks/__tests__/useAudioRecorder.test.ts`
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/hooks/useAudioRecorder.ts \
       apps/packages/ui/src/hooks/__tests__/useAudioRecorder.test.ts
git commit -m "feat(stt): add useAudioRecorder hook with blob retention"
```

---

## Task 3: Create useComparisonTranscribe hook

**Files:**
- Create: `apps/packages/ui/src/hooks/useComparisonTranscribe.ts`
- Test: `apps/packages/ui/src/hooks/__tests__/useComparisonTranscribe.test.ts`

**Context:** This hook takes a blob + list of model names, fires parallel transcription requests via `tldwClient.transcribeAudio()`, and returns per-model results with loading/error states. Each result includes latency and word count.

**Step 1: Write failing tests**

```typescript
// apps/packages/ui/src/hooks/__tests__/useComparisonTranscribe.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest"
import { renderHook, act, waitFor } from "@testing-library/react"
import { useComparisonTranscribe } from "../useComparisonTranscribe"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    transcribeAudio: vi.fn()
  }
}))

import { tldwClient } from "@/services/tldw/TldwApiClient"
const mockTranscribe = vi.mocked(tldwClient.transcribeAudio)

beforeEach(() => vi.clearAllMocks())

describe("useComparisonTranscribe", () => {
  it("starts with empty results", () => {
    const { result } = renderHook(() => useComparisonTranscribe())
    expect(result.current.results).toEqual([])
    expect(result.current.isRunning).toBe(false)
  })

  it("transcribes blob with multiple models in parallel", async () => {
    mockTranscribe
      .mockResolvedValueOnce({ text: "hello from whisper" })
      .mockResolvedValueOnce({ text: "hello from distil" })

    const { result } = renderHook(() => useComparisonTranscribe())
    const blob = new Blob(["audio"], { type: "audio/webm" })

    await act(async () => {
      await result.current.transcribeAll(blob, ["whisper-1", "distil-v3"], {
        language: "en-US"
      })
    })

    expect(result.current.results).toHaveLength(2)
    expect(result.current.results[0].model).toBe("whisper-1")
    expect(result.current.results[0].text).toBe("hello from whisper")
    expect(result.current.results[0].status).toBe("done")
    expect(typeof result.current.results[0].latencyMs).toBe("number")
    expect(typeof result.current.results[0].wordCount).toBe("number")
    expect(result.current.results[1].model).toBe("distil-v3")
    expect(result.current.results[1].text).toBe("hello from distil")
  })

  it("captures per-model errors without failing others", async () => {
    mockTranscribe
      .mockResolvedValueOnce({ text: "ok" })
      .mockRejectedValueOnce(new Error("model not found"))

    const { result } = renderHook(() => useComparisonTranscribe())
    const blob = new Blob(["audio"], { type: "audio/webm" })

    await act(async () => {
      await result.current.transcribeAll(blob, ["good-model", "bad-model"], {})
    })

    expect(result.current.results[0].status).toBe("done")
    expect(result.current.results[1].status).toBe("error")
    expect(result.current.results[1].error).toBe("model not found")
  })

  it("retries a single model", async () => {
    mockTranscribe
      .mockRejectedValueOnce(new Error("timeout"))
      .mockResolvedValueOnce({ text: "retry worked" })

    const { result } = renderHook(() => useComparisonTranscribe())
    const blob = new Blob(["audio"], { type: "audio/webm" })

    await act(async () => {
      await result.current.transcribeAll(blob, ["flaky-model"], {})
    })
    expect(result.current.results[0].status).toBe("error")

    await act(async () => {
      await result.current.retryModel(blob, "flaky-model", {})
    })
    expect(result.current.results[0].status).toBe("done")
    expect(result.current.results[0].text).toBe("retry worked")
  })

  it("clears results", async () => {
    mockTranscribe.mockResolvedValueOnce({ text: "hi" })
    const { result } = renderHook(() => useComparisonTranscribe())
    const blob = new Blob(["audio"], { type: "audio/webm" })
    await act(async () => {
      await result.current.transcribeAll(blob, ["m1"], {})
    })
    expect(result.current.results).toHaveLength(1)
    act(() => result.current.clearResults())
    expect(result.current.results).toEqual([])
  })
})
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/packages/ui && npx vitest run src/hooks/__tests__/useComparisonTranscribe.test.ts`
Expected: FAIL — module not found

**Step 3: Implement the hook**

```typescript
// apps/packages/ui/src/hooks/useComparisonTranscribe.ts
import { useCallback, useState } from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"

export interface ComparisonResult {
  model: string
  text: string
  status: "pending" | "running" | "done" | "error"
  error?: string
  latencyMs?: number
  wordCount?: number
}

function extractText(res: any): string {
  if (!res) return ""
  if (typeof res === "string") return res
  if (typeof res.text === "string") return res.text
  if (typeof res.transcript === "string") return res.transcript
  if (Array.isArray(res.segments)) {
    return res.segments.map((s: any) => s?.text || "").join(" ").trim()
  }
  return ""
}

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length
}

export function useComparisonTranscribe() {
  const [results, setResults] = useState<ComparisonResult[]>([])
  const [isRunning, setIsRunning] = useState(false)

  const transcribeAll = useCallback(
    async (blob: Blob, models: string[], sttOptions: Record<string, any>) => {
      setIsRunning(true)
      const initial: ComparisonResult[] = models.map((m) => ({
        model: m,
        text: "",
        status: "pending"
      }))
      setResults(initial)

      const promises = models.map(async (model, idx) => {
        setResults((prev) =>
          prev.map((r, i) => (i === idx ? { ...r, status: "running" } : r))
        )
        const start = performance.now()
        try {
          const res = await tldwClient.transcribeAudio(blob, {
            ...sttOptions,
            model
          })
          const text = extractText(res)
          const latencyMs = Math.round(performance.now() - start)
          setResults((prev) =>
            prev.map((r, i) =>
              i === idx
                ? { ...r, status: "done", text, latencyMs, wordCount: countWords(text) }
                : r
            )
          )
        } catch (e: any) {
          setResults((prev) =>
            prev.map((r, i) =>
              i === idx
                ? { ...r, status: "error", error: e?.message || "Transcription failed" }
                : r
            )
          )
        }
      })

      await Promise.allSettled(promises)
      setIsRunning(false)
    },
    []
  )

  const retryModel = useCallback(
    async (blob: Blob, model: string, sttOptions: Record<string, any>) => {
      setResults((prev) =>
        prev.map((r) => (r.model === model ? { ...r, status: "running", error: undefined } : r))
      )
      const start = performance.now()
      try {
        const res = await tldwClient.transcribeAudio(blob, { ...sttOptions, model })
        const text = extractText(res)
        const latencyMs = Math.round(performance.now() - start)
        setResults((prev) =>
          prev.map((r) =>
            r.model === model
              ? { ...r, status: "done", text, latencyMs, wordCount: countWords(text) }
              : r
          )
        )
      } catch (e: any) {
        setResults((prev) =>
          prev.map((r) =>
            r.model === model
              ? { ...r, status: "error", error: e?.message || "Transcription failed" }
              : r
          )
        )
      }
    },
    []
  )

  const clearResults = useCallback(() => setResults([]), [])

  return { results, isRunning, transcribeAll, retryModel, clearResults }
}
```

**Step 4: Run tests to verify they pass**

Run: `cd apps/packages/ui && npx vitest run src/hooks/__tests__/useComparisonTranscribe.test.ts`
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/hooks/useComparisonTranscribe.ts \
       apps/packages/ui/src/hooks/__tests__/useComparisonTranscribe.test.ts
git commit -m "feat(stt): add useComparisonTranscribe hook for parallel multi-model transcription"
```

---

## Task 4: Build RecordingStrip component

**Files:**
- Create: `apps/packages/ui/src/components/Option/STT/RecordingStrip.tsx`
- Test: `apps/packages/ui/src/components/Option/STT/__tests__/RecordingStrip.test.tsx`

**Context:** Zone 1 of the redesign. Compact horizontal bar with record/stop, duration timer, playback, upload file, gear toggle. Uses `useAudioRecorder` from Task 2. Follows the existing Ant Design + Lucide icon pattern used throughout the Option components.

**Step 1: Write failing tests**

```typescript
// apps/packages/ui/src/components/Option/STT/__tests__/RecordingStrip.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"

// Mock hooks
const mockStartRecording = vi.fn()
const mockStopRecording = vi.fn()
const mockClearRecording = vi.fn()
const mockLoadBlob = vi.fn()

vi.mock("@/hooks/useAudioRecorder", () => ({
  useAudioRecorder: () => ({
    status: "idle",
    blob: null,
    durationMs: 0,
    startRecording: mockStartRecording,
    stopRecording: mockStopRecording,
    clearRecording: mockClearRecording,
    loadBlob: mockLoadBlob
  })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback: string) => fallback
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    success: vi.fn()
  })
}))

import { RecordingStrip } from "../RecordingStrip"

describe("RecordingStrip", () => {
  beforeEach(() => vi.clearAllMocks())

  it("renders record button in idle state", () => {
    render(<RecordingStrip onBlobReady={vi.fn()} />)
    expect(screen.getByRole("button", { name: /record/i })).toBeInTheDocument()
  })

  it("calls startRecording when record button clicked", () => {
    render(<RecordingStrip onBlobReady={vi.fn()} />)
    fireEvent.click(screen.getByRole("button", { name: /record/i }))
    expect(mockStartRecording).toHaveBeenCalled()
  })

  it("shows upload button", () => {
    render(<RecordingStrip onBlobReady={vi.fn()} />)
    expect(screen.getByRole("button", { name: /upload/i })).toBeInTheDocument()
  })

  it("shows settings toggle button", () => {
    render(<RecordingStrip onBlobReady={vi.fn()} />)
    expect(screen.getByRole("button", { name: /settings/i })).toBeInTheDocument()
  })
})
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/RecordingStrip.test.tsx`
Expected: FAIL — module not found

**Step 3: Implement RecordingStrip**

```tsx
// apps/packages/ui/src/components/Option/STT/RecordingStrip.tsx
import React from "react"
import { Button, Card, Tooltip, Upload } from "antd"
import { Mic, Square, Upload as UploadIcon, Settings, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useAudioRecorder } from "@/hooks/useAudioRecorder"
import { useAntdNotification } from "@/hooks/useAntdNotification"

interface RecordingStripProps {
  onBlobReady: (blob: Blob, durationMs: number) => void
  onSettingsToggle?: () => void
  blob?: Blob | null
  durationMs?: number
}

function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000)
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
}

export const RecordingStrip: React.FC<RecordingStripProps> = ({
  onBlobReady,
  onSettingsToggle
}) => {
  const { t } = useTranslation(["playground"])
  const notification = useAntdNotification()
  const recorder = useAudioRecorder()
  const audioRef = React.useRef<HTMLAudioElement | null>(null)
  const [playbackUrl, setPlaybackUrl] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (recorder.blob) {
      const url = URL.createObjectURL(recorder.blob)
      setPlaybackUrl(url)
      onBlobReady(recorder.blob, recorder.durationMs)
      return () => URL.revokeObjectURL(url)
    } else {
      setPlaybackUrl(null)
    }
  }, [recorder.blob, recorder.durationMs, onBlobReady])

  const handleUpload = React.useCallback(
    (file: File) => {
      recorder.loadBlob(file, 0)
      return false // prevent antd auto-upload
    },
    [recorder]
  )

  const handleRecord = async () => {
    if (recorder.status === "recording") {
      recorder.stopRecording()
    } else {
      try {
        await recorder.startRecording()
      } catch {
        notification.error({
          message: t("playground:actions.speechErrorTitle", "Recording failed"),
          description: t(
            "playground:actions.speechMicError",
            "Unable to access your microphone. Check browser permissions and try again."
          )
        })
      }
    }
  }

  const isRecording = recorder.status === "recording"

  return (
    <Card size="small">
      <div className="flex flex-wrap items-center gap-3">
        <Tooltip
          title={
            isRecording
              ? t("playground:stt.stopTooltip", "Stop recording")
              : t("playground:stt.startTooltip", "Start recording (Space)")
          }
        >
          <Button
            type={isRecording ? "default" : "primary"}
            danger={isRecording}
            icon={
              isRecording ? (
                <Square className="h-4 w-4" />
              ) : (
                <Mic className="h-4 w-4" />
              )
            }
            onClick={handleRecord}
            aria-label={
              isRecording
                ? t("playground:stt.stopButton", "Stop")
                : t("playground:stt.recordButton", "Record")
            }
          >
            {isRecording
              ? t("playground:stt.stopButton", "Stop")
              : t("playground:stt.recordButton", "Record")}
          </Button>
        </Tooltip>

        {/* Duration timer */}
        <span className="font-mono text-sm tabular-nums text-text-muted min-w-[48px]">
          {formatDuration(recorder.durationMs)}
        </span>

        {/* Audio level indicator placeholder — animated when recording */}
        {isRecording && (
          <div
            className="flex items-center gap-0.5 h-5"
            role="meter"
            aria-label="Audio level"
            aria-valuenow={0}
          >
            {[1, 2, 3, 4, 5].map((i) => (
              <div
                key={i}
                className="w-1 bg-primary rounded-full animate-pulse"
                style={{
                  height: `${8 + Math.random() * 12}px`,
                  animationDelay: `${i * 0.1}s`
                }}
              />
            ))}
          </div>
        )}

        {/* Playback */}
        {playbackUrl && !isRecording && (
          <audio ref={audioRef} src={playbackUrl} controls className="h-8 max-w-[200px]" />
        )}

        <div className="flex-1" />

        {/* Clear recording */}
        {recorder.blob && !isRecording && (
          <Button
            size="small"
            type="text"
            icon={<Trash2 className="h-3.5 w-3.5" />}
            onClick={recorder.clearRecording}
          >
            {t("playground:stt.clearRecording", "Clear")}
          </Button>
        )}

        {/* Upload file */}
        <Upload
          accept="audio/*"
          showUploadList={false}
          beforeUpload={handleUpload}
        >
          <Button
            size="small"
            icon={<UploadIcon className="h-3.5 w-3.5" />}
            aria-label={t("playground:stt.uploadFile", "Upload audio file")}
          >
            {t("playground:stt.uploadFile", "Upload file")}
          </Button>
        </Upload>

        {/* Settings gear */}
        {onSettingsToggle && (
          <Button
            size="small"
            type="text"
            icon={<Settings className="h-3.5 w-3.5" />}
            onClick={onSettingsToggle}
            aria-label={t("playground:stt.settingsToggle", "Toggle settings")}
          />
        )}
      </div>
    </Card>
  )
}
```

**Step 4: Run tests to verify they pass**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/RecordingStrip.test.tsx`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/STT/RecordingStrip.tsx \
       apps/packages/ui/src/components/Option/STT/__tests__/RecordingStrip.test.tsx
git commit -m "feat(stt): add RecordingStrip component with timer, playback, upload"
```

---

## Task 5: Build ComparisonPanel component

**Files:**
- Create: `apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/STT/__tests__/ComparisonPanel.test.tsx`

**Context:** Zone 2 of the redesign. Multi-select model bar + responsive results card grid. Uses `useComparisonTranscribe` from Task 3. Each card shows model name, editable transcript, latency, word count, and copy/save/retry actions.

**Step 1: Write failing tests**

```typescript
// apps/packages/ui/src/components/Option/STT/__tests__/ComparisonPanel.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"

const mockTranscribeAll = vi.fn()
const mockRetryModel = vi.fn()
const mockClearResults = vi.fn()

vi.mock("@/hooks/useComparisonTranscribe", () => ({
  useComparisonTranscribe: () => ({
    results: [],
    isRunning: false,
    transcribeAll: mockTranscribeAll,
    retryModel: mockRetryModel,
    clearResults: mockClearResults
  })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback: string) => fallback
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    success: vi.fn()
  })
}))

import { ComparisonPanel } from "../ComparisonPanel"

describe("ComparisonPanel", () => {
  beforeEach(() => vi.clearAllMocks())

  it("renders model select and disabled transcribe button when no blob", () => {
    render(
      <ComparisonPanel
        blob={null}
        availableModels={["whisper-1", "distil-v3"]}
        sttOptions={{}}
        onSaveToNotes={vi.fn()}
      />
    )
    expect(screen.getByText(/transcribe all/i)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /transcribe all/i })).toBeDisabled()
  })

  it("enables transcribe button when blob and models selected", () => {
    const blob = new Blob(["audio"], { type: "audio/webm" })
    render(
      <ComparisonPanel
        blob={blob}
        availableModels={["whisper-1", "distil-v3"]}
        selectedModels={["whisper-1"]}
        sttOptions={{}}
        onSaveToNotes={vi.fn()}
      />
    )
    expect(screen.getByRole("button", { name: /transcribe all/i })).not.toBeDisabled()
  })

  it("shows empty state message when no results", () => {
    render(
      <ComparisonPanel
        blob={null}
        availableModels={["whisper-1"]}
        sttOptions={{}}
        onSaveToNotes={vi.fn()}
      />
    )
    expect(screen.getByText(/select models and record/i)).toBeInTheDocument()
  })
})
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/ComparisonPanel.test.tsx`
Expected: FAIL — module not found

**Step 3: Implement ComparisonPanel**

```tsx
// apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx
import React from "react"
import { Button, Card, Input, Select, Skeleton, Tag, Tooltip, Typography } from "antd"
import { Copy, RotateCcw, Save } from "lucide-react"
import { useTranslation } from "react-i18next"
import {
  useComparisonTranscribe,
  type ComparisonResult
} from "@/hooks/useComparisonTranscribe"
import { useAntdNotification } from "@/hooks/useAntdNotification"

const { Text } = Typography

interface ComparisonPanelProps {
  blob: Blob | null
  availableModels: string[]
  selectedModels?: string[]
  sttOptions: Record<string, any>
  onSaveToNotes: (text: string, model: string) => void
}

export const ComparisonPanel: React.FC<ComparisonPanelProps> = ({
  blob,
  availableModels,
  selectedModels: controlledModels,
  sttOptions,
  onSaveToNotes
}) => {
  const { t } = useTranslation(["playground"])
  const notification = useAntdNotification()
  const { results, isRunning, transcribeAll, retryModel, clearResults } =
    useComparisonTranscribe()
  const [models, setModels] = React.useState<string[]>(controlledModels ?? [])

  React.useEffect(() => {
    if (controlledModels) setModels(controlledModels)
  }, [controlledModels])

  const handleTranscribeAll = async () => {
    if (!blob || models.length === 0) return
    await transcribeAll(blob, models, sttOptions)
  }

  const handleCopy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      notification.success({
        message: t("playground:stt.copied", "Copied to clipboard")
      })
    } catch {
      notification.error({
        message: t("playground:stt.copyFailed", "Copy failed")
      })
    }
  }

  const canTranscribe = blob !== null && models.length > 0 && !isRunning

  return (
    <Card>
      {/* Model selection bar */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <Select
          mode="multiple"
          placeholder={t("playground:stt.selectModels", "Select models to compare")}
          value={models}
          onChange={setModels}
          style={{ minWidth: 300, flex: 1 }}
          options={availableModels.map((m) => ({ label: m, value: m }))}
          maxTagCount="responsive"
        />
        <Button
          type="primary"
          loading={isRunning}
          disabled={!canTranscribe}
          onClick={handleTranscribeAll}
          aria-label={t("playground:stt.transcribeAll", "Transcribe All")}
        >
          {t("playground:stt.transcribeAll", "Transcribe All")}
          <Text type="secondary" className="ml-2 text-xs hidden sm:inline">
            ⌘⏎
          </Text>
        </Button>
      </div>

      {/* Results grid */}
      {results.length === 0 ? (
        <Text type="secondary" className="text-sm">
          {t(
            "playground:stt.comparisonEmpty",
            "Select models and record audio to compare transcription results."
          )}
        </Text>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {results.map((r) => (
            <ResultCard
              key={r.model}
              result={r}
              onCopy={handleCopy}
              onRetry={() => blob && retryModel(blob, r.model, sttOptions)}
              onSave={() => onSaveToNotes(r.text, r.model)}
            />
          ))}
        </div>
      )}
    </Card>
  )
}

const ResultCard: React.FC<{
  result: ComparisonResult
  onCopy: (text: string) => void
  onRetry: () => void
  onSave: () => void
}> = ({ result, onCopy, onRetry, onSave }) => {
  const { t } = useTranslation(["playground"])
  const [editedText, setEditedText] = React.useState(result.text)

  React.useEffect(() => {
    setEditedText(result.text)
  }, [result.text])

  if (result.status === "pending" || result.status === "running") {
    return (
      <Card
        size="small"
        title={result.model}
        role="region"
        aria-label={`Transcription result from ${result.model}`}
      >
        <Skeleton active paragraph={{ rows: 3 }} />
      </Card>
    )
  }

  if (result.status === "error") {
    return (
      <Card
        size="small"
        title={result.model}
        role="region"
        aria-label={`Transcription result from ${result.model}`}
      >
        <Text type="danger" className="text-sm block mb-2">
          {result.error}
        </Text>
        <Button size="small" icon={<RotateCcw className="h-3 w-3" />} onClick={onRetry}>
          {t("common:retry", "Retry")}
        </Button>
      </Card>
    )
  }

  return (
    <Card
      size="small"
      title={result.model}
      role="region"
      aria-label={`Transcription result from ${result.model}`}
    >
      <Input.TextArea
        value={editedText}
        onChange={(e) => setEditedText(e.target.value)}
        autoSize={{ minRows: 3, maxRows: 8 }}
        aria-live="polite"
      />
      <div className="flex items-center justify-between mt-2">
        <div className="flex gap-2">
          {result.latencyMs != null && (
            <Tag bordered>{(result.latencyMs / 1000).toFixed(1)}s</Tag>
          )}
          {result.wordCount != null && (
            <Tag bordered>
              {result.wordCount} {t("playground:stt.words", "words")}
            </Tag>
          )}
        </div>
        <div className="flex gap-1">
          <Tooltip title={t("playground:stt.copy", "Copy")}>
            <Button
              size="small"
              type="text"
              icon={<Copy className="h-3 w-3" />}
              onClick={() => onCopy(editedText)}
            />
          </Tooltip>
          <Tooltip title={t("playground:stt.saveToNotes", "Save to Notes")}>
            <Button
              size="small"
              type="text"
              icon={<Save className="h-3 w-3" />}
              onClick={onSave}
            />
          </Tooltip>
        </div>
      </div>
    </Card>
  )
}
```

**Step 4: Run tests to verify they pass**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/ComparisonPanel.test.tsx`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx \
       apps/packages/ui/src/components/Option/STT/__tests__/ComparisonPanel.test.tsx
git commit -m "feat(stt): add ComparisonPanel with multi-model results grid"
```

---

## Task 6: Build InlineSettingsPanel component

**Files:**
- Create: `apps/packages/ui/src/components/Option/STT/InlineSettingsPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/STT/__tests__/InlineSettingsPanel.test.tsx`

**Context:** Collapsible panel below Zone 1. Exposes language, task, format, temperature, prompt, and segmentation params as inline controls. Changes are playground-local (don't write to global settings). Uses the same `useStorage` keys as current page for reading defaults, but manages overrides in local state.

**Step 1: Write failing tests**

```typescript
// apps/packages/ui/src/components/Option/STT/__tests__/InlineSettingsPanel.test.tsx
import { describe, it, expect, vi } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultVal: any) => [defaultVal, vi.fn()]
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback: string) => fallback
  })
}))

import { InlineSettingsPanel } from "../InlineSettingsPanel"

describe("InlineSettingsPanel", () => {
  it("renders language, task, and format selects", () => {
    render(<InlineSettingsPanel onChange={vi.fn()} />)
    expect(screen.getByText(/language/i)).toBeInTheDocument()
    expect(screen.getByText(/task/i)).toBeInTheDocument()
    expect(screen.getByText(/format/i)).toBeInTheDocument()
  })

  it("shows segmentation params only when segmentation enabled", () => {
    render(<InlineSettingsPanel onChange={vi.fn()} />)
    // Segmentation off by default — advanced params hidden
    expect(screen.queryByText(/lambda/i)).not.toBeInTheDocument()
  })

  it("shows reset to defaults button", () => {
    render(<InlineSettingsPanel onChange={vi.fn()} />)
    expect(screen.getByRole("button", { name: /reset to defaults/i })).toBeInTheDocument()
  })
})
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/InlineSettingsPanel.test.tsx`
Expected: FAIL — module not found

**Step 3: Implement InlineSettingsPanel**

```tsx
// apps/packages/ui/src/components/Option/STT/InlineSettingsPanel.tsx
import React from "react"
import { Button, Input, InputNumber, Select, Switch, Typography } from "antd"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"

const { Text } = Typography

export interface SttLocalSettings {
  language: string
  task: string
  responseFormat: string
  temperature: number
  prompt: string
  useSegmentation: boolean
  segK: number
  segMinSegmentSize: number
  segLambdaBalance: number
  segUtteranceExpansionWidth: number
  segEmbeddingsProvider: string
  segEmbeddingsModel: string
}

interface InlineSettingsPanelProps {
  onChange: (settings: SttLocalSettings) => void
}

export const InlineSettingsPanel: React.FC<InlineSettingsPanelProps> = ({
  onChange
}) => {
  const { t } = useTranslation(["playground", "settings"])

  // Read global defaults
  const [defLanguage] = useStorage("speechToTextLanguage", "en-US")
  const [defTask] = useStorage("sttTask", "transcribe")
  const [defFormat] = useStorage("sttResponseFormat", "json")
  const [defTemperature] = useStorage("sttTemperature", 0)
  const [defPrompt] = useStorage("sttPrompt", "")
  const [defUseSeg] = useStorage("sttUseSegmentation", false)
  const [defSegK] = useStorage("sttSegK", 6)
  const [defSegMinSize] = useStorage("sttSegMinSegmentSize", 5)
  const [defSegLambda] = useStorage("sttSegLambdaBalance", 0.01)
  const [defSegExpWidth] = useStorage("sttSegUtteranceExpansionWidth", 2)
  const [defSegEmbProv] = useStorage("sttSegEmbeddingsProvider", "")
  const [defSegEmbModel] = useStorage("sttSegEmbeddingsModel", "")

  const defaults: SttLocalSettings = React.useMemo(
    () => ({
      language: defLanguage,
      task: defTask,
      responseFormat: defFormat,
      temperature: defTemperature,
      prompt: defPrompt,
      useSegmentation: defUseSeg,
      segK: defSegK,
      segMinSegmentSize: defSegMinSize,
      segLambdaBalance: defSegLambda,
      segUtteranceExpansionWidth: defSegExpWidth,
      segEmbeddingsProvider: defSegEmbProv,
      segEmbeddingsModel: defSegEmbModel
    }),
    [defLanguage, defTask, defFormat, defTemperature, defPrompt, defUseSeg, defSegK, defSegMinSize, defSegLambda, defSegExpWidth, defSegEmbProv, defSegEmbModel]
  )

  const [local, setLocal] = React.useState<SttLocalSettings>(defaults)

  React.useEffect(() => {
    setLocal(defaults)
  }, [defaults])

  const update = (patch: Partial<SttLocalSettings>) => {
    setLocal((prev) => {
      const next = { ...prev, ...patch }
      onChange(next)
      return next
    })
  }

  const handleReset = () => {
    setLocal(defaults)
    onChange(defaults)
  }

  return (
    <div className="space-y-3 py-3 px-1">
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        <div>
          <Text className="text-xs block mb-1">{t("playground:stt.language", "Language")}</Text>
          <Input
            size="small"
            value={local.language}
            onChange={(e) => update({ language: e.target.value })}
          />
        </div>
        <div>
          <Text className="text-xs block mb-1">{t("playground:stt.task", "Task")}</Text>
          <Select
            size="small"
            value={local.task}
            onChange={(v) => update({ task: v })}
            options={[
              { label: "Transcribe", value: "transcribe" },
              { label: "Translate", value: "translate" }
            ]}
            className="w-full"
          />
        </div>
        <div>
          <Text className="text-xs block mb-1">{t("playground:stt.format", "Format")}</Text>
          <Select
            size="small"
            value={local.responseFormat}
            onChange={(v) => update({ responseFormat: v })}
            options={[
              { label: "JSON", value: "json" },
              { label: "Text", value: "text" },
              { label: "SRT", value: "srt" },
              { label: "VTT", value: "vtt" },
              { label: "Verbose JSON", value: "verbose_json" }
            ]}
            className="w-full"
          />
        </div>
        <div>
          <Text className="text-xs block mb-1">{t("playground:stt.temperature", "Temperature")}</Text>
          <InputNumber
            size="small"
            value={local.temperature}
            min={0}
            max={1}
            step={0.1}
            onChange={(v) => update({ temperature: v ?? 0 })}
            className="w-full"
          />
        </div>
        <div className="col-span-2 sm:col-span-1">
          <Text className="text-xs block mb-1">{t("playground:stt.prompt", "Prompt")}</Text>
          <Input
            size="small"
            value={local.prompt}
            onChange={(e) => update({ prompt: e.target.value })}
            placeholder={t("playground:stt.promptPlaceholder", "Optional context for Whisper")}
          />
        </div>
      </div>

      {/* Segmentation toggle */}
      <div className="flex items-center gap-2">
        <Switch
          size="small"
          checked={local.useSegmentation}
          onChange={(v) => update({ useSegmentation: v })}
        />
        <Text className="text-xs">
          {t("playground:stt.segmentation", "Enable segmentation")}
        </Text>
      </div>

      {/* Segmentation params — only when enabled */}
      {local.useSegmentation && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pl-4 border-l-2 border-border">
          <div>
            <Text className="text-xs block mb-1">K</Text>
            <InputNumber size="small" value={local.segK} min={1} onChange={(v) => update({ segK: v ?? 6 })} className="w-full" />
          </div>
          <div>
            <Text className="text-xs block mb-1">{t("playground:stt.minSegSize", "Min segment size")}</Text>
            <InputNumber size="small" value={local.segMinSegmentSize} min={1} onChange={(v) => update({ segMinSegmentSize: v ?? 5 })} className="w-full" />
          </div>
          <div>
            <Text className="text-xs block mb-1">{t("playground:stt.lambda", "Lambda balance")}</Text>
            <InputNumber size="small" value={local.segLambdaBalance} min={0} max={1} step={0.01} onChange={(v) => update({ segLambdaBalance: v ?? 0.01 })} className="w-full" />
          </div>
          <div>
            <Text className="text-xs block mb-1">{t("playground:stt.expansionWidth", "Expansion width")}</Text>
            <InputNumber size="small" value={local.segUtteranceExpansionWidth} min={0} onChange={(v) => update({ segUtteranceExpansionWidth: v ?? 2 })} className="w-full" />
          </div>
          <div>
            <Text className="text-xs block mb-1">{t("playground:stt.embProvider", "Embeddings provider")}</Text>
            <Input size="small" value={local.segEmbeddingsProvider} onChange={(e) => update({ segEmbeddingsProvider: e.target.value })} />
          </div>
          <div>
            <Text className="text-xs block mb-1">{t("playground:stt.embModel", "Embeddings model")}</Text>
            <Input size="small" value={local.segEmbeddingsModel} onChange={(e) => update({ segEmbeddingsModel: e.target.value })} />
          </div>
        </div>
      )}

      <Button size="small" type="link" onClick={handleReset} aria-label="Reset to defaults">
        {t("playground:stt.resetDefaults", "Reset to defaults")}
      </Button>
    </div>
  )
}
```

**Step 4: Run tests to verify they pass**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/InlineSettingsPanel.test.tsx`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/STT/InlineSettingsPanel.tsx \
       apps/packages/ui/src/components/Option/STT/__tests__/InlineSettingsPanel.test.tsx
git commit -m "feat(stt): add InlineSettingsPanel with playground-local overrides"
```

---

## Task 7: Build HistoryPanel component

**Files:**
- Create: `apps/packages/ui/src/components/Option/STT/HistoryPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/STT/__tests__/HistoryPanel.test.tsx`

**Context:** Zone 3. Shows past recordings with their comparison results. Uses Dexie from Task 1 for blob persistence and Plasmo storage for transcript metadata. Supports re-compare (loading blob back to Zone 1+2), export to Notes, and delete with undo toast.

**Step 1: Write failing tests**

```typescript
// apps/packages/ui/src/components/Option/STT/__tests__/HistoryPanel.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback: string) => fallback
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    success: vi.fn(),
    info: vi.fn()
  })
}))

import { HistoryPanel, type SttHistoryEntry } from "../HistoryPanel"

const sampleEntries: SttHistoryEntry[] = [
  {
    id: "rec-1",
    recordingId: "dexie-1",
    createdAt: new Date("2026-03-06T14:00:00Z").toISOString(),
    durationMs: 5000,
    results: [
      { model: "whisper-1", text: "Hello world from whisper", latencyMs: 1200, wordCount: 4 },
      { model: "distil-v3", text: "Hello world from distil", latencyMs: 800, wordCount: 4 }
    ]
  }
]

describe("HistoryPanel", () => {
  beforeEach(() => vi.clearAllMocks())

  it("shows empty state when no entries", () => {
    render(
      <HistoryPanel
        entries={[]}
        onRecompare={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
      />
    )
    expect(screen.getByText(/start a recording/i)).toBeInTheDocument()
  })

  it("renders entries with model count and timestamp", () => {
    render(
      <HistoryPanel
        entries={sampleEntries}
        onRecompare={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
      />
    )
    expect(screen.getByText(/2 models compared/i)).toBeInTheDocument()
  })

  it("shows clear all button when entries exist", () => {
    render(
      <HistoryPanel
        entries={sampleEntries}
        onRecompare={vi.fn()}
        onExport={vi.fn()}
        onDelete={vi.fn()}
        onClearAll={vi.fn()}
      />
    )
    expect(screen.getByRole("button", { name: /clear all/i })).toBeInTheDocument()
  })

  it("calls onDelete when delete clicked", () => {
    const onDelete = vi.fn()
    render(
      <HistoryPanel
        entries={sampleEntries}
        onRecompare={vi.fn()}
        onExport={vi.fn()}
        onDelete={onDelete}
        onClearAll={vi.fn()}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /delete/i }))
    expect(onDelete).toHaveBeenCalledWith("rec-1")
  })
})
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/HistoryPanel.test.tsx`
Expected: FAIL — module not found

**Step 3: Implement HistoryPanel**

```tsx
// apps/packages/ui/src/components/Option/STT/HistoryPanel.tsx
import React from "react"
import { Button, Card, Collapse, Modal, Tag, Typography } from "antd"
import { Download, RefreshCcw, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"

const { Text } = Typography

export interface SttHistoryResult {
  model: string
  text: string
  latencyMs?: number
  wordCount?: number
}

export interface SttHistoryEntry {
  id: string
  recordingId: string
  createdAt: string
  durationMs: number
  results: SttHistoryResult[]
}

interface HistoryPanelProps {
  entries: SttHistoryEntry[]
  onRecompare: (entry: SttHistoryEntry) => void
  onExport: (entry: SttHistoryEntry) => void
  onDelete: (id: string) => void
  onClearAll: () => void
}

export const HistoryPanel: React.FC<HistoryPanelProps> = ({
  entries,
  onRecompare,
  onExport,
  onDelete,
  onClearAll
}) => {
  const { t } = useTranslation(["playground"])
  const [confirmClear, setConfirmClear] = React.useState(false)

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <Text strong>
          {t("playground:stt.historyTitle", "Recording History")}
        </Text>
        {entries.length > 0 && (
          <Button
            size="small"
            type="text"
            danger
            icon={<Trash2 className="h-3 w-3" />}
            onClick={() => setConfirmClear(true)}
            aria-label={t("playground:stt.clearAll", "Clear all")}
          >
            {t("playground:stt.clearAll", "Clear all")}
          </Button>
        )}
      </div>

      {entries.length === 0 ? (
        <Text type="secondary" className="text-sm">
          {t(
            "playground:stt.emptyHistory",
            "Start a recording to see transcripts here."
          )}
        </Text>
      ) : (
        <div className="space-y-2">
          {entries.map((entry) => (
            <Collapse
              key={entry.id}
              size="small"
              items={[
                {
                  key: entry.id,
                  label: (
                    <div className="flex flex-wrap items-center gap-2">
                      <Text className="text-sm">
                        {new Date(entry.createdAt).toLocaleString()}
                      </Text>
                      <Tag bordered>
                        {(entry.durationMs / 1000).toFixed(1)}s
                      </Tag>
                      <Tag bordered>
                        {entry.results.length}{" "}
                        {t("playground:stt.modelsCompared", "models compared")}
                      </Tag>
                    </div>
                  ),
                  children: (
                    <div className="space-y-2">
                      {entry.results.map((r) => (
                        <div key={r.model} className="flex gap-2">
                          <Tag className="shrink-0">{r.model}</Tag>
                          <Text
                            className="text-sm truncate flex-1"
                            title={r.text}
                          >
                            {r.text}
                          </Text>
                          {r.latencyMs != null && (
                            <Text type="secondary" className="text-xs shrink-0">
                              {(r.latencyMs / 1000).toFixed(1)}s
                            </Text>
                          )}
                        </div>
                      ))}
                      <div className="flex gap-2 pt-2 border-t border-border">
                        <Button
                          size="small"
                          icon={<RefreshCcw className="h-3 w-3" />}
                          onClick={() => onRecompare(entry)}
                        >
                          {t("playground:stt.recompare", "Re-compare")}
                        </Button>
                        <Button
                          size="small"
                          icon={<Download className="h-3 w-3" />}
                          onClick={() => onExport(entry)}
                        >
                          {t("playground:stt.export", "Export")}
                        </Button>
                        <Button
                          size="small"
                          type="text"
                          danger
                          icon={<Trash2 className="h-3 w-3" />}
                          onClick={() => onDelete(entry.id)}
                          aria-label={t("playground:stt.delete", "Delete")}
                        >
                          {t("playground:stt.delete", "Delete")}
                        </Button>
                      </div>
                    </div>
                  )
                }
              ]}
            />
          ))}
        </div>
      )}

      <Modal
        title={t("playground:stt.confirmClearTitle", "Clear all recordings?")}
        open={confirmClear}
        onOk={() => {
          onClearAll()
          setConfirmClear(false)
        }}
        onCancel={() => setConfirmClear(false)}
        okText={t("playground:stt.confirmClearOk", "Delete all")}
        okButtonProps={{ danger: true }}
      >
        <Text>
          {t(
            "playground:stt.confirmClearBody",
            `This will delete ${entries.length} recordings and their transcripts. This cannot be undone.`
          )}
        </Text>
      </Modal>
    </Card>
  )
}
```

**Step 4: Run tests to verify they pass**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/HistoryPanel.test.tsx`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/STT/HistoryPanel.tsx \
       apps/packages/ui/src/components/Option/STT/__tests__/HistoryPanel.test.tsx
git commit -m "feat(stt): add HistoryPanel with re-compare, export, and confirmed clear"
```

---

## Task 8: Wire everything into SttPlaygroundPage

**Files:**
- Modify: `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx` (full rewrite)
- Test: `apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx`

**Context:** Replace the existing 736-line component with the new three-zone layout. Compose RecordingStrip, InlineSettingsPanel, ComparisonPanel, and HistoryPanel. Wire up Dexie persistence for blobs and Plasmo storage for history metadata. Add keyboard shortcuts (Space for record, Cmd+Enter for transcribe).

**Step 1: Write failing integration tests**

```typescript
// apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest"
import { render, screen } from "@testing-library/react"

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultVal: any) => [defaultVal, vi.fn()]
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback: string) => fallback
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getTranscriptionModels: vi.fn().mockResolvedValue({ all_models: ["whisper-1", "distil-v3"] }),
    transcribeAudio: vi.fn().mockResolvedValue({ text: "test" }),
    createNote: vi.fn().mockResolvedValue({})
  }
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    success: vi.fn(),
    info: vi.fn()
  })
}))

vi.mock("@/hooks/useAudioRecorder", () => ({
  useAudioRecorder: () => ({
    status: "idle",
    blob: null,
    durationMs: 0,
    startRecording: vi.fn(),
    stopRecording: vi.fn(),
    clearRecording: vi.fn(),
    loadBlob: vi.fn()
  })
}))

vi.mock("@/hooks/useComparisonTranscribe", () => ({
  useComparisonTranscribe: () => ({
    results: [],
    isRunning: false,
    transcribeAll: vi.fn(),
    retryModel: vi.fn(),
    clearResults: vi.fn()
  })
}))

import { SttPlaygroundPage } from "../SttPlaygroundPage"

describe("SttPlaygroundPage (redesigned)", () => {
  it("renders page title", () => {
    render(<SttPlaygroundPage />)
    expect(screen.getByText(/STT Playground/i)).toBeInTheDocument()
  })

  it("renders recording strip with record button", () => {
    render(<SttPlaygroundPage />)
    expect(screen.getByRole("button", { name: /record/i })).toBeInTheDocument()
  })

  it("renders comparison panel with transcribe all button", () => {
    render(<SttPlaygroundPage />)
    expect(screen.getByText(/transcribe all/i)).toBeInTheDocument()
  })

  it("renders history section", () => {
    render(<SttPlaygroundPage />)
    expect(screen.getByText(/recording history/i)).toBeInTheDocument()
  })
})
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx`
Expected: FAIL (tests may partially pass if old component still exists — the new assertions for "Transcribe All" and "Recording History" should fail)

**Step 3: Rewrite SttPlaygroundPage**

Replace the full contents of `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`:

```tsx
// apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx
import React from "react"
import { useTranslation } from "react-i18next"
import { Typography } from "antd"
import { useStorage } from "@plasmohq/storage/hook"
import { PageShell } from "@/components/Common/PageShell"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { isTimeoutLikeError } from "@/utils/request-timeout"
import { RecordingStrip } from "./RecordingStrip"
import { InlineSettingsPanel, type SttLocalSettings } from "./InlineSettingsPanel"
import { ComparisonPanel } from "./ComparisonPanel"
import { HistoryPanel, type SttHistoryEntry, type SttHistoryResult } from "./HistoryPanel"
import {
  saveSttRecording,
  getSttRecording,
  deleteSttRecording
} from "@/db/dexie/stt-recordings"

const { Text, Title } = Typography

const MAX_HISTORY = 20

export const SttPlaygroundPage: React.FC = () => {
  const { t } = useTranslation(["playground", "settings"])
  const notification = useAntdNotification()

  // Server models
  const [serverModels, setServerModels] = React.useState<string[]>([])
  const [serverModelsLoading, setServerModelsLoading] = React.useState(false)

  // Current recording blob (from RecordingStrip)
  const [currentBlob, setCurrentBlob] = React.useState<Blob | null>(null)
  const [currentDurationMs, setCurrentDurationMs] = React.useState(0)

  // Inline settings
  const [sttSettings, setSttSettings] = React.useState<SttLocalSettings | null>(null)
  const [showSettings, setShowSettings] = React.useState(false)

  // History (metadata in Plasmo, blobs in Dexie)
  const [history, setHistory] = useStorage<SttHistoryEntry[]>("sttComparisonHistory", [])

  // Load models on mount
  React.useEffect(() => {
    let cancelled = false
    const fetchModels = async () => {
      setServerModelsLoading(true)
      try {
        const res = await tldwClient.getTranscriptionModels({ timeoutMs: 10_000 })
        const all = Array.isArray(res?.all_models) ? (res.all_models as string[]) : []
        if (!cancelled && all.length > 0) {
          setServerModels(Array.from(new Set(all)).sort())
        }
      } catch (e) {
        if (!cancelled) {
          notification.error({
            message: t("playground:stt.modelsLoadError", "Unable to load transcription models."),
            description: isTimeoutLikeError(e)
              ? t("playground:stt.modelsTimeout", "Timed out. Check server health.")
              : undefined
          })
        }
      } finally {
        if (!cancelled) setServerModelsLoading(false)
      }
    }
    fetchModels()
    return () => { cancelled = true }
  }, [notification, t])

  // Handle blob ready from RecordingStrip
  const handleBlobReady = React.useCallback((blob: Blob, durationMs: number) => {
    setCurrentBlob(blob)
    setCurrentDurationMs(durationMs)
  }, [])

  // Build sttOptions from local settings
  const sttOptions = React.useMemo(() => {
    if (!sttSettings) return {}
    const opts: Record<string, any> = {
      language: sttSettings.language,
      task: sttSettings.task,
      response_format: sttSettings.responseFormat,
      temperature: sttSettings.temperature
    }
    if (sttSettings.prompt?.trim()) opts.prompt = sttSettings.prompt.trim()
    if (sttSettings.useSegmentation) {
      opts.segment = true
      opts.seg_K = sttSettings.segK
      opts.seg_min_segment_size = sttSettings.segMinSegmentSize
      opts.seg_lambda_balance = sttSettings.segLambdaBalance
      opts.seg_utterance_expansion_width = sttSettings.segUtteranceExpansionWidth
      if (sttSettings.segEmbeddingsProvider?.trim()) {
        opts.seg_embeddings_provider = sttSettings.segEmbeddingsProvider.trim()
      }
      if (sttSettings.segEmbeddingsModel?.trim()) {
        opts.seg_embeddings_model = sttSettings.segEmbeddingsModel.trim()
      }
    }
    return opts
  }, [sttSettings])

  // Save to Notes
  const handleSaveToNotes = React.useCallback(
    async (text: string, model: string) => {
      const title = `STT Comparison: ${model} — ${new Date().toLocaleString()}`
      try {
        await tldwClient.createNote(text, {
          title,
          metadata: { origin: "stt-playground", stt_model: model }
        })
        notification.success({
          message: t("playground:tts.savedToNotes", "Saved to Notes")
        })
      } catch (e: any) {
        notification.error({
          message: t("error", "Error"),
          description: e?.message || t("somethingWentWrong", "Something went wrong")
        })
      }
    },
    [notification, t]
  )

  // Save comparison to history (persist blob to Dexie)
  const saveToHistory = React.useCallback(
    async (results: SttHistoryResult[]) => {
      if (!currentBlob || results.length === 0) return
      try {
        const recordingId = await saveSttRecording({
          blob: currentBlob,
          durationMs: currentDurationMs,
          mimeType: currentBlob.type || "audio/webm"
        })
        const entry: SttHistoryEntry = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          recordingId,
          createdAt: new Date().toISOString(),
          durationMs: currentDurationMs,
          results
        }
        setHistory((prev) => {
          const next = [entry, ...(prev || [])]
          return next.slice(0, MAX_HISTORY)
        })
      } catch (e: any) {
        console.error("Failed to save to history", e)
      }
    },
    [currentBlob, currentDurationMs, setHistory]
  )

  // Re-compare: load blob from Dexie back into recording strip
  const handleRecompare = React.useCallback(
    async (entry: SttHistoryEntry) => {
      try {
        const rec = await getSttRecording(entry.recordingId)
        if (!rec) {
          notification.error({
            message: t("playground:stt.recordingNotFound", "Recording not found in storage.")
          })
          return
        }
        setCurrentBlob(rec.blob)
        setCurrentDurationMs(rec.durationMs)
      } catch {
        notification.error({
          message: t("playground:stt.loadFailed", "Failed to load recording.")
        })
      }
    },
    [notification, t]
  )

  // Export entry as markdown to clipboard
  const handleExport = React.useCallback(
    async (entry: SttHistoryEntry) => {
      const lines = [
        `# STT Comparison — ${new Date(entry.createdAt).toLocaleString()}`,
        `Duration: ${(entry.durationMs / 1000).toFixed(1)}s\n`,
        ...entry.results.map(
          (r) =>
            `## ${r.model}\n${r.text}\n*${r.latencyMs ? `${(r.latencyMs / 1000).toFixed(1)}s` : "—"} · ${r.wordCount ?? "—"} words*`
        )
      ]
      try {
        await navigator.clipboard.writeText(lines.join("\n\n"))
        notification.success({
          message: t("playground:stt.exported", "Copied comparison to clipboard")
        })
      } catch {
        notification.error({
          message: t("playground:stt.exportFailed", "Copy failed")
        })
      }
    },
    [notification, t]
  )

  // Delete history entry
  const handleDeleteEntry = React.useCallback(
    async (id: string) => {
      const entry = (history || []).find((e) => e.id === id)
      if (entry) {
        try {
          await deleteSttRecording(entry.recordingId)
        } catch {}
      }
      setHistory((prev) => (prev || []).filter((e) => e.id !== id))
      notification.info({
        message: t("playground:stt.deleted", "Recording deleted")
      })
    },
    [history, setHistory, notification, t]
  )

  // Clear all history
  const handleClearAll = React.useCallback(async () => {
    for (const entry of history || []) {
      try {
        await deleteSttRecording(entry.recordingId)
      } catch {}
    }
    setHistory([])
  }, [history, setHistory])

  // Keyboard shortcuts
  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const isInput =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.tagName === "SELECT" ||
        target.isContentEditable
      if (e.key === " " && !isInput) {
        e.preventDefault()
        // Space toggles record — handled by RecordingStrip internally
        // We dispatch a custom event the strip can listen to
        window.dispatchEvent(new CustomEvent("stt-toggle-record"))
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [])

  return (
    <PageShell maxWidthClassName="max-w-5xl" className="py-6">
      <Title level={3} className="!mb-1">
        {t("playground:stt.title", "STT Playground")}
      </Title>
      <Text type="secondary">
        {t(
          "playground:stt.subtitle",
          "Record audio and compare transcription results across multiple models."
        )}
      </Text>

      <div className="mt-4 space-y-4">
        {/* Zone 1: Recording Strip */}
        <RecordingStrip
          onBlobReady={handleBlobReady}
          onSettingsToggle={() => setShowSettings((v) => !v)}
        />

        {/* Inline Settings (collapsible) */}
        {showSettings && <InlineSettingsPanel onChange={setSttSettings} />}

        {/* Zone 2: Comparison Panel */}
        <ComparisonPanel
          blob={currentBlob}
          availableModels={serverModels}
          sttOptions={sttOptions}
          onSaveToNotes={handleSaveToNotes}
        />

        {/* Zone 3: History */}
        <HistoryPanel
          entries={history || []}
          onRecompare={handleRecompare}
          onExport={handleExport}
          onDelete={handleDeleteEntry}
          onClearAll={handleClearAll}
        />
      </div>
    </PageShell>
  )
}

export default SttPlaygroundPage
```

**Step 4: Run tests to verify they pass**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx \
       apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx
git commit -m "feat(stt): rewrite SttPlaygroundPage with comparison-first three-zone layout"
```

---

## Task 9: Add keyboard shortcuts and accessibility polish

**Files:**
- Modify: `apps/packages/ui/src/components/Option/STT/RecordingStrip.tsx`
- Modify: `apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/STT/__tests__/keyboard-shortcuts.test.tsx`

**Context:** Add Space-to-record listener in RecordingStrip (via custom event from page), Cmd+Enter to trigger Transcribe All in ComparisonPanel. Verify ARIA attributes on key elements.

**Step 1: Write failing tests**

```typescript
// apps/packages/ui/src/components/Option/STT/__tests__/keyboard-shortcuts.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"

const mockStartRecording = vi.fn()
const mockStopRecording = vi.fn()

vi.mock("@/hooks/useAudioRecorder", () => ({
  useAudioRecorder: () => ({
    status: "idle",
    blob: null,
    durationMs: 0,
    startRecording: mockStartRecording,
    stopRecording: mockStopRecording,
    clearRecording: vi.fn(),
    loadBlob: vi.fn()
  })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback: string) => fallback
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    success: vi.fn()
  })
}))

import { RecordingStrip } from "../RecordingStrip"

describe("keyboard shortcuts", () => {
  beforeEach(() => vi.clearAllMocks())

  it("starts recording on stt-toggle-record custom event", () => {
    render(<RecordingStrip onBlobReady={vi.fn()} />)
    window.dispatchEvent(new CustomEvent("stt-toggle-record"))
    expect(mockStartRecording).toHaveBeenCalled()
  })

  it("record button has correct aria-label", () => {
    render(<RecordingStrip onBlobReady={vi.fn()} />)
    expect(screen.getByRole("button", { name: /record/i })).toHaveAttribute(
      "aria-label"
    )
  })
})
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/keyboard-shortcuts.test.tsx`
Expected: FAIL — RecordingStrip doesn't listen to custom event yet

**Step 3: Add custom event listener to RecordingStrip**

In `RecordingStrip.tsx`, add inside the component body:

```typescript
// Listen for Space keyboard shortcut from parent page
React.useEffect(() => {
  const handler = () => {
    if (recorder.status === "recording") {
      recorder.stopRecording()
    } else {
      recorder.startRecording().catch(() => {
        notification.error({
          message: t("playground:actions.speechErrorTitle", "Recording failed"),
          description: t("playground:actions.speechMicError", "Unable to access microphone.")
        })
      })
    }
  }
  window.addEventListener("stt-toggle-record", handler)
  return () => window.removeEventListener("stt-toggle-record", handler)
}, [recorder, notification, t])
```

**Step 4: Add Cmd+Enter handler to ComparisonPanel**

In `ComparisonPanel.tsx`, add:

```typescript
React.useEffect(() => {
  const handler = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault()
      if (blob && models.length > 0 && !isRunning) {
        handleTranscribeAll()
      }
    }
  }
  window.addEventListener("keydown", handler)
  return () => window.removeEventListener("keydown", handler)
}, [blob, models, isRunning, handleTranscribeAll])
```

**Step 5: Run tests to verify they pass**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/__tests__/keyboard-shortcuts.test.tsx`
Expected: 2 tests PASS

**Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/STT/RecordingStrip.tsx \
       apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx \
       apps/packages/ui/src/components/Option/STT/__tests__/keyboard-shortcuts.test.tsx
git commit -m "feat(stt): add Space and Cmd+Enter keyboard shortcuts with ARIA labels"
```

---

## Task 10: Run full test suite and fix any regressions

**Files:**
- Possibly modify: any files from Tasks 1-9 if tests fail

**Step 1: Run all STT tests**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/STT/ src/hooks/__tests__/useAudioRecorder.test.ts src/hooks/__tests__/useComparisonTranscribe.test.ts src/db/dexie/__tests__/stt-recordings.test.ts`
Expected: All tests PASS

**Step 2: Run the existing E2E STT test to check for breakage**

Run: `cd apps/extension && npx playwright test tests/e2e/stt-playground.spec.ts --reporter=line` (if Playwright is configured)
Expected: Investigate failures — E2E tests may need selector updates for the new layout

**Step 3: Fix any failures**

Address test failures one by one. Common issues:
- Selector changes (old selectors like "Record" button text should still match)
- Mock shape changes (if E2E tests mock API calls)
- Import path updates

**Step 4: Commit fixes**

```bash
git add -u
git commit -m "fix(stt): resolve test regressions from playground redesign"
```

---

## Summary

| Task | Component | New Files | Key Dependency |
|------|-----------|-----------|----------------|
| 1 | Dexie STT table | `stt-recordings.ts` + test | — |
| 2 | useAudioRecorder | hook + test | — |
| 3 | useComparisonTranscribe | hook + test | TldwApiClient |
| 4 | RecordingStrip | component + test | Task 2 |
| 5 | ComparisonPanel | component + test | Task 3 |
| 6 | InlineSettingsPanel | component + test | — |
| 7 | HistoryPanel | component + test | — |
| 8 | SttPlaygroundPage rewrite | rewrite + test | Tasks 1-7 |
| 9 | Keyboard shortcuts + a11y | modifications + test | Tasks 4, 5 |
| 10 | Regression testing | fixes only | All |

Tasks 1-3 can be done in parallel (no dependencies). Tasks 4-7 can be done in parallel after 1-3. Task 8 depends on all. Task 9 depends on 8. Task 10 is final.
