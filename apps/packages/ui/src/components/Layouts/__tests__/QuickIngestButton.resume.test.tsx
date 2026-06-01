// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

import {
  createInitialQuickIngestLastRunSummary,
  useQuickIngestStore,
} from "@/store/quick-ingest"
import {
  createEmptyQuickIngestSession,
  useQuickIngestSessionStore,
} from "@/store/quick-ingest-session"
import {
  QuickIngestButton,
  QuickIngestModalHost,
} from "../QuickIngestButton"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [k: string]: unknown
          },
      interpolation?: Record<string, unknown>
    ) => {
      if (typeof defaultValueOrOptions === "string") {
        return defaultValueOrOptions.replace(/\{\{(\w+)\}\}/g, (_m, token) =>
          String(interpolation?.[token] ?? "")
        )
      }
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_m, token) => String(defaultValueOrOptions?.[token] ?? interpolation?.[token] ?? "")
        )
      }
      return key
    },
  }),
}))

vi.mock("lucide-react", () => ({
  UploadCloud: () => <span data-testid="upload-cloud" />,
}))

vi.mock("@/components/Common/QuickIngestWizardModal", () => ({
  QuickIngestWizardModal: ({
    open,
    autoProcessQueued,
    onClose,
  }: {
    open: boolean
    autoProcessQueued?: boolean
    onClose: () => void
  }) => (
    <div
      data-testid="quick-ingest-modal-mock"
      data-open={open ? "true" : "false"}
      data-auto-process={autoProcessQueued ? "true" : "false"}
    >
      <button type="button" onClick={onClose}>
        close-modal
      </button>
    </div>
  ),
}))

describe("QuickIngestButton resume behavior", () => {
  beforeEach(async () => {
    sessionStorage.clear()
    useQuickIngestStore.setState((prev) => ({
      ...prev,
      queuedCount: 0,
      hadRecentFailure: false,
      lastRunSummary: createInitialQuickIngestLastRunSummary(),
    }))
    useQuickIngestSessionStore.setState({
      session: null,
      triggerSummary: { count: 0, label: null, hadFailure: false },
    })
    if (useQuickIngestSessionStore.persist?.clearStorage) {
      await useQuickIngestSessionStore.persist.clearStorage()
    }
  })

  it("reopens an existing hidden session instead of creating a new one", async () => {
    const user = userEvent.setup()
    const session = createEmptyQuickIngestSession()

    useQuickIngestSessionStore.getState().upsertSession({
      id: session.id,
      lifecycle: "processing",
      visibility: "hidden",
      badge: {
        queueCount: 0,
        hasRecentFailure: false,
      },
    })

    render(<QuickIngestButton />)

    expect(screen.queryByTestId("process-queued-ingest-header")).not.toBeInTheDocument()

    await user.click(screen.getByTestId("open-quick-ingest"))

    expect(useQuickIngestSessionStore.getState().session?.id).toBe(session.id)
    expect(useQuickIngestSessionStore.getState().session?.visibility).toBe("visible")
    expect(screen.getByTestId("quick-ingest-modal-mock")).toHaveAttribute(
      "data-open",
      "true"
    )
  })

  it("shows the secondary CTA only for draft sessions with queued items", async () => {
    const user = userEvent.setup()

    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "draft",
      visibility: "hidden",
      badge: {
        queueCount: 2,
        hasRecentFailure: false,
      },
      queueItems: [
        {
          id: "queued-url-1",
          kind: "url",
          url: "https://example.com/article",
          detectedType: "web",
          icon: "Globe",
          fileSize: 0,
          validation: { valid: true },
        },
      ] as any,
    })

    render(<QuickIngestButton />)

    const cta = screen.getByTestId("process-queued-ingest-header")
    expect(cta).toBeVisible()

    await user.click(cta)

    expect(useQuickIngestSessionStore.getState().session?.visibility).toBe("visible")
    expect(screen.getByTestId("quick-ingest-modal-mock")).toHaveAttribute(
      "data-auto-process",
      "true"
    )
  })

  it("keeps the modal host mounted while a resumable session exists", () => {
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "processing",
      visibility: "hidden",
      badge: {
        queueCount: 0,
        hasRecentFailure: false,
      },
    })

    render(<QuickIngestModalHost />)

    expect(screen.getByTestId("quick-ingest-modal-mock")).toHaveAttribute(
      "data-open",
      "false"
    )
  })

  it("opens when a pending quick-ingest request exists before the host mounts", () => {
    ;(window as typeof window & {
      __tldwPendingQuickIngestOpen?: { mode: "normal" | "intro"; at: number }
    }).__tldwPendingQuickIngestOpen = {
      mode: "normal",
      at: Date.now(),
    }

    render(<QuickIngestModalHost />)

    expect(screen.getByTestId("quick-ingest-modal-mock")).toHaveAttribute(
      "data-open",
      "true"
    )
  })

  it("hydrates first-source pending opens into the quick first-source preset", () => {
    ;(window as typeof window & {
      __tldwPendingQuickIngestOpen?: {
        mode: "normal" | "intro"
        at: number
        detail: {
          source: "first_source_milestone"
          preferredPreset: "quick"
          firstSource: true
        }
      }
    }).__tldwPendingQuickIngestOpen = {
      mode: "normal",
      at: Date.now(),
      detail: {
        source: "first_source_milestone",
        preferredPreset: "quick",
        firstSource: true,
      },
    }

    render(<QuickIngestModalHost />)

    const session = useQuickIngestSessionStore.getState().session
    expect(session?.selectedPreset).toBe("quick")
    expect(session?.customBasePreset).toBe("quick")
    expect(session?.presetConfig.storeRemote).toBe(true)
    expect(session?.presetConfig.reviewBeforeStorage).toBe(false)
    expect(session?.presetConfig.common.perform_analysis).toBe(false)
    expect(session?.presetConfig.common.perform_chunking).toBe(true)
    expect(session?.presetConfig.typeDefaults.document?.ocr).toBe(false)
  })
})
