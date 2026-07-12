// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, render, screen } from "@testing-library/react"
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
import { resolvePresetMap } from "@/components/Common/QuickIngest/presets"

const presetStorage = vi.hoisted(() => ({
  value: undefined as ReturnType<typeof resolvePresetMap> | undefined,
  isLoading: false,
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [
    presetStorage.value,
    vi.fn(),
    { isLoading: presetStorage.isLoading, setRenderValue: vi.fn() },
  ],
}))

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
    openRevision,
    createNewDraft,
  }: {
    open: boolean
    autoProcessQueued?: boolean
    onClose: () => void
    openRevision?: number
    createNewDraft?: () => void
  }) => (
    <div
      data-testid="quick-ingest-modal-mock"
      data-open={open ? "true" : "false"}
      data-auto-process={autoProcessQueued ? "true" : "false"}
      data-open-revision={String(openRevision ?? "")}
    >
      <button type="button" onClick={onClose}>
        close-modal
      </button>
      <button type="button" onClick={() => createNewDraft?.()}>
        new-draft
      </button>
    </div>
  ),
}))

describe("QuickIngestButton resume behavior", () => {
  beforeEach(async () => {
    sessionStorage.clear()
    ;(window as typeof window & {
      __tldwPendingQuickIngestOpen?: unknown
    }).__tldwPendingQuickIngestOpen = undefined
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
    presetStorage.value = resolvePresetMap()
    presetStorage.isLoading = false
  })

  it("waits for preset storage before consuming a pending open", () => {
    presetStorage.isLoading = true
    ;(window as typeof window & {
      __tldwPendingQuickIngestOpen?: { mode: "normal" | "intro"; at: number }
    }).__tldwPendingQuickIngestOpen = {
      mode: "normal",
      at: Date.now(),
    }

    render(<QuickIngestModalHost />)

    expect(screen.queryByTestId("quick-ingest-modal-mock")).not.toBeInTheDocument()
    expect(useQuickIngestSessionStore.getState().session).toBeNull()
  })

  it("rebases a rehydrated visible draft only after preset storage is ready", async () => {
    presetStorage.isLoading = true
    presetStorage.value = resolvePresetMap({
      standard: {
        ...resolvePresetMap().standard,
        advancedValues: { api_name: "openai" },
      },
    })
    useQuickIngestSessionStore.getState().upsertSession({
      ...createEmptyQuickIngestSession(),
      lifecycle: "draft",
      visibility: "visible",
    })

    const { rerender } = render(<QuickIngestModalHost />)
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })

    expect(screen.queryByTestId("quick-ingest-modal-mock")).not.toBeInTheDocument()
    expect(
      useQuickIngestSessionStore.getState().session?.presetConfig.advancedValues
        ?.api_name
    ).toBeUndefined()

    presetStorage.isLoading = false
    rerender(<QuickIngestModalHost />)

    expect(await screen.findByTestId("quick-ingest-modal-mock")).toBeInTheDocument()
    expect(
      useQuickIngestSessionStore.getState().session?.presetConfig.advancedValues
        ?.api_name
    ).toBe("openai")
  })

  it("creates a regular draft from the captured saved preset", () => {
    presetStorage.value = resolvePresetMap({
      standard: {
        ...resolvePresetMap().standard,
        advancedValues: { api_name: "openai" },
      },
    })
    ;(window as typeof window & {
      __tldwPendingQuickIngestOpen?: { mode: "normal" | "intro"; at: number }
    }).__tldwPendingQuickIngestOpen = {
      mode: "normal",
      at: Date.now(),
    }

    render(<QuickIngestModalHost />)

    expect(
      useQuickIngestSessionStore.getState().session?.presetConfig.advancedValues
        ?.api_name
    ).toBe("openai")
  })

  it("captures current settings when Ingest More creates a new draft", async () => {
    const user = userEvent.setup()
    const completed = createEmptyQuickIngestSession()
    useQuickIngestSessionStore.getState().upsertSession({
      ...completed,
      lifecycle: "completed",
      visibility: "visible",
      selectedPreset: "standard",
      customBasePreset: "standard",
      presetConfig: {
        ...resolvePresetMap().standard,
        advancedValues: { api_name: "session-only" },
      },
    })
    const { rerender } = render(<QuickIngestModalHost />)

    presetStorage.value = resolvePresetMap({
      standard: {
        ...resolvePresetMap().standard,
        advancedValues: { api_name: "anthropic" },
      },
    })
    rerender(<QuickIngestModalHost />)
    await user.click(await screen.findByText("new-draft"))

    const next = useQuickIngestSessionStore.getState().session
    expect(next?.id).not.toBe(completed.id)
    expect(next?.presetConfig.advancedValues?.api_name).toBe("anthropic")
  })

  it.each([
    ["processing", "standard", null],
    ["interrupted", "standard", null],
    ["cancelled", "standard", null],
    ["partial_failure", "standard", null],
    ["completed", "standard", null],
    ["draft", "custom", null],
    ["draft", "quick", "web_url"],
  ] as const)(
    "preserves %s %s session configuration when opening",
    async (lifecycle, selectedPreset, firstSourceAddMode) => {
      presetStorage.value = resolvePresetMap({
        standard: {
          ...resolvePresetMap().standard,
          advancedValues: { api_name: "new-default" },
        },
        quick: {
          ...resolvePresetMap().quick,
          advancedValues: { api_name: "new-quick-default" },
        },
      })
      const existing = createEmptyQuickIngestSession()
      useQuickIngestSessionStore.getState().upsertSession({
        ...existing,
        lifecycle,
        visibility: "visible",
        selectedPreset,
        customBasePreset: selectedPreset === "custom" ? "standard" : selectedPreset,
        firstSourceAddMode,
        presetConfig: {
          ...(selectedPreset === "quick"
            ? resolvePresetMap().quick
            : resolvePresetMap().standard),
          advancedValues: { api_name: "session-provider" },
        },
      })

      render(<QuickIngestModalHost />)
      await screen.findByTestId("quick-ingest-modal-mock")

      expect(
        useQuickIngestSessionStore.getState().session?.presetConfig.advancedValues
          ?.api_name
      ).toBe("session-provider")
    }
  )

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
    presetStorage.value = resolvePresetMap({
      quick: {
        ...resolvePresetMap().quick,
        advancedValues: { api_name: "saved-quick-provider" },
      },
    })
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
