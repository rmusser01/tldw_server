import React from "react"
import { useTranslation } from "react-i18next"
import { useParams } from "react-router-dom"

import { Button } from "@/components/Common/Button"
import { PageShell } from "@/components/Common/PageShell"
import { Badge, LoadingState, StatePanel } from "@/components/ui"
import { useConnectionState } from "@/hooks/useConnectionState"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useServerOnline } from "@/hooks/useServerOnline"
import { getScreenshotFromCurrentTab } from "@/libs/get-screenshot"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { resolveSidepanelChatWebUiBaseUrl } from "@/services/tldw/sidepanel-chat-webui-handoff"
import { EMPTY_STATE_LABEL, READY_STATE_LABEL } from "@/design-system"

type SeedImage = {
  dataB64: string
  mime: string
  alt: string
  previewUrl: string
}

const createSlideId = (): string =>
  globalThis.crypto?.randomUUID?.() ||
  `slide-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`

const readFileAsDataUrl = async (file: File): Promise<string> =>
  await new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ""))
    reader.onerror = () => reject(new Error("Failed to read image file."))
    reader.readAsDataURL(file)
  })

const parseImageDataUrl = (
  value: string,
  fallbackAlt: string
): Pick<SeedImage, "dataB64" | "mime" | "alt" | "previewUrl"> | null => {
  const match = /^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)$/i.exec(value.trim())
  if (!match) {
    return null
  }
  return {
    mime: match[1].toLowerCase(),
    dataB64: match[2],
    alt: fallbackAlt,
    previewUrl: value
  }
}

const resolveServerOrigin = (serverUrl: string | null | undefined): string | null => {
  if (!serverUrl) {
    return null
  }
  try {
    const parsed = new URL(serverUrl)
    return parsed.protocol === "http:" || parsed.protocol === "https:"
      ? parsed.origin
      : null
  } catch {
    return null
  }
}

const AUTHORITY_EVENTS = [
  "tldw:config-updated",
  "tldw:auth-principal-changed",
  "tldw:slides-scope-mismatch"
] as const

const MAX_PROJECT_ID_SCALARS = 256
const MAX_METADATA_KIND_SCALARS = 256
const MAX_METADATA_PROVENANCE_SCALARS = 256
const MAX_METADATA_TITLE_SCALARS = 512
const MAX_METADATA_DESCRIPTION_SCALARS = 2_048

type SafeProvenance = {
  sourceKind: string | null
  provider: string | null
  model: string | null
}

type SafePresentationMetadata = {
  id: string
  title: string
  description: string | null
  provenance: SafeProvenance
} & (
  | {
      contentKind: "structured_slides"
      slideCount: number
    }
  | {
      contentKind: "standalone_html"
      slideCount: number
      htmlBytes: number
    }
  | {
      contentKind: "unsupported"
      unsupportedKind: string
    }
)

type MetadataView =
  | { projectId: string; status: "loading" }
  | { projectId: string; status: "load_error" }
  | { projectId: string; status: "invalid" }
  | { projectId: string; status: "ready"; record: SafePresentationMetadata }

type TrustedReadyMetadata = {
  projectId: string
  metadataEpoch: number
  record: SafePresentationMetadata
}

type SourceFreeWebUiConfig = {
  serverUrl?: string | null
  webUiUrl?: string | null
  webuiUrl?: string | null
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isBidiControl = (codePoint: number): boolean =>
  codePoint === 0x061c ||
  codePoint === 0x200e ||
  codePoint === 0x200f ||
  (codePoint >= 0x202a && codePoint <= 0x202e) ||
  (codePoint >= 0x2066 && codePoint <= 0x206f)

const isBoundedMetadataString = (
  value: unknown,
  maximumScalars: number,
  required: boolean
): value is string => {
  if (typeof value !== "string") return false
  if (required && value.trim().length === 0) return false

  let scalars = 0
  for (let index = 0; index < value.length; index += 1) {
    const first = value.charCodeAt(index)
    let codePoint = first
    if (first >= 0xd800 && first <= 0xdbff) {
      const second = value.charCodeAt(index + 1)
      if (!Number.isInteger(second) || second < 0xdc00 || second > 0xdfff) {
        return false
      }
      codePoint = 0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
      index += 1
    } else if (first >= 0xdc00 && first <= 0xdfff) {
      return false
    }

    if (
      codePoint <= 0x1f ||
      (codePoint >= 0x7f && codePoint <= 0x9f) ||
      isBidiControl(codePoint)
    ) {
      return false
    }
    scalars += 1
    if (scalars > maximumScalars) return false
  }
  return true
}

const isTrustedProjectId = (value: unknown): value is string =>
  isBoundedMetadataString(value, MAX_PROJECT_ID_SCALARS, true) &&
  value !== "." &&
  value !== ".."

const readNullableMetadataString = (
  value: unknown,
  maximumScalars: number
): string | null | undefined => {
  if (value === null) return null
  return isBoundedMetadataString(value, maximumScalars, false)
    ? value
    : undefined
}

const isBoundedCount = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= 0

const projectPresentationMetadata = (
  response: unknown,
  routeProjectId: string
): SafePresentationMetadata | null => {
  if (!isRecord(response) || !isRecord(response.record)) return null
  const record = response.record
  if (
    !isTrustedProjectId(record.id) ||
    record.id !== routeProjectId ||
    !isBoundedMetadataString(record.title, MAX_METADATA_TITLE_SCALARS, true) ||
    !isBoundedMetadataString(record.theme, MAX_METADATA_KIND_SCALARS, true) ||
    !isBoundedMetadataString(record.created_at, MAX_METADATA_PROVENANCE_SCALARS, true) ||
    !isBoundedMetadataString(record.last_modified, MAX_METADATA_PROVENANCE_SCALARS, true) ||
    typeof record.deleted !== "boolean" ||
    !isBoundedCount(record.version) ||
    !isRecord(record.provenance)
  ) {
    return null
  }

  const description = readNullableMetadataString(
    record.description,
    MAX_METADATA_DESCRIPTION_SCALARS
  )
  const sourceKind = readNullableMetadataString(
    record.provenance.source_kind,
    MAX_METADATA_PROVENANCE_SCALARS
  )
  const provider = readNullableMetadataString(
    record.provenance.provider,
    MAX_METADATA_PROVENANCE_SCALARS
  )
  const model = readNullableMetadataString(
    record.provenance.model,
    MAX_METADATA_PROVENANCE_SCALARS
  )
  if (
    description === undefined ||
    sourceKind === undefined ||
    provider === undefined ||
    model === undefined ||
    !isBoundedMetadataString(
      record.content_kind,
      MAX_METADATA_KIND_SCALARS,
      true
    )
  ) {
    return null
  }

  const base = {
    id: record.id,
    title: record.title,
    description,
    provenance: { sourceKind, provider, model }
  }

  if (record.content_kind === "structured_slides") {
    return isBoundedCount(record.slide_count)
      ? { ...base, contentKind: "structured_slides", slideCount: record.slide_count }
      : null
  }
  if (record.content_kind === "standalone_html") {
    return isBoundedCount(record.html_slide_count) && isBoundedCount(record.html_bytes)
      ? {
          ...base,
          contentKind: "standalone_html",
          slideCount: record.html_slide_count,
          htmlBytes: record.html_bytes
        }
      : null
  }
  if (record.content_kind === "unsupported") {
    return record.read_only === true &&
      isBoundedMetadataString(
        record.unsupported_content_kind,
        MAX_METADATA_KIND_SCALARS,
        true
      )
      ? {
          ...base,
          contentKind: "unsupported",
          unsupportedKind: record.unsupported_content_kind
        }
      : null
  }
  return null
}

const isConfiguredHttpUrl = (value: unknown): value is string => {
  if (typeof value !== "string" || value.trim().length === 0) return false
  try {
    const parsed = new URL(value)
    return parsed.protocol === "http:" || parsed.protocol === "https:"
  } catch {
    return false
  }
}

const projectSourceFreeWebUiConfig = (
  value: unknown
): SourceFreeWebUiConfig | null => {
  if (!isRecord(value)) return null
  const config: SourceFreeWebUiConfig = {}
  for (const key of ["serverUrl", "webUiUrl", "webuiUrl"] as const) {
    const candidate = value[key]
    if (typeof candidate === "string") {
      config[key] = candidate
    } else if (candidate === null) {
      config[key] = null
    }
  }
  return Object.values(config).some(isConfiguredHttpUrl) ? config : null
}

const buildPresentationWebUiTarget = (
  rawConfig: unknown,
  trustedProjectId: string
): string | null => {
  if (!isTrustedProjectId(trustedProjectId)) return null
  const config = projectSourceFreeWebUiConfig(rawConfig)
  if (!config) return null
  const base = resolveSidepanelChatWebUiBaseUrl(config)
  try {
    const baseUrl = new URL(`${base.replace(/\/+$/, "")}/`)
    if (baseUrl.protocol !== "http:" && baseUrl.protocol !== "https:") return null
    return new URL(
      `presentation-studio/${encodeURIComponent(trustedProjectId)}`,
      baseUrl
    ).toString()
  } catch {
    return null
  }
}

const useAuthorityFence = (onInvalidate: () => void) => {
  const mountedRef = React.useRef(true)
  const epochRef = React.useRef(0)
  const onInvalidateRef = React.useRef(onInvalidate)
  const [boundaryEpoch, setBoundaryEpoch] = React.useState(0)
  onInvalidateRef.current = onInvalidate

  React.useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      epochRef.current += 1
    }
  }, [])

  React.useEffect(() => {
    const invalidate = () => {
      epochRef.current += 1
      onInvalidateRef.current()
      setBoundaryEpoch((current) => current + 1)
    }
    for (const eventName of AUTHORITY_EVENTS) {
      window.addEventListener(eventName, invalidate)
    }
    return () => {
      for (const eventName of AUTHORITY_EVENTS) {
        window.removeEventListener(eventName, invalidate)
      }
    }
  }, [])

  const isCurrent = React.useCallback(
    (epoch: number) => mountedRef.current && epochRef.current === epoch,
    []
  )
  const capture = React.useCallback(() => epochRef.current, [])
  const retire = React.useCallback(() => {
    epochRef.current += 1
  }, [])

  return { boundaryEpoch, capture, isCurrent, retire }
}

const getActiveTabTitle = async (): Promise<string | null> => {
  try {
    if (typeof browser !== "undefined" && browser.tabs?.query) {
      const tabs = await browser.tabs.query({ active: true, currentWindow: true })
      const title = tabs[0]?.title
      return typeof title === "string" && title.trim().length > 0 ? title.trim() : null
    }
  } catch {
    // ignore browser runtime title lookup failures
  }

  try {
    if (typeof chrome !== "undefined" && chrome.tabs?.query) {
      return await new Promise((resolve) => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          const title = tabs?.[0]?.title
          resolve(typeof title === "string" && title.trim().length > 0 ? title.trim() : null)
        })
      })
    }
  } catch {
    // ignore chrome runtime title lookup failures
  }

  return null
}

export const ExtensionStartPanel: React.FC = () => {
  const { t } = useTranslation("option")
  const isOnline = useServerOnline()
  const { capabilities, loading } = useServerCapabilities()
  const { serverUrl } = useConnectionState()
  const [projectTitle, setProjectTitle] = React.useState("")
  const [narrationSeed, setNarrationSeed] = React.useState("")
  const [imageSeed, setImageSeed] = React.useState<SeedImage | null>(null)
  const [captureError, setCaptureError] = React.useState<string | null>(null)
  const [submitError, setSubmitError] = React.useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = React.useState(false)
  const authority = useAuthorityFence(() => {
    setIsSubmitting(false)
    setSubmitError(null)
  })

  React.useEffect(() => {
    let active = true
    void getActiveTabTitle().then((title) => {
      if (!active || !title) {
        return
      }
      setProjectTitle((previous) => (previous.trim().length > 0 ? previous : title))
    })
    return () => {
      active = false
    }
  }, [])

  const hasSeedContent = narrationSeed.trim().length > 0 || imageSeed !== null
  const serverOrigin = resolveServerOrigin(serverUrl)
  const readyStateLabel = t("presentationStudio.start.status.ready", READY_STATE_LABEL)
  const emptyStateLabel = t("presentationStudio.start.status.empty", EMPTY_STATE_LABEL)

  const handleImageFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ): Promise<void> => {
    const file = event.target.files?.[0]
    if (!file) {
      setImageSeed(null)
      return
    }
    if (!file.type.startsWith("image/")) {
      setCaptureError("Select an image file to seed the first slide.")
      return
    }

    try {
      const parsed = parseImageDataUrl(
        await readFileAsDataUrl(file),
        projectTitle.trim() || file.name || "Seed image"
      )
      if (!parsed) {
        setCaptureError("Failed to parse the selected image.")
        return
      }
      setCaptureError(null)
      setImageSeed(parsed)
    } catch (error) {
      setCaptureError(error instanceof Error ? error.message : "Failed to read image file.")
    } finally {
      event.target.value = ""
    }
  }

  const handleCaptureScreenshot = async (): Promise<void> => {
    const result = await getScreenshotFromCurrentTab()
    if (!result.success || !result.screenshot) {
      setCaptureError(result.error || "Failed to capture screenshot.")
      return
    }
    const parsed = parseImageDataUrl(
      result.screenshot,
      projectTitle.trim() || "Current tab screenshot"
    )
    if (!parsed) {
      setCaptureError("Failed to parse the captured screenshot.")
      return
    }
    setCaptureError(null)
    setImageSeed(parsed)
  }

  const createProject = async (mode: "blank" | "seeded"): Promise<void> => {
    if (!serverOrigin) {
      setSubmitError("Configure your server URL under Settings → tldw server first.")
      return
    }

    if (mode === "seeded" && !hasSeedContent) {
      setSubmitError("Add narration or an image before creating a seeded project.")
      return
    }

    const finalTitle = projectTitle.trim() || "Untitled Presentation"
    const slideId = createSlideId()
    const metadata: Record<string, unknown> = {
      studio: {
        slideId,
        transition: "fade",
        timing_mode: "auto",
        manual_duration_ms: null,
        audio: { status: "missing" },
        image: { status: imageSeed ? "ready" : "missing" }
      }
    }

    if (imageSeed) {
      metadata.images = [
        {
          id: `${slideId}-image-1`,
          mime: imageSeed.mime,
          data_b64: imageSeed.dataB64,
          alt: imageSeed.alt
        }
      ]
    }

    setIsSubmitting(true)
    setSubmitError(null)
    const operationEpoch = authority.capture()
    let projectResponse: unknown = null
    let configResponse: unknown = null
    try {
      projectResponse = await tldwClient.createPresentation({
        title: finalTitle,
        description: null,
        theme: "black",
        studio_data: {
          origin: mode === "seeded" ? "extension_capture" : "blank",
          entry_surface: "extension_start",
          has_narration_seed: narrationSeed.trim().length > 0,
          has_image_seed: Boolean(imageSeed)
        },
        slides: [
          {
            order: 0,
            layout: mode === "seeded" ? "content" : "title",
            title: finalTitle,
            content: "",
            speaker_notes: mode === "seeded" ? narrationSeed.trim() : "",
            metadata
          }
        ]
      })
      if (!authority.isCurrent(operationEpoch)) return
      if (
        !isRecord(projectResponse) ||
        !isTrustedProjectId(projectResponse.id)
      ) {
        projectResponse = null
        setSubmitError("Presentation project ID could not be verified")
        return
      }
      const trustedProjectId = projectResponse.id
      projectResponse = null

      configResponse = await tldwClient.getConfig()
      if (!authority.isCurrent(operationEpoch)) return
      const destination = buildPresentationWebUiTarget(
        configResponse,
        trustedProjectId
      )
      configResponse = null
      if (!destination) {
        setSubmitError("A valid WebUI address is not configured")
        return
      }
      if (!authority.isCurrent(operationEpoch)) return
      window.open(destination, "_blank", "noopener,noreferrer")
    } catch {
      if (authority.isCurrent(operationEpoch)) {
        setSubmitError("Failed to create project.")
      }
    } finally {
      projectResponse = null
      configResponse = null
      if (authority.isCurrent(operationEpoch)) {
        setIsSubmitting(false)
      }
    }
  }

  if (!isOnline) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Presentation Studio Quick Start</h1>
        <p className="mt-2 text-sm text-slate-600">
          Server is offline. Connect to seed a Presentation Studio project from the extension.
        </p>
      </section>
    )
  }

  if (!loading && capabilities && !capabilities.hasPresentationStudio) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Presentation Studio Quick Start</h1>
        <p className="mt-2 text-sm text-slate-600">
          Presentation Studio is not available on this server.
        </p>
      </section>
    )
  }

  return (
    <section className="space-y-6">
      <header className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Presentation Studio Quick Start</h1>
        <p className="mt-2 max-w-2xl text-sm text-slate-600">
          Start a blank narrated deck or seed the first slide with narration and an image,
          then continue editing in the full WebUI studio.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <div className="space-y-4">
            <div>
              <label
                className="mb-2 block text-sm font-medium text-slate-900"
                htmlFor="presentation-studio-start-title"
              >
                Project title
              </label>
              <input
                id="presentation-studio-start-title"
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900"
                value={projectTitle}
                onChange={(event) => setProjectTitle(event.target.value)}
                placeholder="Quarterly product review"
              />
            </div>

            <div>
              <label
                className="mb-2 block text-sm font-medium text-slate-900"
                htmlFor="presentation-studio-start-narration"
              >
                Narration seed
              </label>
              <textarea
                id="presentation-studio-start-narration"
                className="min-h-[160px] w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900"
                value={narrationSeed}
                onChange={(event) => setNarrationSeed(event.target.value)}
                placeholder="Paste selected text, opening narration, or the first-slide talking points."
              />
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <div className="flex flex-wrap items-center gap-3">
                <label className="inline-flex cursor-pointer items-center rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700">
                  <span>Upload image</span>
                  <input
                    className="sr-only"
                    type="file"
                    accept="image/*"
                    onChange={(event) => {
                      void handleImageFileChange(event)
                    }}
                  />
                </label>
                <button
                  type="button"
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
                  onClick={() => {
                    void handleCaptureScreenshot()
                  }}
                >
                  Use current tab screenshot
                </button>
                {imageSeed ? (
                  <button
                    type="button"
                    className="rounded-lg border border-transparent px-3 py-2 text-sm text-slate-500 hover:text-slate-700"
                    onClick={() => setImageSeed(null)}
                  >
                    Clear image
                  </button>
                ) : null}
              </div>

              {captureError ? (
                <p className="mt-3 text-sm text-rose-600">{captureError}</p>
              ) : null}

              {imageSeed ? (
                <div className="mt-4 overflow-hidden rounded-lg border border-slate-200 bg-white">
                  <img
                    alt={imageSeed.alt}
                    className="h-40 w-full object-cover"
                    src={imageSeed.previewUrl}
                  />
                </div>
              ) : (
                <p className="mt-3 text-sm text-slate-500">
                  Add an image or screenshot if you want the first slide seeded visually.
                </p>
              )}
            </div>
          </div>
        </section>

        <aside className="rounded-xl border border-slate-200 bg-white p-6">
          <h2 className="text-lg font-semibold text-slate-900">Launch options</h2>
          <p className="mt-2 text-sm text-slate-600">
            The extension creates a server-backed project first, then opens the WebUI editor
            at the matching project ID.
          </p>

          <div className="mt-4 space-y-3">
            <button
              type="button"
              className="w-full rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
              disabled={isSubmitting}
              onClick={() => {
                void createProject("blank")
              }}
            >
              Create blank project
            </button>
            <button
              type="button"
              className="w-full rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-900 hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-400"
              disabled={isSubmitting || !hasSeedContent}
              onClick={() => {
                void createProject("seeded")
              }}
            >
              Create seeded project
            </button>
          </div>

          <dl className="mt-5 space-y-3 text-sm">
            <div className="flex items-center justify-between gap-3">
              <dt className="text-slate-500">Server</dt>
              <dd className="truncate text-right text-slate-700">
                {serverOrigin || "Not configured"}
              </dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="text-slate-500">Narration seed</dt>
              <dd className="text-slate-700">
                {narrationSeed.trim().length > 0 ? readyStateLabel : emptyStateLabel}
              </dd>
            </div>
            <div className="flex items-center justify-between gap-3">
              <dt className="text-slate-500">Image seed</dt>
              <dd className="text-slate-700">
                {imageSeed ? readyStateLabel : emptyStateLabel}
              </dd>
            </div>
          </dl>

          {submitError ? <p className="mt-4 text-sm text-rose-600">{submitError}</p> : null}
        </aside>
      </div>
    </section>
  )
}

type ExtensionPresentationProjectPanelProps = {
  structuredDetail: React.ReactNode
}

const provenanceLabel = (value: string | null): string => {
  if (!value) return "Not provided"
  const normalized = value.replace(/_/g, " ")
  return `${normalized.charAt(0).toUpperCase()}${normalized.slice(1)}`
}

const presentationKindLabel = (record: SafePresentationMetadata): string => {
  if (record.contentKind === "standalone_html") {
    return "Standalone HTML + JavaScript"
  }
  if (record.contentKind === "structured_slides") return "Structured slides"
  return `Unknown kind: ${record.unsupportedKind}`
}

export const ExtensionPresentationProjectPanel: React.FC<
  ExtensionPresentationProjectPanelProps
> = ({ structuredDetail }) => {
  const { projectId: routeProjectId = "" } = useParams<{ projectId: string }>()
  const trustedProjectId = isTrustedProjectId(routeProjectId)
    ? routeProjectId
    : null
  const online = useServerOnline()
  const capabilityState = useServerCapabilities()
  const [retryEpoch, setRetryEpoch] = React.useState(0)
  const [view, setView] = React.useState<MetadataView | null>(null)
  const [opening, setOpening] = React.useState(false)
  const [handoffError, setHandoffError] = React.useState<string | null>(null)
  const trustedReadyRef = React.useRef<TrustedReadyMetadata | null>(null)
  const currentProjectIdRef = React.useRef<string | null>(trustedProjectId)
  if (currentProjectIdRef.current !== trustedProjectId) {
    currentProjectIdRef.current = trustedProjectId
    trustedReadyRef.current = null
  }
  const { boundaryEpoch, capture, isCurrent, retire } = useAuthorityFence(() => {
    trustedReadyRef.current = null
    setOpening(false)
    setHandoffError(null)
    if (trustedProjectId) {
      setView({ projectId: trustedProjectId, status: "loading" })
    }
  })

  React.useEffect(() => {
    if (
      !trustedProjectId ||
      !online ||
      capabilityState.loading ||
      capabilityState.capabilities?.hasPresentationStudio !== true
    ) {
      return
    }

    const operationEpoch = capture()
    let response: unknown = null
    trustedReadyRef.current = null
    setView({ projectId: trustedProjectId, status: "loading" })
    setHandoffError(null)
    setOpening(false)

    void (async () => {
      try {
        response = await tldwClient.getPresentationMetadata(trustedProjectId)
        if (!isCurrent(operationEpoch)) {
          response = null
          return
        }
        const record = projectPresentationMetadata(response, trustedProjectId)
        response = null
        if (!isCurrent(operationEpoch)) return
        if (record) {
          trustedReadyRef.current = {
            projectId: trustedProjectId,
            metadataEpoch: operationEpoch,
            record
          }
          setView({ projectId: trustedProjectId, status: "ready", record })
        } else {
          trustedReadyRef.current = null
          setView({ projectId: trustedProjectId, status: "invalid" })
        }
      } catch {
        response = null
        if (isCurrent(operationEpoch)) {
          trustedReadyRef.current = null
          setView({ projectId: trustedProjectId, status: "load_error" })
        }
      }
    })()

    return () => {
      response = null
      if (trustedReadyRef.current?.metadataEpoch === operationEpoch) {
        trustedReadyRef.current = null
      }
      retire()
    }
  }, [
    boundaryEpoch,
    capabilityState.capabilities?.hasPresentationStudio,
    capabilityState.loading,
    capture,
    isCurrent,
    online,
    retire,
    retryEpoch,
    trustedProjectId
  ])

  const handleOpenInWebUi = async (): Promise<void> => {
    const readyMetadata = trustedReadyRef.current
    if (
      !readyMetadata ||
      readyMetadata.projectId !== trustedProjectId ||
      currentProjectIdRef.current !== readyMetadata.projectId ||
      !isCurrent(readyMetadata.metadataEpoch) ||
      readyMetadata.record.contentKind === "structured_slides"
    ) {
      return
    }

    const operationEpoch = capture()
    let configResponse: unknown = null
    setOpening(true)
    setHandoffError(null)
    try {
      configResponse = await tldwClient.getConfig()
      if (!isCurrent(operationEpoch)) {
        configResponse = null
        return
      }
      const destination = buildPresentationWebUiTarget(
        configResponse,
        readyMetadata.record.id
      )
      configResponse = null
      if (!destination) {
        setHandoffError("A valid WebUI address is not configured")
        return
      }
      if (
        !isCurrent(operationEpoch) ||
        currentProjectIdRef.current !== readyMetadata.projectId ||
        trustedReadyRef.current?.metadataEpoch !== readyMetadata.metadataEpoch ||
        trustedReadyRef.current.record !== readyMetadata.record
      ) {
        return
      }
      window.open(destination, "_blank", "noopener,noreferrer")
    } catch {
      configResponse = null
      if (isCurrent(operationEpoch)) {
        setHandoffError("The WebUI handoff could not be prepared")
      }
    } finally {
      configResponse = null
      if (isCurrent(operationEpoch)) setOpening(false)
    }
  }

  const renderState = (
    title: string,
    message: string,
    state: "loading" | "unavailable" | "error" | "blocked",
    action?: { label: string; onClick: () => void }
  ) => (
    <PageShell className="py-6" maxWidthClassName="max-w-3xl">
      <StatePanel
        state={state}
        title={title}
        titleHeadingLevel={1}
        message={message}
        primaryAction={action}
        role={state === "loading" ? "status" : state === "error" ? "alert" : undefined}
        aria-live="polite"
      >
        {state === "loading" ? <LoadingState mode="skeleton" rows={3} /> : null}
      </StatePanel>
    </PageShell>
  )

  if (!trustedProjectId) {
    return renderState(
      "Presentation metadata could not be verified",
      "The requested presentation identifier is not safe to use.",
      "blocked"
    )
  }
  if (!online) {
    return renderState(
      "Presentation handoff is offline",
      "Reconnect to verify this presentation before continuing.",
      "unavailable"
    )
  }
  if (capabilityState.loading) {
    return renderState(
      "Checking Presentation Studio availability",
      "Waiting for a source-free capability check.",
      "loading"
    )
  }
  if (!capabilityState.capabilities) {
    return renderState(
      "Presentation Studio availability could not load",
      "Retry the capability check before opening this presentation.",
      "error",
      {
        label: "Retry",
        onClick: () => {
          setRetryEpoch((current) => current + 1)
          void capabilityState.refresh?.()
        }
      }
    )
  }
  if (capabilityState.capabilities.hasPresentationStudio !== true) {
    return renderState(
      "Presentation Studio is not available",
      "This server does not advertise Presentation Studio support.",
      "unavailable"
    )
  }

  const currentView = view?.projectId === trustedProjectId ? view : null
  if (!currentView || currentView.status === "loading") {
    return (
      <PageShell className="py-6" maxWidthClassName="max-w-3xl">
        <section
          role="status"
          aria-label="Loading presentation metadata"
          className="rounded-lg border border-border bg-surface p-4"
        >
          <h1 className="text-base font-semibold text-text">
            Loading presentation metadata
          </h1>
          <div className="mt-3">
            <LoadingState mode="skeleton" rows={3} />
          </div>
        </section>
      </PageShell>
    )
  }
  if (currentView.status === "load_error") {
    return renderState(
      "Presentation metadata could not load",
      "Check the server connection, then retry the source-free metadata request.",
      "error",
      {
        label: "Retry",
        onClick: () => setRetryEpoch((current) => current + 1)
      }
    )
  }
  if (currentView.status === "invalid") {
    return renderState(
      "Presentation metadata could not be verified",
      "The server returned metadata that is unsafe or does not match this route.",
      "blocked"
    )
  }
  if (currentView.record.contentKind === "structured_slides") {
    return <>{structuredDetail}</>
  }

  const record = currentView.record
  return (
    <PageShell className="space-y-6 py-6" maxWidthClassName="max-w-3xl">
      <section className="space-y-5 rounded-lg border border-border bg-surface p-6">
        <header className="space-y-3">
          <Badge
            variant={record.contentKind === "unsupported" ? "warning" : "secondary"}
          >
            {presentationKindLabel(record)}
          </Badge>
          <div className="space-y-2">
            <h1 className="text-2xl font-semibold text-text">{record.title}</h1>
            {record.description ? (
              <p className="text-sm text-text-muted">{record.description}</p>
            ) : null}
          </div>
        </header>

        <p className="text-sm text-text-muted">
          This presentation is read only in this extension. Open it in the WebUI to
          continue.
        </p>

        <dl className="grid gap-3 text-sm sm:grid-cols-2">
          {record.contentKind === "standalone_html" ? (
            <>
              <div>
                <dt className="font-medium text-text">Slides</dt>
                <dd className="text-text-muted">{record.slideCount}</dd>
              </div>
              <div>
                <dt className="font-medium text-text">Size</dt>
                <dd className="text-text-muted">{record.htmlBytes} bytes</dd>
              </div>
            </>
          ) : null}
          <div>
            <dt className="font-medium text-text">Source</dt>
            <dd className="text-text-muted">
              {provenanceLabel(record.provenance.sourceKind)}
            </dd>
          </div>
          <div>
            <dt className="font-medium text-text">Provider</dt>
            <dd className="text-text-muted">
              {record.provenance.provider || "Not provided"}
            </dd>
          </div>
          <div>
            <dt className="font-medium text-text">Model</dt>
            <dd className="text-text-muted">
              {record.provenance.model || "Not provided"}
            </dd>
          </div>
        </dl>

        <div className="flex flex-wrap items-center gap-3">
          <Button
            variant="primary"
            size="lg"
            loading={opening}
            onClick={() => {
              void handleOpenInWebUi()
            }}
          >
            Open in WebUI
          </Button>
          {handoffError ? (
            <p role="alert" className="text-sm text-state-error">
              {handoffError}
            </p>
          ) : null}
        </div>
      </section>
    </PageShell>
  )
}
