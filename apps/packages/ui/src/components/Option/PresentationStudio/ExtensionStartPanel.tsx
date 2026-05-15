import React from "react"
import { useTranslation } from "react-i18next"

import { useConnectionState } from "@/hooks/useConnectionState"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useServerOnline } from "@/hooks/useServerOnline"
import { getScreenshotFromCurrentTab } from "@/libs/get-screenshot"
import { tldwClient } from "@/services/tldw/TldwApiClient"
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
    return new URL(serverUrl).origin
  } catch {
    return null
  }
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

  const openWebUiProject = (projectId: string): void => {
    if (!serverOrigin) {
      throw new Error("Configure a valid server URL before starting Presentation Studio.")
    }
    const destination = new URL(
      `/presentation-studio/${encodeURIComponent(projectId)}`,
      serverOrigin
    ).toString()
    window.open(destination, "_blank", "noopener,noreferrer")
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
    try {
      const project = await tldwClient.createPresentation({
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
      openWebUiProject(project.id)
    } catch (error) {
      setSubmitError(
        error instanceof Error ? error.message || "Failed to create project." : "Failed to create project."
      )
    } finally {
      setIsSubmitting(false)
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
