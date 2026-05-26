import React from "react"
import { Input, Radio, Upload, message, Button, Select, Spin } from "antd"
import { ImageIcon, Link, X, Upload as UploadIcon, Sparkles, RefreshCw } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives/Alert"
import {
  ALLOWED_IMAGE_MIME_TYPES,
  createImageDataUrl,
  decodeBase64Header,
  detectImageMime
} from "@/utils/image-utils"
import { tldwClient, type ImageBackend } from "@/services/tldw/TldwApiClient"

export type AvatarMode = "url" | "upload" | "generate"

export interface AvatarFieldValue {
  mode: AvatarMode
  url?: string
  base64?: string
}

interface AvatarFieldProps {
  value?: AvatarFieldValue
  onChange?: (value: AvatarFieldValue) => void
  characterName?: string
  characterDescription?: string
}

/**
 * Unified avatar field that allows choosing between URL input, file upload, or AI generation.
 * Stores both mode and the corresponding value.
 */
export function AvatarField({
  value,
  onChange,
  characterName,
  characterDescription
}: AvatarFieldProps) {
  const { t } = useTranslation(["settings", "common"])
  const [loading, setLoading] = React.useState(false)
  const [urlImgError, setUrlImgError] = React.useState(false)

  // Generate mode state
  const [prompt, setPrompt] = React.useState("")
  const [isGenerating, setIsGenerating] = React.useState(false)
  const [generationError, setGenerationError] = React.useState<string | null>(null)
  const [backends, setBackends] = React.useState<ImageBackend[]>([])
  const [selectedBackend, setSelectedBackend] = React.useState<string>("")
  const [backendsLoading, setBackendsLoading] = React.useState(false)
  const [backendsFetched, setBackendsFetched] = React.useState(false)

  const mode = value?.mode || "url"
  const urlValue = value?.url || ""
  const base64Value = value?.base64 || ""

  React.useEffect(() => {
    setUrlImgError(false)
  }, [urlValue])

  // Fetch available backends when switching to generate mode
  React.useEffect(() => {
    if (mode === "generate" && !backendsLoading && !backendsFetched) {
      setBackendsLoading(true)
      tldwClient
        .getImageBackends()
        .then((result) => {
          setBackends(result)
          // Auto-select first configured backend
          const configured = result.filter((b) => b.is_configured)
          if (configured.length > 0 && !selectedBackend) {
            setSelectedBackend(configured[0].id)
          }
        })
        .catch((err) => {
          console.warn("Failed to fetch image backends:", err)
          setBackends([])
        })
        .finally(() => {
          setBackendsLoading(false)
          setBackendsFetched(true)
        })
    }
  }, [mode, backendsLoading, backendsFetched, selectedBackend])

  // Auto-populate prompt from character context
  React.useEffect(() => {
    if (mode === "generate" && !prompt && (characterName || characterDescription)) {
      const parts: string[] = []
      if (characterName) {
        parts.push(`Portrait of ${characterName}`)
      }
      if (characterDescription) {
        parts.push(characterDescription)
      }
      if (parts.length > 0) {
        setPrompt(parts.join(", "))
      }
    }
  }, [mode, prompt, characterName, characterDescription])

  const handleModeChange = (newMode: AvatarMode) => {
    setGenerationError(null)
    onChange?.({
      mode: newMode,
      url: newMode === "url" ? urlValue : "",
      base64: newMode === "upload" || newMode === "generate" ? base64Value : ""
    })
  }

  const handleUrlChange = (newUrl: string) => {
    onChange?.({
      mode: "url",
      url: newUrl,
      base64: ""
    })
  }

  const handleUpload = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      message.error(
        t("settings:manageCharacters.avatar.selectImageError", {
          defaultValue: "Please select an image file"
        })
      )
      return false
    }

    setLoading(true)
    try {
      const result = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => {
          if (typeof reader.result === "string") {
            resolve(reader.result)
            return
          }
          reject(new Error("Invalid image data"))
        }
        reader.onerror = () => {
          reject(reader.error || new Error("Failed to process image"))
        }
        reader.readAsDataURL(file)
      })

      const base64Match = result.match(/^data:image\/[^;]+;base64,(.+)$/)
      if (!base64Match) {
        message.error(
          t("settings:manageCharacters.avatar.processError", {
            defaultValue: "Failed to process image"
          })
        )
        return false
      }

      const rawBase64 = base64Match[1]
      const headerBytes = decodeBase64Header(rawBase64)
      if (!headerBytes) {
        message.error(
          t("settings:manageCharacters.avatar.invalidError", {
            defaultValue: "Invalid image file"
          })
        )
        return false
      }

      const mime = detectImageMime(headerBytes)
      if (mime && ALLOWED_IMAGE_MIME_TYPES.has(mime)) {
        onChange?.({
          mode: "upload",
          url: "",
          base64: rawBase64
        })
      } else {
        message.error(
          t("settings:manageCharacters.avatar.formatError", {
            defaultValue: "Only PNG, JPEG, GIF, and WebP images are supported"
          })
        )
      }
    } catch {
      message.error(
        t("settings:manageCharacters.avatar.processError", {
          defaultValue: "Failed to process image"
        })
      )
    } finally {
      setLoading(false)
    }
    return false // Prevent default upload behavior
  }

  const handleClearUpload = () => {
    onChange?.({
      mode: mode as AvatarMode,
      url: "",
      base64: ""
    })
  }

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      message.warning(
        t("settings:manageCharacters.avatar.generate.promptRequired", {
          defaultValue: "Please enter a prompt to generate an avatar"
        })
      )
      return
    }

    const backend = selectedBackend || backends.find((b) => b.is_configured)?.id
    if (!backend) {
      setGenerationError(
        t("settings:manageCharacters.avatar.generate.noBackend", {
          defaultValue: "No image generation backend available"
        })
      )
      return
    }

    setIsGenerating(true)
    setGenerationError(null)

    try {
      const response = await tldwClient.createImageArtifact({
        backend,
        prompt: prompt.trim(),
        negativePrompt: "blurry, low quality, deformed, distorted",
        width: 512,
        height: 512,
        steps: 25,
        persist: false,
        timeoutMs: 60_000
      })

      const content_b64 = response.artifact?.export?.content_b64
      if (content_b64) {
        onChange?.({
          mode: "generate",
          url: "",
          base64: content_b64
        })
      } else {
        throw new Error("No image data received")
      }
    } catch (err: any) {
      console.error("Image generation failed:", err)
      const rawMessage =
        typeof err?.message === "string"
          ? err.message
          : typeof err === "string"
            ? err
            : ""
      const normalized = rawMessage.toLowerCase()
      const status = typeof err?.status === "number" ? err.status : undefined
      let mappedMessage = ""

      if (normalized.includes("image_backend_unavailable")) {
        mappedMessage = t("settings:manageCharacters.avatar.generate.backendUnavailable", {
          defaultValue: "Image backend not available"
        })
      } else if (normalized.includes("image_generation_failed")) {
        mappedMessage = t("settings:manageCharacters.avatar.generate.failed", {
          defaultValue: "Generation failed. Try again."
        })
      } else if (normalized.includes("abort") || normalized.includes("timeout")) {
        mappedMessage = t("settings:manageCharacters.avatar.generate.timeout", {
          defaultValue: "Generation timed out"
        })
      } else if (status === 0 || normalized.includes("network")) {
        mappedMessage = t("settings:manageCharacters.avatar.generate.networkError", {
          defaultValue: "Unable to connect to server"
        })
      }

      setGenerationError(
        mappedMessage ||
          rawMessage ||
          t("settings:manageCharacters.avatar.generate.failed", {
            defaultValue: "Generation failed. Try again."
          })
      )
    } finally {
      setIsGenerating(false)
    }
  }

  const previewUrl = React.useMemo(() => {
    if (mode === "url" && urlValue) {
      return urlValue
    }
    if ((mode === "upload" || mode === "generate") && base64Value) {
      return createImageDataUrl(base64Value)
    }
    return null
  }, [mode, urlValue, base64Value])

  const configuredBackends = backends.filter((b) => b.is_configured)
  const hasConfiguredBackends = configuredBackends.length > 0

  return (
    <div className="space-y-3">
      {/* Mode selector */}
      <Radio.Group
        value={mode}
        onChange={(e) => handleModeChange(e.target.value)}
        className="flex gap-4">
        <Radio value="url" className="flex items-center gap-1">
          <Link className="w-4 h-4 inline-block mr-1" />
          {t("settings:manageCharacters.avatar.tabUrl", {
            defaultValue: "URL"
          })}
        </Radio>
        <Radio value="upload" className="flex items-center gap-1">
          <UploadIcon className="w-4 h-4 inline-block mr-1" />
          {t("settings:manageCharacters.avatar.tabUpload", {
            defaultValue: "Upload"
          })}
        </Radio>
        <Radio value="generate" className="flex items-center gap-1">
          <Sparkles className="w-4 h-4 inline-block mr-1" />
          {t("settings:manageCharacters.avatar.tabGenerate", {
            defaultValue: "Generate"
          })}
        </Radio>
      </Radio.Group>

      {/* URL input */}
      {mode === "url" && (
        <Input
          value={urlValue}
          onChange={(e) => handleUrlChange(e.target.value)}
          placeholder={t("settings:manageCharacters.form.avatarUrl.placeholder", {
            defaultValue: "https://example.com/avatar.png"
          })}
          prefix={<Link className="w-4 h-4 text-text-subtle" />}
        />
      )}

      {/* Upload area */}
      {mode === "upload" && (
        <div className="space-y-2">
          {base64Value ? (
            <div className="relative inline-block">
              <img
                src={previewUrl || ""}
                alt="Avatar preview"
                className="w-16 h-16 rounded-lg object-cover border border-border"
              />
              <button
                type="button"
                onClick={handleClearUpload}
                className="absolute -top-2 -right-2 rounded-full bg-danger p-1 text-white shadow-sm hover:bg-danger focus:outline-none focus:ring-2 focus:ring-danger focus:ring-offset-1"
                aria-label={t("common:clear", { defaultValue: "Clear" })}>
                <X className="w-3 h-3" />
              </button>
            </div>
          ) : (
            <Upload.Dragger
              accept="image/png,image/jpeg,image/gif"
              showUploadList={false}
              beforeUpload={handleUpload}
              disabled={loading}
              className="!border-dashed !border-border hover:!border-primary">
              <div className="flex flex-col items-center gap-2 py-4">
                <ImageIcon className="w-8 h-8 text-text-subtle" />
                <p className="text-sm text-text-muted">
                  {loading
                    ? t("common:loading.title", {
                        defaultValue: "Loading..."
                      })
                    : t("settings:manageCharacters.avatar.dropzone", {
                        defaultValue: "Click or drag image to upload"
                      })}
                </p>
                <p className="text-xs text-text-subtle">
                  {t("settings:manageCharacters.avatar.formats", {
                    defaultValue: "PNG, JPEG, or GIF"
                  })}
                </p>
              </div>
            </Upload.Dragger>
          )}
        </div>
      )}

      {/* Generate area */}
      {mode === "generate" && (
        <div className="space-y-3">
          {backendsLoading ? (
            <div className="flex items-center gap-2 text-text-subtle">
              <Spin size="small" />
              <span className="text-sm">
                {t("settings:manageCharacters.avatar.generate.loadingBackends", {
                  defaultValue: "Loading image backends..."
                })}
              </span>
            </div>
          ) : !hasConfiguredBackends ? (
            <Alert
              variant="info"
              title={t("settings:manageCharacters.avatar.generate.noBackendsTitle", {
                defaultValue: "No image backends configured"
              })}
            >
              {t("settings:manageCharacters.avatar.generate.noBackendsDesc", {
                defaultValue:
                  "Configure stable-diffusion or SwarmUI in server settings to enable avatar generation."
              })}
            </Alert>
          ) : (
            <>
              {/* Prompt input */}
              <div>
                <label className="block text-sm font-medium text-text-default mb-1">
                  {t("settings:manageCharacters.avatar.generate.promptLabel", {
                    defaultValue: "Describe the avatar:"
                  })}
                </label>
                <Input.TextArea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={t("settings:manageCharacters.avatar.generate.promptPlaceholder", {
                    defaultValue: "Portrait of a wise mentor with kind eyes..."
                  })}
                  autoSize={{ minRows: 2, maxRows: 4 }}
                  disabled={isGenerating}
                />
              </div>

              {/* Backend selector (only if multiple configured) */}
              {configuredBackends.length > 1 && (
                <div>
                  <label className="block text-sm font-medium text-text-default mb-1">
                    {t("settings:manageCharacters.avatar.generate.backendLabel", {
                      defaultValue: "Backend:"
                    })}
                  </label>
                  <Select
                    value={selectedBackend}
                    onChange={setSelectedBackend}
                    disabled={isGenerating}
                    className="w-full"
                    options={configuredBackends.map((b) => ({
                      value: b.id,
                      label: b.name
                    }))}
                  />
                </div>
              )}

              {/* Error display */}
              {generationError && (
                <Alert
                  variant="error"
                  title={generationError}
                  dismissible
                  dismissLabel={t("common:close", { defaultValue: "Close" })}
                  onDismiss={() => setGenerationError(null)}
                />
              )}

              {/* Generated image preview or generate button */}
              {base64Value ? (
                <div className="flex items-center gap-3">
                  <div className="relative inline-block">
                    <img
                      src={previewUrl || ""}
                      alt="Generated avatar"
                      className="w-16 h-16 rounded-lg object-cover border border-border"
                    />
                    <button
                      type="button"
                      onClick={handleClearUpload}
                      className="absolute -top-2 -right-2 rounded-full bg-danger p-1 text-white shadow-sm hover:bg-danger focus:outline-none focus:ring-2 focus:ring-danger focus:ring-offset-1"
                      aria-label={t("common:clear", { defaultValue: "Clear" })}>
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                  <Button
                    type="default"
                    icon={<RefreshCw className="w-4 h-4" />}
                    onClick={handleGenerate}
                    loading={isGenerating}
                    disabled={!prompt.trim()}>
                    {t("settings:manageCharacters.avatar.generate.regenerate", {
                      defaultValue: "Regenerate"
                    })}
                  </Button>
                </div>
              ) : (
                <Button
                  type="primary"
                  icon={<Sparkles className="w-4 h-4" />}
                  onClick={handleGenerate}
                  loading={isGenerating}
                  disabled={!prompt.trim()}>
                  {isGenerating
                    ? t("settings:manageCharacters.avatar.generate.generating", {
                        defaultValue: "Generating..."
                      })
                    : t("settings:manageCharacters.avatar.generate.button", {
                        defaultValue: "Generate Avatar"
                      })}
                </Button>
              )}
            </>
          )}
        </div>
      )}

      {/* Preview for URL mode */}
      {mode === "url" && urlValue && !urlImgError && (
        <div className="flex items-center gap-2">
          <img
            src={urlValue}
            alt="Avatar preview"
            className="w-10 h-10 rounded-lg object-cover border border-border"
            onError={() => {
              setUrlImgError(true)
            }}
          />
          <span className="text-xs text-text-subtle">
            {t("settings:manageCharacters.avatar.preview", {
              defaultValue: "Preview"
            })}
          </span>
        </div>
      )}
    </div>
  )
}

/**
 * Helper to extract avatar_url and image_base64 from AvatarFieldValue for form submission.
 */
export function extractAvatarValues(avatar?: AvatarFieldValue): {
  avatar_url?: string
  image_base64?: string
} {
  if (!avatar) return {}
  return {
    avatar_url: avatar.mode === "url" ? avatar.url || undefined : undefined,
    image_base64:
      avatar.mode === "upload" || avatar.mode === "generate"
        ? avatar.base64 || undefined
        : undefined
  }
}

/**
 * Helper to create AvatarFieldValue from existing avatar_url and image_base64.
 */
export function createAvatarValue(
  avatar_url?: string | null,
  image_base64?: string | null
): AvatarFieldValue {
  if (image_base64) {
    return { mode: "generate", url: "", base64: image_base64 }
  }
  return { mode: "url", url: avatar_url || "", base64: "" }
}
