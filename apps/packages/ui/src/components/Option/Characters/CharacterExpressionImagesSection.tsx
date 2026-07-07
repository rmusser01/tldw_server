import React from "react"
import { Button, Form, Input, Radio, Upload, message } from "antd"
import type { FormListFieldData, UploadProps } from "antd"
import {
  ImageIcon,
  Link,
  Plus,
  RefreshCw,
  Sparkles,
  Trash2
} from "lucide-react"
import {
  ALLOWED_IMAGE_MIME_TYPES,
  createImageDataUrl,
  decodeBase64Header,
  detectImageMime
} from "@/utils/image-utils"
import { tldwClient, type ImageBackend } from "@/services/tldw/TldwApiClient"
import {
  EXPRESSION_IMAGE_STARTER_STATES,
  createEmptyCustomExpressionRow,
  normalizeExpressionImageRows,
  type ExpressionImageRow,
  type ExpressionImageRowErrorReason
} from "./character-expression-images"
import { MAX_AVATAR_IMAGE_BYTES, type AvatarFieldValue, type AvatarMode } from "./AvatarField"

type CharacterExpressionImagesSectionProps = {
  characterName?: string
  characterDescription?: string
  baseAvatar?: AvatarFieldValue
}

type ExpressionRowEditorProps = {
  field: FormListFieldData
  row?: ExpressionImageRow
  errors: ExpressionImageRowErrorReason[]
  characterName?: string
  characterDescription?: string
  onRemove: () => void
}

const ERROR_MESSAGES: Record<ExpressionImageRowErrorReason, string> = {
  "duplicate": "Expression state is duplicated.",
  "invalid-state": "Use letters, numbers, spaces, hyphens, or underscores.",
  "missing-image": "Custom expressions need an image.",
  "missing-state": "Expression state is required."
}

const modeOptions = [
  { label: "URL", value: "url" },
  { label: "Upload", value: "upload" },
  { label: "Generate", value: "generate" }
]

const readFileAsDataUrl = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
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

const getRowImageUrl = (value?: AvatarFieldValue): string => {
  if (!value) return ""
  if (value.mode === "url") return value.url?.trim() || ""
  return value.base64 ? createImageDataUrl(value.base64) || "" : ""
}

const getRowLabel = (row?: ExpressionImageRow): string =>
  row?.state?.trim() || "custom expression"

const toEffectiveRow = (
  row: Partial<ExpressionImageRow> | undefined,
  index: number
): ExpressionImageRow => {
  const state = row?.state || ""
  const image: Partial<AvatarFieldValue> = row?.image || {}
  return {
    id: row?.id || `expression-row-${index}`,
    state,
    starter:
      row?.starter ??
      (EXPRESSION_IMAGE_STARTER_STATES[index] === state),
    image: {
      mode: image.mode || "url",
      url: image.url || "",
      base64: image.base64 || ""
    }
  }
}

function ExpressionRowEditor({
  field,
  row,
  errors,
  characterName,
  characterDescription,
  onRemove
}: ExpressionRowEditorProps) {
  const form = Form.useFormInstance()
  const [previewError, setPreviewError] = React.useState(false)
  const [uploading, setUploading] = React.useState(false)
  const [prompt, setPrompt] = React.useState("")
  const [backends, setBackends] = React.useState<ImageBackend[]>([])
  const [selectedBackend, setSelectedBackend] = React.useState("")
  const [backendsLoading, setBackendsLoading] = React.useState(false)
  const [backendsFetched, setBackendsFetched] = React.useState(false)
  const [generating, setGenerating] = React.useState(false)
  const [generationError, setGenerationError] = React.useState<string | null>(null)

  const image = row?.image || { mode: "url", url: "", base64: "" }
  const mode = image.mode || "url"
  const imageUrl = getRowImageUrl(image)
  const rowLabel = getRowLabel(row)
  const configuredBackends = backends.filter((backend) => backend.is_configured)

  React.useEffect(() => {
    setPreviewError(false)
  }, [imageUrl])

  React.useEffect(() => {
    if (mode !== "generate" || backendsLoading || backendsFetched) return

    setBackendsLoading(true)
    tldwClient
      .getImageBackends()
      .then((result) => {
        setBackends(result)
        const firstConfigured = result.find((backend) => backend.is_configured)
        if (firstConfigured && !selectedBackend) {
          setSelectedBackend(firstConfigured.id)
        }
      })
      .catch(() => {
        setBackends([])
      })
      .finally(() => {
        setBackendsLoading(false)
        setBackendsFetched(true)
      })
  }, [mode, backendsFetched, backendsLoading, selectedBackend])

  React.useEffect(() => {
    if (mode !== "generate" || prompt) return

    const parts = [`Portrait of ${characterName || "the character"}`]
    if (row?.state?.trim()) parts.push(`showing ${row.state.trim()}`)
    if (characterDescription) parts.push(characterDescription)
    setPrompt(parts.join(", "))
  }, [mode, prompt, row?.state, characterName, characterDescription])

  const setRowImage = (nextImage: AvatarFieldValue) => {
    const rows = [...(form.getFieldValue("expression_images") || [])]
    rows[field.name] = {
      ...rows[field.name],
      image: nextImage
    }
    form.setFieldValue("expression_images", rows)
  }

  const handleModeChange = (nextMode: AvatarMode) => {
    setGenerationError(null)
    setRowImage({
      mode: nextMode,
      url: nextMode === "url" ? image.url || "" : "",
      base64: nextMode === "upload" || nextMode === "generate" ? image.base64 || "" : ""
    })
  }

  const handleUpload: UploadProps["beforeUpload"] = async (file) => {
    if (!file.type.startsWith("image/")) {
      message.error("Please select an image file")
      return false
    }

    if (file.size > MAX_AVATAR_IMAGE_BYTES) {
      message.error("Image is too large. Please choose an image around 5 MB or less.")
      return false
    }

    setUploading(true)
    try {
      const dataUrl = await readFileAsDataUrl(file)
      const base64Match = dataUrl.match(/^data:image\/[^;]+;base64,(.+)$/)
      if (!base64Match) {
        message.error("Failed to process image")
        return false
      }

      const rawBase64 = base64Match[1]
      const headerBytes = decodeBase64Header(rawBase64)
      if (!headerBytes) {
        message.error("Invalid image file")
        return false
      }

      const mime = detectImageMime(headerBytes)
      if (mime && ALLOWED_IMAGE_MIME_TYPES.has(mime)) {
        setRowImage({ mode: "upload", url: "", base64: rawBase64 })
      } else {
        message.error("Only PNG, JPEG, GIF, and WebP images are supported")
      }
    } catch {
      message.error("Failed to process image")
    } finally {
      setUploading(false)
    }

    return false
  }

  const handleGenerate = async () => {
    const backend = selectedBackend || configuredBackends[0]?.id
    if (!backend) {
      setGenerationError("No image generation backend available.")
      return
    }
    if (!prompt.trim()) {
      setGenerationError("Describe the expression image first.")
      return
    }

    setGenerating(true)
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
      const contentBase64 = response.artifact?.export?.content_b64
      if (!contentBase64) throw new Error("No image data received")
      setRowImage({ mode: "generate", url: "", base64: contentBase64 })
    } catch (error) {
      const messageText =
        error instanceof Error && error.message ? error.message : "Generation failed."
      setGenerationError(messageText)
    } finally {
      setGenerating(false)
    }
  }

  return (
    <div className="rounded-md border border-border p-3 space-y-3">
      <div className="grid gap-2 md:grid-cols-[minmax(9rem,1fr)_auto_auto] md:items-start">
        <Form.Item name={[field.name, "state"]} className="!mb-0">
          <Input
            aria-label={
              row?.starter
                ? `Expression state ${rowLabel}`
                : row?.state
                  ? `Custom expression state ${rowLabel}`
                  : "Custom expression state"
            }
            placeholder="state"
            size="small"
          />
        </Form.Item>
        <Radio.Group
          size="small"
          optionType="button"
          buttonStyle="solid"
          options={modeOptions}
          value={mode}
          onChange={(event) => handleModeChange(event.target.value)}
          aria-label={`Image source for ${rowLabel}`}
        />
        <Button
          size="small"
          danger
          type="text"
          icon={<Trash2 className="h-4 w-4" />}
          aria-label={`Remove expression ${rowLabel}`}
          onClick={onRemove}
        />
      </div>

      {mode === "url" ? (
        <Form.Item name={[field.name, "image", "url"]} className="!mb-0">
          <Input
            size="small"
            aria-label={`Expression image URL for ${rowLabel}`}
            placeholder="https://example.com/expression.png"
            prefix={<Link className="h-4 w-4 text-text-subtle" />}
            onChange={(event) =>
              setRowImage({ mode: "url", url: event.target.value, base64: "" })
            }
          />
        </Form.Item>
      ) : null}

      {mode === "upload" ? (
        <Upload
          accept="image/png,image/jpeg,image/gif,image/webp"
          showUploadList={false}
          beforeUpload={handleUpload}
          disabled={uploading}
        >
          <Button
            size="small"
            icon={<ImageIcon className="h-4 w-4" />}
            loading={uploading}
          >
            Upload image
          </Button>
        </Upload>
      ) : null}

      {mode === "generate" ? (
        <div className="space-y-2">
          {backendsLoading ? (
            <p className="text-xs text-text-subtle">Loading image backends...</p>
          ) : configuredBackends.length === 0 ? (
            <p className="text-xs text-text-subtle">No image backends configured.</p>
          ) : (
            <>
              <Input.TextArea
                aria-label={`Generation prompt for ${rowLabel}`}
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                autoSize={{ minRows: 2, maxRows: 3 }}
                disabled={generating}
              />
              {configuredBackends.length > 1 ? (
                <select
                  className="w-full rounded-md border border-border bg-surface px-2 py-1 text-sm"
                  aria-label={`Image backend for ${rowLabel}`}
                  value={selectedBackend}
                  onChange={(event) => setSelectedBackend(event.target.value)}
                  disabled={generating}
                >
                  {configuredBackends.map((backend) => (
                    <option key={backend.id} value={backend.id}>
                      {backend.name}
                    </option>
                  ))}
                </select>
              ) : null}
              {generationError ? (
                <p role="alert" className="text-xs text-danger">
                  {generationError}
                </p>
              ) : null}
              <Button
                size="small"
                type="primary"
                icon={
                  image.base64 ? (
                    <RefreshCw className="h-4 w-4" />
                  ) : (
                    <Sparkles className="h-4 w-4" />
                  )
                }
                loading={generating}
                onClick={handleGenerate}
              >
                {image.base64 ? "Regenerate image" : "Generate image"}
              </Button>
            </>
          )}
        </div>
      ) : null}

      <div className="flex items-center gap-3">
        {imageUrl && !previewError ? (
          <img
            src={imageUrl}
            alt={`${rowLabel} thumbnail`}
            className="h-12 w-12 rounded-md border border-border object-cover"
            onError={() => setPreviewError(true)}
          />
        ) : (
          <div className="flex h-12 w-12 items-center justify-center rounded-md border border-dashed border-border text-xs text-text-subtle">
            No image
          </div>
        )}
        {errors.length > 0 ? (
          <div className="space-y-1 text-xs text-danger">
            {errors.map((error) => (
              <p key={error} role="alert">
                {ERROR_MESSAGES[error]}
              </p>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}

export function CharacterExpressionImagesSection({
  characterName,
  characterDescription,
  baseAvatar
}: CharacterExpressionImagesSectionProps) {
  const form = Form.useFormInstance()
  const watchedRows = Form.useWatch("expression_images", form) as
    | Partial<ExpressionImageRow>[]
    | undefined
  const rows = (watchedRows || []).map(toEffectiveRow)
  const [previewState, setPreviewState] = React.useState("")
  const [failedPreviewImages, setFailedPreviewImages] = React.useState<Set<string>>(
    () => new Set()
  )

  const errorsById = React.useMemo(() => {
    const map = new Map<string, ExpressionImageRowErrorReason[]>()
    normalizeExpressionImageRows(rows).errors.forEach((error) => {
      map.set(error.id, [...(map.get(error.id) || []), error.reason])
    })
    return map
  }, [rows])

  const previewRows = rows.filter((row) => row?.id && row?.state?.trim())
  const selectedPreviewRow =
    previewRows.find((row) => row.state === previewState) || previewRows[0]
  const selectedImageUrl = getRowImageUrl(selectedPreviewRow?.image)
  const selectedImageKey =
    selectedPreviewRow && selectedImageUrl
      ? `${selectedPreviewRow.id}:${selectedImageUrl}`
      : ""
  const baseAvatarUrl = getRowImageUrl(baseAvatar)
  const previewSrc =
    selectedImageUrl && !failedPreviewImages.has(selectedImageKey)
      ? selectedImageUrl
      : baseAvatarUrl

  React.useEffect(() => {
    if (!previewRows.length) {
      setPreviewState("")
      return
    }
    if (previewState && !previewRows.some((row) => row.state === previewState)) {
      setPreviewState("")
    }
  }, [previewRows, previewState])

  const copyDirective = async () => {
    const state = selectedPreviewRow?.state?.trim()
    if (!state) return
    await navigator.clipboard?.writeText(`Emote: ${state}`)
  }

  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-base font-semibold text-text-default">Expression images</h3>
        <p className="text-sm text-text-subtle">
          Map Emote: &lt;state&gt; directives to character images.
        </p>
      </div>

      <Form.List name="expression_images">
        {(fields, { add, remove }) => (
          <div className="space-y-3">
            {fields.map((field) => {
              const row = rows[field.name]
              return (
                <ExpressionRowEditor
                  key={field.key}
                  field={field}
                  row={row}
                  errors={row?.id ? errorsById.get(row.id) || [] : []}
                  characterName={characterName}
                  characterDescription={characterDescription}
                  onRemove={() => remove(field.name)}
                />
              )
            })}
            <Button
              type="dashed"
              icon={<Plus className="h-4 w-4" />}
              onClick={() => add(createEmptyCustomExpressionRow())}
            >
              Add expression
            </Button>
          </div>
        )}
      </Form.List>

      <div className="flex flex-wrap items-end gap-3 rounded-md border border-border p-3">
        <label className="flex min-w-48 flex-col gap-1 text-sm">
          <span className="font-medium text-text-default">Preview</span>
          <select
            className="rounded-md border border-border bg-surface px-2 py-1"
            aria-label="Preview expression"
            value={previewState}
            onChange={(event) => setPreviewState(event.target.value)}
          >
            <option value="">Auto preview</option>
            {previewRows.map((row) => (
              <option key={row.id} value={row.state}>
                {row.state}
              </option>
            ))}
          </select>
        </label>

        {previewSrc ? (
          <img
            src={previewSrc}
            alt={`${characterName || "Character"} expression preview`}
            className="h-24 w-24 rounded-md border border-border object-cover"
            onError={() => {
              if (selectedImageKey) {
                setFailedPreviewImages((current) => {
                  const next = new Set(current)
                  next.add(selectedImageKey)
                  return next
                })
              }
            }}
          />
        ) : (
          <div className="flex h-24 w-24 items-center justify-center rounded-md border border-dashed border-border text-sm text-text-subtle">
            No preview
          </div>
        )}

        <Button
          icon={<Link className="h-4 w-4" />}
          onClick={copyDirective}
          disabled={!selectedPreviewRow?.state?.trim()}
          aria-label="Copy emote directive"
        >
          Copy emote directive
        </Button>
      </div>
    </section>
  )
}
