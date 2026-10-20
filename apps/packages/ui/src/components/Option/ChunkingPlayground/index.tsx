import React, { useState, useCallback, useEffect } from "react"
import { useTranslation } from "react-i18next"
import {
  Card,
  Form,
  Input,
  AutoComplete,
  Select,
  InputNumber,
  Button,
  Tabs,
  Segmented,
  Space,
  Collapse,
  Switch,
  Divider,
  Alert,
  Spin,
  Typography,
  Upload,
  message
} from "antd"
import type { UploadProps } from "antd"
import { UploadOutlined, ScissorOutlined, SaveOutlined, DownloadOutlined } from "@ant-design/icons"
import { useQuery } from "@tanstack/react-query"
import { useDesktop } from "@/hooks/useMediaQuery"

import {
  chunkText,
  chunkFile,
  processPdfForChunking,
  getChunkingCapabilities,
  listChunkingTemplates,
  getChunkingTemplate,
  calculateChunkStats,
  DEFAULT_CHUNKING_OPTIONS,
  type Chunk,
  type ChunkingOptions,
  type ChunkingCapabilities,
  type ChunkingResponse,
  type ChunkingTemplateListResponse
} from "@/services/chunking"
import { bgRequest } from "@/services/background-proxy"
import { downloadBlob } from "@/utils/download-blob"

import { ChunkCardView } from "./ChunkCardView"
import { ChunkInlineView } from "./ChunkInlineView"
import { SampleTexts } from "./SampleTexts"
import { MediaSelector } from "./MediaSelector"
import { CompareView } from "./CompareView"
import { ChunkingTemplatesPanel } from "./ChunkingTemplatesPanel"
import { ChunkingCapabilitiesPanel } from "./ChunkingCapabilitiesPanel"
import { SaveAsTemplateModal } from "./SaveAsTemplateModal"
import { getLanguageOptions } from "./constants"
import { SplitView } from "./SplitView"

const { TextArea } = Input
const { Text, Title } = Typography

type InputSource = "paste" | "upload" | "sample" | "media" | "pdf"
type ViewMode = "cards" | "inline" | "split"
type PlaygroundMode = "single" | "compare" | "templates" | "capabilities"
type RequestMode = "json" | "file"

interface PdfOptions {
  parsingEngine: string
  enableOcr: boolean
  ocrBackend?: string
  ocrMode: "always" | "fallback"
  ocrLang: string
  ocrDpi: number
  ocrMinPageTextChars: number
  ocrOutputFormat?: string
  ocrPromptPreset?: string
}

interface ChunkingPlaygroundProps {
  className?: string
}

export const ChunkingPlayground: React.FC<ChunkingPlaygroundProps> = ({
  className
}) => {
  const { t } = useTranslation(["settings", "common"])
  const isDesktop = useDesktop()

  // Input state
  const [inputSource, setInputSource] = useState<InputSource>("paste")
  const [inputText, setInputText] = useState("")
  const [inputFile, setInputFile] = useState<File | null>(null)

  // Settings state (schema-driven)
  const [options, setOptions] = useState<ChunkingOptions>({
    ...DEFAULT_CHUNKING_OPTIONS
  })
  const [templateName, setTemplateName] = useState("")
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [pdfMetadata, setPdfMetadata] = useState<Record<string, any> | null>(
    null
  )
  const [pdfOptions, setPdfOptions] = useState<PdfOptions>({
    parsingEngine: "pymupdf4llm",
    enableOcr: false,
    ocrBackend: undefined,
    ocrMode: "fallback",
    ocrLang: "eng",
    ocrDpi: 300,
    ocrMinPageTextChars: 40,
    ocrOutputFormat: undefined,
    ocrPromptPreset: undefined
  })

  const [requestMode, setRequestMode] = useState<RequestMode>("json")

  // Results state
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastResponse, setLastResponse] = useState<ChunkingResponse | null>(null)

  // View state
  const [viewMode, setViewMode] = useState<ViewMode>("cards")
  const [playgroundMode, setPlaygroundMode] = useState<PlaygroundMode>("single")
  const [highlightedChunkIndex, setHighlightedChunkIndex] = useState<
    number | null
  >(null)

  // Save as Template modal state
  const [saveAsTemplateOpen, setSaveAsTemplateOpen] = useState(false)
  const [loadingTemplate, setLoadingTemplate] = useState(false)

  // Fetch capabilities from server
  const {
    data: capabilities,
    isLoading: capabilitiesLoading,
    refetch: refetchCapabilities
  } = useQuery<ChunkingCapabilities>({
    queryKey: ["chunking-capabilities"],
    queryFn: getChunkingCapabilities,
    staleTime: 5 * 60 * 1000 // Cache for 5 minutes
  })

  const {
    data: templateList,
    isLoading: templatesLoading,
    error: templatesError,
    refetch: refetchTemplates
  } = useQuery<ChunkingTemplateListResponse>({
    queryKey: ["chunking-templates", "playground"],
    queryFn: () =>
      listChunkingTemplates({ includeBuiltin: true, includeCustom: true }),
    staleTime: 60 * 1000
  })

  const { data: ocrBackends, error: ocrBackendsError } = useQuery<
    Record<string, { available?: boolean }>
  >({
    queryKey: ["ocr-backends"],
    queryFn: async () => {
      return await bgRequest<Record<string, { available?: boolean }>>({
        path: "/api/v1/ocr/backends",
        method: "GET"
      })
    },
    staleTime: 5 * 60 * 1000,
    enabled: inputSource === "pdf" && pdfOptions.enableOcr
  })

  const methodOptions = React.useMemo(() => {
    if (!capabilities?.methods) {
      return [
        { value: "words", label: "Words" },
        { value: "sentences", label: "Sentences" },
        { value: "paragraphs", label: "Paragraphs" }
      ]
    }
    return capabilities.methods.map((m) => ({
      value: m,
      label: m.charAt(0).toUpperCase() + m.slice(1).replace(/_/g, " ")
    }))
  }, [capabilities])

  const pdfMethodAllowlist = React.useMemo(
    () =>
      new Set([
        "semantic",
        "tokens",
        "paragraphs",
        "sentences",
        "words",
        "ebook_chapters",
        "json",
        "propositions"
      ]),
    []
  )

  const pdfMethodOptions = React.useMemo(
    () =>
      methodOptions.filter((option) =>
        pdfMethodAllowlist.has(String(option.value))
      ),
    [methodOptions, pdfMethodAllowlist]
  )

  const languageOptions = React.useMemo(() => getLanguageOptions(t), [t])
  const templateOptions = React.useMemo(() => {
    return (
      templateList?.templates?.map((template) => ({
        value: template.name,
        label: template.name
      })) ?? []
    )
  }, [templateList])

  const ocrBackendOptions = React.useMemo(() => {
    if (!ocrBackends) return []
    return Object.entries(ocrBackends).map(([name, info]) => ({
      value: name,
      label: info?.available === false ? `${name} (unavailable)` : name,
      disabled: info?.available === false
    }))
  }, [ocrBackends])
  const ocrBackendsErrorMessage = React.useMemo(() => {
    if (!ocrBackendsError) return null
    if (ocrBackendsError instanceof Error && ocrBackendsError.message) {
      return ocrBackendsError.message
    }
    return String(ocrBackendsError)
  }, [ocrBackendsError])

  const updatePdfOptions = useCallback(
    (updates: Partial<PdfOptions>) => {
      setPdfOptions((prev) => ({
        ...prev,
        ...updates
      }))
    },
    [setPdfOptions]
  )

  // Prefer server-provided parsing engines when available; fall back to known defaults.
  const pdfParsingEngineOptions = React.useMemo(() => {
    const fallbackEngines = ["pymupdf4llm", "pymupdf", "docling"]
    const engines = capabilities?.pdf_parsing_engines?.length
      ? capabilities.pdf_parsing_engines
      : fallbackEngines
    const normalized =
      pdfOptions.parsingEngine && !engines.includes(pdfOptions.parsingEngine)
        ? [pdfOptions.parsingEngine, ...engines]
        : engines
    return normalized.map((engine) => ({
      value: engine,
      label: engine
    }))
  }, [capabilities, pdfOptions.parsingEngine])

  const fallbackSchema = React.useMemo(() => {
    const properties: Record<string, any> = {}
    Object.entries(DEFAULT_CHUNKING_OPTIONS).forEach(([key, value]) => {
      let type = "string"
      if (typeof value === "number") {
        type = Number.isInteger(value) ? "integer" : "number"
      } else if (typeof value === "boolean") {
        type = "boolean"
      } else if (typeof value === "object" && value !== null) {
        type = "object"
      }
      properties[key] = {
        type,
        default: value
      }
    })
    return { type: "object", properties }
  }, [])

  const optionsSchema = React.useMemo(
    () => capabilities?.options_schema ?? fallbackSchema,
    [capabilities, fallbackSchema]
  )
  const schemaProps: Record<string, any> =
    (optionsSchema as Record<string, any>)?.properties ?? {}
  const schemaDefs: Record<string, any> =
    (optionsSchema as Record<string, any>)?.$defs ??
    (optionsSchema as Record<string, any>)?.definitions ??
    {}

  const setOptionValue = useCallback(
    (name: string, value: any) => {
      setOptions((prev) => ({
        ...prev,
        [name]: value
      }))
    },
    [setOptions]
  )

  const setNestedOptionValue = useCallback(
    (parent: string, child: string, value: any) => {
      setOptions((prev) => ({
        ...prev,
        [parent]: {
          ...((typeof (prev as Record<string, any>)[parent] === "object" &&
            (prev as Record<string, any>)[parent] !== null)
            ? (prev as Record<string, any>)[parent]
            : {}),
          [child]: value
        }
      }))
    },
    [setOptions]
  )

  const buildChunkingOptions = useCallback((): ChunkingOptions => {
    const base: ChunkingOptions = { ...options }
    if (templateName.trim()) {
      base.template_name = templateName.trim()
    }
    if (base.language === "auto") {
      delete base.language
    }

    const cleanedEntries = Object.entries(base).filter(([, value]) => {
      if (value === undefined || value === null) return false
      if (typeof value === "string" && value.trim() === "") return false
      return true
    })
    const cleaned = Object.fromEntries(cleanedEntries) as ChunkingOptions

    if (cleaned.llm_options_for_internal_steps) {
      const llm = cleaned.llm_options_for_internal_steps
      const hasValues = Object.values(llm).some((v) => v != null && v !== "")
      if (!hasValues) {
        delete cleaned.llm_options_for_internal_steps
      }
    }

    return cleaned
  }, [options, templateName])

  const handleChunk = useCallback(async () => {
    if (inputSource === "pdf") {
      if (!pdfFile || !pdfUrl) {
        message.warning(
          t("settings:chunkingPlayground.noPdf", "Please upload a PDF")
        )
        return
      }

      setIsLoading(true)
      setError(null)
      setChunks([])
      setPdfMetadata(null)

      const builtOptions = buildChunkingOptions()

      try {
        const response = await processPdfForChunking(pdfFile, builtOptions, {
          pdf_parsing_engine: pdfOptions.parsingEngine,
          enable_ocr: pdfOptions.enableOcr,
          ocr_backend: pdfOptions.ocrBackend,
          ocr_lang: pdfOptions.ocrLang,
          ocr_dpi: pdfOptions.ocrDpi,
          ocr_mode: pdfOptions.ocrMode,
          ocr_min_page_text_chars: pdfOptions.ocrMinPageTextChars,
          ocr_output_format: pdfOptions.ocrOutputFormat,
          ocr_prompt_preset: pdfOptions.ocrPromptPreset
        })
        const result = response.results?.[0]
        const status = String(result?.status || "").toLowerCase()
        if (!result || status === "error") {
          const errorMsg = result?.error || "PDF processing failed"
          setError(errorMsg)
          message.error(errorMsg)
          setLastResponse(null)
          return
        }

        const text =
          (result.conversion_text ?? result.content ?? "") as string
        setInputText(text)
        setChunks(result.chunks ?? [])
        setPdfMetadata(result.metadata ?? null)
        setLastResponse({
          chunks: result.chunks ?? [],
          original_file_name: result.input_ref ?? pdfFile.name,
          applied_options: builtOptions
        })
      } catch (err: unknown) {
        const errorMsg =
          err instanceof Error ? err.message : "PDF processing failed"
        setError(errorMsg)
        message.error(errorMsg)
        setLastResponse(null)
      } finally {
        setIsLoading(false)
      }
      return
    }

    if (!inputText.trim() && !inputFile) {
      message.warning(
        t("settings:chunkingPlayground.noInput", "Please provide text to chunk")
      )
      return
    }

    setIsLoading(true)
    setError(null)
    setChunks([])

    const builtOptions = buildChunkingOptions()

    try {
      let response: ChunkingResponse
      if (inputFile && requestMode === "file") {
        response = await chunkFile(inputFile, builtOptions)
      } else {
        response = await chunkText(inputText, builtOptions, inputFile?.name)
      }
      setChunks(response.chunks)
      setLastResponse(response)
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : "Chunking failed"
      setError(errorMsg)
      message.error(errorMsg)
      setLastResponse(null)
    } finally {
      setIsLoading(false)
    }
  }, [
    inputSource,
    pdfFile,
    pdfUrl,
    pdfOptions,
    inputText,
    inputFile,
    requestMode,
    buildChunkingOptions,
    t
  ])

  const handleExport = useCallback(() => {
    if (isLoading || !lastResponse?.chunks.length) return

    try {
      downloadBlob(
        new Blob([JSON.stringify(lastResponse, null, 2)], {
          type: "application/json"
        }),
        "chunking-result.json"
      )
      setError(null)
    } catch {
      setError(
        t(
          "settings:chunkingPlayground.exportFailed",
          "Unable to download chunks. Please try again."
        )
      )
    }
  }, [isLoading, lastResponse, t])

  const resetToDefaults = useCallback(() => {
    const defaults = capabilities?.default_options ?? DEFAULT_CHUNKING_OPTIONS
    setOptions({
      ...DEFAULT_CHUNKING_OPTIONS,
      ...(defaults || {})
    })
    setTemplateName("")
  }, [capabilities])

  const loadTemplateSettings = useCallback(async () => {
    if (!templateName.trim()) {
      message.warning(
        t(
          "settings:chunkingPlayground.loadTemplate.noTemplate",
          "Please select or enter a template name"
        )
      )
      return
    }

    setLoadingTemplate(true)
    try {
      const template = await getChunkingTemplate(templateName.trim())
      let config: Record<string, any>
      try {
        config = JSON.parse(template.template_json)
      } catch {
        throw new Error("Template contains invalid JSON")
      }
      const chunking = config.chunking || {}
      const chunkingConfig = chunking.config || {}

      const nextOptions: ChunkingOptions = {
        ...options
      }

      if (chunking.method) nextOptions.method = String(chunking.method)
      Object.entries(chunkingConfig).forEach(([key, value]) => {
        if (value !== undefined) {
          ;(nextOptions as Record<string, any>)[key] = value
        }
      })
      if (chunking.llm_options_for_internal_steps) {
        nextOptions.llm_options_for_internal_steps = {
          ...(chunking.llm_options_for_internal_steps || {})
        }
      }

      setOptions(nextOptions)

      message.success(
        t(
          "settings:chunkingPlayground.loadTemplate.success",
          "Template settings loaded"
        )
      )
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to load template"
      message.error(errorMsg)
    } finally {
      setLoadingTemplate(false)
    }
  }, [templateName, options, t])

  const handleSampleSelect = useCallback((text: string) => {
    setInputText(text)
    setInputFile(null)
    setPdfFile(null)
    setPdfUrl(null)
    setPdfMetadata(null)
    setInputSource("paste")
    setChunks([])
    setLastResponse(null)
  }, [])

  const handleMediaSelect = useCallback((content: string) => {
    setInputText(content)
    setInputFile(null)
    setPdfFile(null)
    setPdfUrl(null)
    setPdfMetadata(null)
    setInputSource("paste")
    setChunks([])
    setLastResponse(null)
  }, [])

  useEffect(() => {
    if (inputSource !== "upload" && inputFile) {
      setInputFile(null)
    }
  }, [inputSource, inputFile])

  useEffect(() => {
    if (inputSource !== "pdf" && pdfFile) {
      setPdfFile(null)
      setPdfUrl(null)
      setPdfMetadata(null)
    }
  }, [inputSource, pdfFile])

  useEffect(() => {
    if (inputSource === "pdf" && options.method) {
      const methodValue = String(options.method)
      if (!pdfMethodAllowlist.has(methodValue)) {
        setOptionValue("method", "words")
      }
    }
  }, [inputSource, options.method, pdfMethodAllowlist, setOptionValue])

  useEffect(() => {
    if (!inputFile) {
      setRequestMode("json")
    }
  }, [inputFile])

  useEffect(() => {
    if (inputSource !== "pdf" && viewMode === "split") {
      setViewMode("cards")
    }
  }, [inputSource, viewMode])

  useEffect(() => {
    if (!pdfUrl && viewMode === "split") {
      setViewMode("cards")
    }
  }, [pdfUrl, viewMode])

  useEffect(() => {
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl)
      }
    }
  }, [pdfUrl])

  const uploadProps: UploadProps = {
    accept: ".txt,.md,.text",
    maxCount: 1,
    beforeUpload: (file) => {
      // Read file content directly
      const reader = new FileReader()
      reader.onload = (e) => {
        const content = e.target?.result as string
        setInputText(content)
        setInputFile(file)
        setChunks([])
        setLastResponse(null)
      }
      reader.readAsText(file)
      return false // Prevent automatic upload
    },
    onRemove: () => {
      setInputFile(null)
      setInputText("")
      setLastResponse(null)
    }
  }

  const pdfUploadProps: UploadProps = {
    accept: ".pdf",
    maxCount: 1,
    beforeUpload: (file) => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl)
      }
      const url = URL.createObjectURL(file)
      setPdfFile(file)
      setPdfUrl(url)
      setPdfMetadata(null)
      setInputText("")
      setChunks([])
      setLastResponse(null)
      return false
    },
    onRemove: () => {
      setPdfFile(null)
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl)
      }
      setPdfUrl(null)
      setPdfMetadata(null)
      setInputText("")
      setLastResponse(null)
    }
  }

  const stats = React.useMemo(() => calculateChunkStats(chunks), [chunks])
  const methodLower = String(options.method || "").toLowerCase()
  const isCodeMethod = methodLower === "code"
  const isSemanticMethod = methodLower === "semantic"
  const isJsonMethod = methodLower.includes("json")
  const isRollingMethod = methodLower === "rolling_summarize"
  const isPropositionMethod = methodLower === "propositions"
  const isChapterMethod = methodLower === "ebook_chapters"
  const llmRequired =
    capabilities?.llm_required_methods?.includes(options.method ?? "") ?? false

  const renderInputSection = () => (
    <div className="space-y-4">
      <Segmented
        value={inputSource}
        onChange={(v) => setInputSource(v as InputSource)}
        options={[
          {
            value: "paste",
            label: t("settings:chunkingPlayground.inputSource.paste", "Paste Text")
          },
          {
            value: "upload",
            label: t("settings:chunkingPlayground.inputSource.upload", "Upload File")
          },
          {
            value: "pdf",
            label: t("settings:chunkingPlayground.inputSource.pdf", "Upload PDF")
          },
          {
            value: "sample",
            label: t("settings:chunkingPlayground.inputSource.sample", "Sample Text")
          },
          {
            value: "media",
            label: t(
              "settings:chunkingPlayground.inputSource.media",
              "From Media Library"
            )
          }
        ]}
      />

      {inputSource === "paste" && (
        <TextArea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder={t(
            "settings:chunkingPlayground.inputPlaceholder",
            "Paste text here..."
          )}
          rows={8}
          className="font-mono text-sm"
        />
      )}

      {inputSource === "upload" && (
        <div className="space-y-2">
          <Upload.Dragger {...uploadProps}>
            <p className="ant-upload-drag-icon">
              <UploadOutlined />
            </p>
            <p className="ant-upload-text">
              {t(
                "settings:chunkingPlayground.uploadDragText",
                "Click or drag file to upload"
              )}
            </p>
            <p className="ant-upload-hint">
              {t(
                "settings:chunkingPlayground.uploadHint",
                "Supports .txt and .md files"
              )}
            </p>
          </Upload.Dragger>
          {inputFile && (
            <div className="mt-2">
              <Text type="secondary">
                {inputFile.name} ({(inputText.length / 1024).toFixed(1)} KB)
              </Text>
            </div>
          )}
        </div>
      )}

      {inputSource === "pdf" && (
        <div className="space-y-3">
          <Upload.Dragger {...pdfUploadProps}>
            <p className="ant-upload-drag-icon">
              <UploadOutlined />
            </p>
            <p className="ant-upload-text">
              {t(
                "settings:chunkingPlayground.pdfDragText",
                "Click or drag PDF to upload"
              )}
            </p>
            <p className="ant-upload-hint">
              {t(
                "settings:chunkingPlayground.pdfHint",
                "PDF preview is required"
              )}
            </p>
          </Upload.Dragger>
          {pdfFile && (
            <div className="text-xs text-text-muted">
              {pdfFile.name}
            </div>
          )}
          <Collapse
            ghost
            items={[
              {
                key: "pdf-options",
                label: t(
                  "settings:chunkingPlayground.pdfOptions",
                  "PDF Parsing Options"
                ),
                children: (
                  <div className="space-y-3">
                    {ocrBackendsErrorMessage ? (
                      <Alert
                        type="warning"
                        showIcon
                        title={t(
                          "settings:chunkingPlayground.ocrBackendsError",
                          "Failed to load OCR backends"
                        )}
                        description={ocrBackendsErrorMessage}
                      />
                    ) : null}
                    <Form layout="vertical" size="small">
                      <Form.Item
                        label={t(
                          "settings:chunkingPlayground.pdfEngine",
                          "Parsing Engine"
                        )}>
                        <Select
                          value={pdfOptions.parsingEngine}
                          onChange={(v) => updatePdfOptions({ parsingEngine: v })}
                          options={pdfParsingEngineOptions}
                        />
                      </Form.Item>
                      <Form.Item
                        label={t(
                          "settings:chunkingPlayground.pdfEnableOcr",
                          "Enable OCR"
                        )}>
                        <Switch
                          checked={pdfOptions.enableOcr}
                          onChange={(checked) => updatePdfOptions({ enableOcr: checked })}
                        />
                      </Form.Item>
                      <Form.Item
                        label={t(
                          "settings:chunkingPlayground.pdfOcrMode",
                          "OCR Mode"
                        )}>
                        <Select
                          value={pdfOptions.ocrMode}
                          onChange={(v) =>
                            updatePdfOptions({ ocrMode: v as "always" | "fallback" })
                          }
                          options={[
                            { value: "fallback", label: "Fallback (low-text pages)" },
                            { value: "always", label: "Always" }
                          ]}
                          disabled={!pdfOptions.enableOcr}
                        />
                      </Form.Item>
                      <Form.Item
                        label={t(
                          "settings:chunkingPlayground.pdfOcrBackend",
                          "OCR Backend"
                        )}>
                        <Select
                          value={pdfOptions.ocrBackend}
                          onChange={(v) => updatePdfOptions({ ocrBackend: v })}
                          options={ocrBackendOptions}
                          allowClear
                          disabled={!pdfOptions.enableOcr}
                        />
                      </Form.Item>
                      <Form.Item
                        label={t(
                          "settings:chunkingPlayground.pdfOcrLang",
                          "OCR Language"
                        )}>
                        <Input
                          value={pdfOptions.ocrLang}
                          onChange={(e) => updatePdfOptions({ ocrLang: e.target.value })}
                          placeholder="eng"
                          disabled={!pdfOptions.enableOcr}
                        />
                      </Form.Item>
                      <Form.Item
                        label={t(
                          "settings:chunkingPlayground.pdfOcrDpi",
                          "OCR DPI"
                        )}>
                        <InputNumber
                          value={pdfOptions.ocrDpi}
                          min={72}
                          max={600}
                          onChange={(v) =>
                            updatePdfOptions({ ocrDpi: Number(v ?? 300) })
                          }
                          className="w-full"
                          disabled={!pdfOptions.enableOcr}
                        />
                      </Form.Item>
                      <Form.Item
                        label={t(
                          "settings:chunkingPlayground.pdfOcrMinPageTextChars",
                          "Min chars per page (fallback)"
                        )}>
                        <InputNumber
                          value={pdfOptions.ocrMinPageTextChars}
                          min={0}
                          max={2000}
                          onChange={(v) =>
                            updatePdfOptions({
                              ocrMinPageTextChars: Number(v ?? 40)
                            })
                          }
                          className="w-full"
                          disabled={!pdfOptions.enableOcr}
                        />
                      </Form.Item>
                      <Form.Item
                        label={t(
                          "settings:chunkingPlayground.pdfOcrOutputFormat",
                          "OCR Output Format"
                        )}>
                        <Select
                          value={pdfOptions.ocrOutputFormat}
                          onChange={(v) => updatePdfOptions({ ocrOutputFormat: v })}
                          allowClear
                          options={[
                            { value: "text", label: "text" },
                            { value: "markdown", label: "markdown" },
                            { value: "json", label: "json" }
                          ]}
                          disabled={!pdfOptions.enableOcr}
                        />
                      </Form.Item>
                      <Form.Item
                        label={t(
                          "settings:chunkingPlayground.pdfOcrPromptPreset",
                          "OCR Prompt Preset"
                        )}>
                        <Input
                          value={pdfOptions.ocrPromptPreset}
                          onChange={(e) =>
                            updatePdfOptions({ ocrPromptPreset: e.target.value })
                          }
                          placeholder="general"
                          disabled={!pdfOptions.enableOcr}
                        />
                      </Form.Item>
                    </Form>
                  </div>
                )
              }
            ]}
          />
        </div>
      )}

      {inputSource === "sample" && (
        <SampleTexts onSelect={handleSampleSelect} />
      )}

      {inputSource === "media" && (
        <MediaSelector onSelect={handleMediaSelect} />
      )}

      {/* Show text preview when not in paste mode but we have text */}
      {inputSource !== "paste" && inputText && (
        <div className="mt-2">
          <Text type="secondary" className="text-xs">
            {t("settings:chunkingPlayground.textLoaded", "Text loaded")}: {inputText.length}{" "}
            {t("settings:chunkingPlayground.characters", "characters")}
          </Text>
        </div>
      )}
    </div>
  )

  const renderSettingsSection = () => {
    const coreFields = ["method", "max_size", "overlap", "language"]
    const hiddenFields = new Set(["template_name"])
    const pdfFieldWhitelist = new Set([
      "method",
      "max_size",
      "overlap",
      "language",
      "adaptive",
      "multi_level",
      "custom_chapter_pattern"
    ])

    const resolveSchema = (schema: any, depth = 0): any => {
      if (depth > 10) return schema
      if (!schema) return {}
      if (schema.$ref) {
        const refKey = String(schema.$ref)
          .replace("#/$defs/", "")
          .replace("#/definitions/", "")
        return resolveSchema(schemaDefs[refKey], depth + 1)
      }
      if (schema.anyOf || schema.oneOf) {
        const variants = schema.anyOf || schema.oneOf || []
        const nonNull = variants.find((v: any) => v && v.type !== "null")
        return resolveSchema(nonNull || variants[0], depth + 1)
      }
      if (schema.allOf && schema.allOf.length > 0) {
        return resolveSchema(schema.allOf[0], depth + 1)
      }
      return schema
    }

    const isFieldDisabled = (fieldName: string) => {
      if (fieldName === "code_mode") return !isCodeMethod
      if (
        fieldName === "semantic_similarity_threshold" ||
        fieldName === "semantic_overlap_sentences"
      ) {
        return !isSemanticMethod
      }
      if (
        fieldName === "json_chunkable_data_key" ||
        fieldName === "enable_frontmatter_parsing" ||
        fieldName === "frontmatter_sentinel_key"
      ) {
        return !isJsonMethod
      }
      if (fieldName === "summarization_detail") return !isRollingMethod
      if (fieldName.startsWith("proposition_")) return !isPropositionMethod
      return false
    }

    const renderField = (
      name: string,
      schema: any,
      parent?: string
    ): React.ReactNode => {
      const resolved = resolveSchema(schema)
      if (!resolved) return null

      const fieldLabel =
        resolved.title ||
        name
          .replace(/_/g, " ")
          .replace(/\b\w/g, (c: string) => c.toUpperCase())
      const description = resolved.description
      const fieldKey = parent ? `${parent}.${name}` : name
      const rawValue = parent
        ? ((options as Record<string, any>)[parent] || {})[name]
        : (options as Record<string, any>)[name]
      const displayValue =
        rawValue !== undefined && rawValue !== null ? rawValue : resolved.default

      if (resolved.type === "object" && resolved.properties) {
        const nestedProps = resolved.properties as Record<string, any>
        return (
          <div key={fieldKey} className="space-y-2">
            <Text type="secondary" className="text-xs">
              {fieldLabel}
            </Text>
            {description && (
              <Text type="secondary" className="text-xs">
                {description}
              </Text>
            )}
            <div className="space-y-2 pl-2">
              {Object.entries(nestedProps).map(([childName, childSchema]) =>
                renderField(childName, childSchema, name)
              )}
            </div>
          </div>
        )
      }

      const disabled =
        (parent === "llm_options_for_internal_steps" && !llmRequired) ||
        isFieldDisabled(name)

      if (resolved.enum || name === "method" || name === "language") {
        const enumValues = resolved.enum as Array<string | number> | undefined
        const selectOptions =
          name === "method"
            ? inputSource === "pdf"
              ? pdfMethodOptions
              : methodOptions
            : name === "language"
            ? languageOptions
            : (enumValues || []).map((val) => ({
                value: val,
                label: String(val)
              }))
        return (
          <Form.Item key={fieldKey} label={fieldLabel} help={description}>
            <Select
              value={displayValue}
              onChange={(v) =>
                parent
                  ? setNestedOptionValue(parent, name, v)
                  : setOptionValue(name, v)
              }
              options={selectOptions}
              allowClear={resolved.default === undefined}
              disabled={disabled}
            />
          </Form.Item>
        )
      }

      if (resolved.type === "boolean") {
        return (
          <Form.Item key={fieldKey} label={fieldLabel} help={description}>
            <Switch
              checked={Boolean(displayValue)}
              onChange={(v) =>
                parent
                  ? setNestedOptionValue(parent, name, v)
                  : setOptionValue(name, v)
              }
              disabled={disabled}
            />
          </Form.Item>
        )
      }

      if (resolved.type === "integer" || resolved.type === "number") {
        return (
          <Form.Item key={fieldKey} label={fieldLabel} help={description}>
            <InputNumber
              value={displayValue}
              onChange={(v) =>
                parent
                  ? setNestedOptionValue(parent, name, v)
                  : setOptionValue(name, v)
              }
              min={resolved.minimum}
              max={resolved.maximum}
              step={resolved.type === "integer" ? 1 : resolved.multipleOf || 0.1}
              className="w-full"
              disabled={disabled}
            />
          </Form.Item>
        )
      }

      return (
        <Form.Item key={fieldKey} label={fieldLabel} help={description}>
          <Input
            value={displayValue ?? ""}
            onChange={(e) =>
              parent
                ? setNestedOptionValue(parent, name, e.target.value)
                : setOptionValue(name, e.target.value)
            }
            disabled={disabled}
          />
        </Form.Item>
      )
    }

    const availableFields = Object.keys(schemaProps).filter(
      (field) => !hiddenFields.has(field)
    )
    const visibleFields =
      inputSource === "pdf"
        ? availableFields.filter((f) => pdfFieldWhitelist.has(f))
        : availableFields
    const advancedFields = visibleFields.filter(
      (field) => !coreFields.includes(field)
    )

    return (
      <Card
        size="small"
        title={t("settings:chunkingPlayground.settingsTitle", "Settings")}
        className="h-full">
        <Form layout="vertical" size="small">
          {coreFields
            .filter((field) => visibleFields.includes(field))
            .map((field) => renderField(field, schemaProps[field]))}

          {llmRequired && (
            <Text type="warning" className="text-xs">
              {t(
                "settings:chunkingPlayground.advanced.methodRequiresLlm",
                "This method requires an LLM provider on the server."
              )}
            </Text>
          )}

          <Collapse
            ghost
            items={[
              {
                key: "advanced",
                label: t(
                  "settings:chunkingPlayground.advanced.title",
                  "Advanced Options"
                ),
                children: (
                  <div className="space-y-3">
                    <Form.Item
                      label={t(
                        "settings:chunkingPlayground.advanced.templateNameLabel",
                        "Template name"
                      )}>
                      <div className="flex gap-2">
                        <AutoComplete
                          value={templateName}
                          onChange={setTemplateName}
                          options={templateOptions}
                          placeholder={t(
                            "settings:chunkingPlayground.advanced.templateNamePlaceholder",
                            "Select or type a template name"
                          )}
                          filterOption={(inputValue, option) =>
                            (option?.value ?? "")
                              .toString()
                              .toLowerCase()
                              .includes(inputValue.toLowerCase())
                          }
                          className="flex-1"
                        />
                        <Button
                          size="small"
                          icon={<DownloadOutlined />}
                          onClick={loadTemplateSettings}
                          loading={loadingTemplate}
                          disabled={!templateName.trim()}>
                          {t(
                            "settings:chunkingPlayground.advanced.loadSettings",
                            "Load"
                          )}
                        </Button>
                        <Button
                          size="small"
                          onClick={() => refetchTemplates()}
                          loading={templatesLoading}>
                          {t("common:refresh", "Refresh")}
                        </Button>
                      </div>
                    </Form.Item>

                    <Text type="secondary" className="text-xs">
                      {t(
                        "settings:chunkingPlayground.advanced.templateOverridesHint",
                        "Template defaults are applied first; form values override them."
                      )}
                    </Text>

                    {templatesError && (templatesError as Error)?.message && (
                      <Alert
                        type="warning"
                        showIcon
                        title={(templatesError as Error)?.message}
                      />
                    )}

                    {inputFile && (
                      <Form.Item
                        label={t(
                          "settings:chunkingPlayground.advanced.requestModeLabel",
                          "Request Endpoint"
                        )}>
                        <Space orientation="vertical" size={4} className="w-full">
                          <Segmented
                            value={requestMode}
                            onChange={(v) => setRequestMode(v as RequestMode)}
                            options={[
                              {
                                value: "json",
                                label: t(
                                  "settings:chunkingPlayground.advanced.requestModeJson",
                                  "JSON (/chunk_text)"
                                )
                              },
                              {
                                value: "file",
                                label: t(
                                  "settings:chunkingPlayground.advanced.requestModeFile",
                                  "Multipart (/chunk_file)"
                                )
                              }
                            ]}
                          />
                          <Text type="secondary" className="text-xs">
                            {t(
                              "settings:chunkingPlayground.advanced.requestModeHint",
                              "Use JSON for full option support; multipart matches /chunk_file."
                            )}
                          </Text>
                        </Space>
                      </Form.Item>
                    )}

                    <Divider className="my-2" />

                    {advancedFields.map((field) =>
                      renderField(field, schemaProps[field])
                    )}
                  </div>
                )
              }
            ]}
          />

          <Divider className="my-2" />

          <Space wrap>
            <Button size="small" onClick={resetToDefaults}>
              {t(
                "settings:chunkingPlayground.advanced.resetDefaults",
                "Reset to defaults"
              )}
            </Button>
            <Button
              size="small"
              icon={<SaveOutlined />}
              onClick={() => setSaveAsTemplateOpen(true)}>
              {t(
                "settings:chunkingPlayground.saveAsTemplate.button",
                "Save as Template"
              )}
            </Button>
          </Space>
        </Form>
      </Card>
    )
  }

  const renderResultsSection = () => (
    <div className="space-y-4">
      {/* View toggle and stats */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <Segmented
          value={viewMode}
          onChange={(v) => setViewMode(v as ViewMode)}
          options={[
            {
              value: "cards",
              label: t("settings:chunkingPlayground.viewCards", "Cards")
            },
            {
              value: "inline",
              label: t("settings:chunkingPlayground.viewInline", "Inline")
            },
            ...(pdfUrl
              ? [
                  {
                    value: "split",
                    label: t("settings:chunkingPlayground.viewSplit", "Split")
                  }
                ]
              : [])
          ]}
        />

        <Space wrap>
          {chunks.length > 0 && (
            <Text type="secondary">
              {t("settings:chunkingPlayground.stats", "{{count}} chunks, avg {{avgSize}} chars", {
                count: stats.count,
                avgSize: stats.avgCharCount
              })}
            </Text>
          )}
          {!!lastResponse?.chunks.length && (
            <Button
              icon={<DownloadOutlined aria-hidden="true" />}
              onClick={handleExport}
              disabled={isLoading}>
              {t("settings:chunkingPlayground.exportJson", "Export JSON")}
            </Button>
          )}
        </Space>
      </div>

      {/* Error display */}
      {error && (
        <Alert
          type="error"
          title={error}
          closable
          onClose={() => setError(null)}
        />
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="flex justify-center py-8">
          <Spin size="large" />
        </div>
      )}

      {/* Results */}
      {!isLoading && chunks.length > 0 && (
        <>
          {viewMode === "split" ? (
            <SplitView
              pdfUrl={pdfUrl}
              chunks={chunks}
              highlightedIndex={highlightedChunkIndex}
              onChunkHover={setHighlightedChunkIndex}
            />
          ) : viewMode === "cards" ? (
            <ChunkCardView
              chunks={chunks}
              highlightedIndex={highlightedChunkIndex}
              onChunkHover={setHighlightedChunkIndex}
            />
          ) : (
            <ChunkInlineView
              originalText={inputText}
              chunks={chunks}
              highlightedIndex={highlightedChunkIndex}
              onChunkClick={setHighlightedChunkIndex}
            />
          )}
        </>
      )}

      {!isLoading && lastResponse && (
        <Collapse
          ghost
          items={[
            {
              key: "applied",
              label: t(
                "settings:chunkingPlayground.appliedOptionsTitle",
                "Applied options"
              ),
              children: (
                <pre className="text-xs bg-surface2 rounded p-2 overflow-x-auto">
                  {JSON.stringify(lastResponse.applied_options, null, 2)}
                </pre>
              )
            },
            {
              key: "meta",
              label: t(
                "settings:chunkingPlayground.responseMetaTitle",
                "Response metadata"
              ),
              children: (
                <div className="text-xs text-text-muted space-y-1">
                  {lastResponse.original_file_name && (
                    <div>
                      {lastResponse.original_file_name}
                    </div>
                  )}
                  <div>{chunks.length} chunks</div>
                </div>
              )
            }
          ]}
        />
      )}

      {!isLoading && pdfMetadata && (
        <Collapse
          ghost
          items={[
            {
              key: "pdf-metadata",
              label: t(
                "settings:chunkingPlayground.extractionMetadata",
                "Extraction metadata"
              ),
              children: (
                <pre className="text-xs bg-surface2 rounded p-2 overflow-x-auto">
                  {JSON.stringify(pdfMetadata, null, 2)}
                </pre>
              )
            }
          ]}
        />
      )}

      {/* Empty state */}
      {!isLoading && chunks.length === 0 && !error && (
        <div className="text-center py-8 text-text-muted ">
          <ScissorOutlined className="text-4xl mb-2" />
          <p>
            {t(
              "settings:chunkingPlayground.emptyState",
              "Enter text and click 'Chunk Text' to see results"
            )}
          </p>
        </div>
      )}
    </div>
  )

  const renderSingleMode = () => (
    isDesktop ? (
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Input section - 3 cols */}
        <div className="lg:col-span-3 space-y-4">
          {renderInputSection()}

          <Button
            type="primary"
            icon={<ScissorOutlined />}
            onClick={handleChunk}
            loading={isLoading}
            disabled={
              inputSource === "pdf"
                ? !pdfFile
                : !inputText.trim() && !inputFile
            }
            size="large">
            {t("settings:chunkingPlayground.chunkButton", "Chunk Text")}
          </Button>

          {renderResultsSection()}
        </div>

        {/* Settings section - 1 col */}
        <div className="lg:col-span-1">{renderSettingsSection()}</div>
      </div>
    ) : (
      <div className="space-y-4">
        {renderInputSection()}
        {renderSettingsSection()}
        <Button
          type="primary"
          icon={<ScissorOutlined />}
          onClick={handleChunk}
          loading={isLoading}
          disabled={
            inputSource === "pdf"
              ? !pdfFile
              : !inputText.trim() && !inputFile
          }
          size="large">
          {t("settings:chunkingPlayground.chunkButton", "Chunk Text")}
        </Button>
        {renderResultsSection()}
      </div>
    )
  )

  return (
    <div className={className}>
      <div className="mb-6">
        <Title level={3}>
          {t("settings:chunkingPlayground.title", "Chunking Playground")}
        </Title>
        <Text type="secondary">
          {t(
            "settings:chunkingPlayground.description",
            "Experiment with different chunking settings to see how text gets split"
          )}
        </Text>
      </div>

      <Tabs
        activeKey={playgroundMode}
        onChange={(k) => setPlaygroundMode(k as PlaygroundMode)}
        items={[
          {
            key: "single",
            label: t("settings:chunkingPlayground.tabSingle", "Single"),
            children: renderSingleMode()
          },
          {
            key: "compare",
            label: t("settings:chunkingPlayground.tabCompare", "Compare"),
            children: (
              <CompareView
                inputText={inputText}
                onTextChange={setInputText}
                capabilities={capabilities}
              />
            )
          },
          {
            key: "templates",
            label: t("settings:chunkingPlayground.tabTemplates", "Templates"),
            children: (
              <ChunkingTemplatesPanel />
            )
          },
          {
            key: "capabilities",
            label: t("settings:chunkingPlayground.tabCapabilities", "Capabilities"),
            children: (
              <ChunkingCapabilitiesPanel
                capabilities={capabilities}
                loading={capabilitiesLoading}
                onRefresh={() => refetchCapabilities()}
              />
            )
          }
        ]}
      />

      <SaveAsTemplateModal
        open={saveAsTemplateOpen}
        onClose={() => setSaveAsTemplateOpen(false)}
        chunkingOptions={buildChunkingOptions()}
      />
    </div>
  )
}

export default ChunkingPlayground
