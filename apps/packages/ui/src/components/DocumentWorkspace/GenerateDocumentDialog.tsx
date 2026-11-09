import React from "react"
import { Alert, Button, Input, Modal } from "antd"
import { ModelSelect } from "@/components/Common/ModelSelect"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"
import { useChatModelCatalog } from "@/hooks/useChatModelCatalog"
import {
  isConfiguredUsableModel,
  resolveModelSelectorSelection
} from "@/hooks/playground/modelSelectorUtils"
import { useStoreChatModelSettings } from "@/store/model"
import {
  createDocumentClient,
  parseDocumentDraft,
  type DocumentDraft
} from "@/services/document-generation"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { tldwClient } from "@/services/tldw"
import { downloadBlob } from "@/utils/download-blob"
import { formatErrorMessage } from "@/utils/format-error-message"
import {
  extractMediaId,
  extractUploadResult,
  extractWarnings,
  getDocumentUploadProblem
} from "./document-utils"

export default function GenerateDocumentDialog({
  operation,
  onClose,
  onLibrary,
  onOpenSaved
}: {
  operation: QuickIngestOperation
  onClose: () => void
  onLibrary: () => void
  onOpenSaved: (id: number) => Promise<boolean>
}) {
  const { selectedModel, selectedModelIsLoading } = useSelectedModel()
  const catalog = useChatModelCatalog()
  const settings = useStoreChatModelSettings()
  const selection = resolveModelSelectorSelection(
    catalog.data,
    selectedModel,
    settings.apiProvider
  )
  const ready = Boolean(
    selection.modelId &&
    selection.provider &&
    isConfiguredUsableModel(selection.model) &&
    catalog.isSuccess &&
    !catalog.isFetching &&
    !selectedModelIsLoading
  )
  const api = React.useMemo(() => createDocumentClient(operation), [operation])
  const [instructions, setInstructions] = React.useState("")
  const [draft, setDraft] = React.useState<DocumentDraft | null>(null)
  const [busy, setBusy] = React.useState<"generation" | "save" | "open" | null>(
    null
  )
  const [error, setError] = React.useState<string | null>(null)
  const [notice, setNotice] = React.useState<string | null>(null)
  const [saved, setSaved] = React.useState<{
    id: number
    original: boolean
  } | null>(null)
  const [uncertain, setUncertain] = React.useState(false)
  const work = React.useRef<AbortController | null>(null)
  const retained = React.useRef<{ key: string; file: File } | null>(null)
  const mounted = React.useRef(true)
  const current = () => mounted.current && operation.isCurrent()
  React.useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      work.current?.abort()
    }
  }, [])
  React.useEffect(() => {
    if (busy !== "save") return
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = ""
    }
    window.addEventListener("beforeunload", warn)
    return () => window.removeEventListener("beforeunload", warn)
  }, [busy])
  const stop = () => {
    work.current?.abort()
    work.current = null
    setBusy(null)
    setNotice(
      draft
        ? "Generation stopped. Your previous draft is still available."
        : "Generation stopped. The saved document is unchanged."
    )
  }
  const generate = async () => {
    if (work.current || !ready || !current() || saved || uncertain) return
    const controller = new AbortController()
    work.current = controller
    setBusy("generation")
    setError(null)
    setNotice(null)
    try {
      const value = await api.generate(
        {
          instructions,
          model: selection.modelId!,
          provider: selection.provider!,
          temperature: settings.temperature,
          topP: settings.topP,
          maxTokens: settings.numPredict
        },
        controller.signal
      )
      if (
        current() &&
        work.current === controller &&
        !controller.signal.aborted
      ) {
        setDraft(value)
        retained.current = null
      }
    } catch (failure) {
      if (
        current() &&
        work.current === controller &&
        !controller.signal.aborted
      )
        setError(formatErrorMessage(failure, "Document generation failed."))
    } finally {
      if (work.current === controller) {
        work.current = null
        if (current()) setBusy(null)
      }
    }
  }
  const openSaved = async (id: number) => {
    const opened = await onOpenSaved(id)
    if (current()) {
      if (opened) onClose()
      else
        setError(
          "The PDF was saved, but could not be opened. Use Open saved document without uploading again."
        )
    }
  }
  const save = async () => {
    if (!draft || work.current || saved || uncertain || !current()) return
    const controller = new AbortController()
    work.current = controller
    setBusy("save")
    setError(null)
    setNotice(null)
    let uploading = false
    try {
      let value: DocumentDraft
      try {
        value = parseDocumentDraft(draft)
      } catch {
        throw new Error(
          "Use a nonblank title and headings (up to 200 characters), and 1–10 nonblank pages of up to 6,000 characters each."
        )
      }
      const key = JSON.stringify(value)
      if (retained.current?.key !== key) {
        const blob = await api.render(value, controller.signal)
        if (!current() || controller.signal.aborted) return
        retained.current = {
          key,
          file: new File([blob], `document-${crypto.randomUUID()}.pdf`, {
            type: "application/pdf"
          })
        }
      }
      operation.assertCurrent()
      controller.signal.throwIfAborted()
      uploading = true
      const response = await tldwClient.uploadMedia(
        retained.current.file,
        { media_type: "pdf", keep_original_file: true, title: value.title },
        {
          ...operation,
          signal: AbortSignal.any([operation.signal, controller.signal])
        }
      )
      if (!current()) return
      const id = Number(extractMediaId(response))
      if (!Number.isSafeInteger(id) || id < 1)
        throw new Error("No saved document ID was returned.")
      uploading = false
      const result = extractUploadResult(response)
      const importProblem = getDocumentUploadProblem(result)
      const original =
        !importProblem &&
        result?.original_file_stored !== false &&
        !extractWarnings(result).some((warning) =>
          warning.includes("original file")
        )
      setSaved({ id, original })
      if (importProblem) {
        setError(
          importProblem +
            " A Media record remains. Open Library to inspect it; your PDF is retained and will not be uploaded again automatically."
        )
        return
      }
      if (!original) {
        setError(
          "The document was saved, but its original PDF was not preserved. Open Library to inspect it; another copy will not be uploaded automatically."
        )
        return
      }
      await openSaved(id)
    } catch (failure) {
      if (current()) {
        if (uploading) {
          setUncertain(true)
          setError(
            "The save acknowledgement was lost. Check Library before creating another copy. Your PDF is retained below."
          )
        } else
          setError(
            formatErrorMessage(failure, "The document could not be saved.")
          )
      }
    } finally {
      if (work.current === controller) {
        work.current = null
        if (current()) setBusy(null)
      }
    }
  }
  const retryOpen = async () => {
    if (!saved || busy || !current()) return
    setBusy("open")
    try {
      await openSaved(saved.id)
    } catch {
      if (current())
        setError(
          "The PDF is saved. Open it from Library when the connection is restored."
        )
    } finally {
      if (current()) setBusy(null)
    }
  }
  const editPage = (index: number, field: "heading" | "text", value: string) =>
    setDraft(
      (previous) =>
        previous && {
          ...previous,
          pages: previous.pages.map((page, i) =>
            i === index ? { ...page, [field]: value } : page
          )
        }
    )
  const locked = Boolean(busy || saved || uncertain)
  return (
    <Modal
      title="Generate document"
      open
      centered
      width={760}
      onCancel={onClose}
      closable={busy !== "save" && busy !== "open"}
      keyboard={busy !== "save" && busy !== "open"}
      maskClosable={busy !== "save" && busy !== "open"}
      styles={{ body: { maxHeight: "65vh", overflowY: "auto" } }}
      footer={[
        <Button
          key="cancel"
          onClick={onClose}
          disabled={busy === "save" || busy === "open"}
        >
          Cancel
        </Button>,
        busy === "generation" ? (
          <Button key="stop" onClick={stop}>
            Stop
          </Button>
        ) : (
          <Button
            key="generate"
            onClick={() => void generate()}
            disabled={!ready || !instructions.trim() || locked}
          >
            Generate draft
          </Button>
        ),
        saved?.original ? (
          <Button
            key="open"
            type="primary"
            onClick={() => void retryOpen()}
            loading={busy === "open"}
          >
            Open saved document
          </Button>
        ) : (
          <Button
            key="save"
            type="primary"
            onClick={() => void save()}
            disabled={!draft || locked}
            loading={busy === "save"}
          >
            Save PDF
          </Button>
        )
      ]}
    >
      <div className="space-y-4">
        <p className="text-sm text-text-secondary">
          Generate and edit a plain-text draft, then save it as a new PDF. Your
          current document stays unchanged.
        </p>
        <ModelSelect showSelectedName />
        {!ready && (
          <Alert
            type="info"
            showIcon
            title="Select a configured available model to generate a document."
          />
        )}
        <label className="block">
          Document instructions
          <Input.TextArea
            aria-label="Document instructions"
            value={instructions}
            maxLength={4000}
            rows={3}
            disabled={locked}
            onChange={(event) => setInstructions(event.target.value)}
          />
        </label>
        {error && <Alert type="error" showIcon title={error} />}
        {notice && <Alert type="info" showIcon title={notice} />}
        {(saved || uncertain) && (
          <div className="flex gap-2">
            <Button onClick={onLibrary} disabled={Boolean(busy)}>
              Open Library
            </Button>
            {retained.current && (
              <Button
                onClick={() => {
                  if (current())
                    downloadBlob(
                      retained.current!.file,
                      retained.current!.file.name
                    )
                }}
              >
                Download retained PDF
              </Button>
            )}
          </div>
        )}
        {draft && (
          <div className="space-y-4">
            <label className="block">
              Document title
              <Input
                aria-label="Document title"
                value={draft.title}
                maxLength={200}
                disabled={locked}
                onChange={(event) =>
                  setDraft({ ...draft, title: event.target.value })
                }
              />
            </label>
            {draft.pages.map((page, index) => (
              <fieldset
                key={index}
                className="space-y-2 rounded border border-border p-3"
              >
                <legend>Page {index + 1}</legend>
                <label className="block">
                  Heading
                  <Input
                    aria-label={`Page ${index + 1} heading`}
                    value={page.heading}
                    maxLength={200}
                    disabled={locked}
                    onChange={(event) =>
                      editPage(index, "heading", event.target.value)
                    }
                  />
                </label>
                <label className="block">
                  Text
                  <Input.TextArea
                    aria-label={`Page ${index + 1} text`}
                    value={page.text}
                    maxLength={6000}
                    rows={7}
                    disabled={locked}
                    onChange={(event) =>
                      editPage(index, "text", event.target.value)
                    }
                  />
                </label>
              </fieldset>
            ))}
          </div>
        )}
      </div>
    </Modal>
  )
}
