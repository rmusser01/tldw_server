import React from "react"
import { Image } from "antd"
import { RotateCcw, Trash2 } from "lucide-react"
import { EditMessageForm } from "./EditMessageForm"
import { parseReasoning } from "@/libs/reasoning"
import { highlightText } from "@/utils/text-highlight"
import { ReasoningBlock } from "./ReasoningBlock"
import { DynamicMessageRenderer } from "@/components/Common/DynamicUI/DynamicMessageRenderer"
import { DynamicUISourceFallback } from "@/components/Common/DynamicUI/DynamicUISourceFallback"
import { normalizeDynamicUIEnvelope } from "@/utils/dynamic-ui"
import type { TFunction } from "i18next"
import type { ChatErrorPayload } from "@/utils/chat-error-message"
import type { ImageGenerationRequestSnapshot } from "@/utils/image-generation-chat"
import type { DynamicUISurface } from "@/types/dynamic-ui"
import type { MessageMetadataExtra } from "@/store/option/types"
import { useStorage } from "@plasmohq/storage/hook"
import { DEFAULT_CHAT_SETTINGS } from "@/types/chat-settings"

const Markdown = React.lazy(() => import("../../Common/Markdown"))

const MARKDOWN_BASE_CLASSES =
  "prose break-words text-message dark:prose-invert prose-p:leading-relaxed prose-pre:p-0 dark:prose-dark max-w-none"

const ErrorBubble: React.FC<{
  payload: ChatErrorPayload
  toggleLabels: { show: string; hide: string }
  recoveryActions?: Array<{
    id: string
    label: string
    onClick: () => void
  }>
}> = ({ payload, toggleLabels, recoveryActions = [] }) => {
  const [showDetails, setShowDetails] = React.useState(false)

  return (
    <div
      role="alert"
      aria-live="assertive"
      className="rounded-md border border-danger/30 bg-danger/10 p-3 text-sm text-danger">
      <p className="font-semibold">{payload.summary}</p>
      {payload.hint && (
        <p className="mt-1 text-xs text-danger">
          {payload.hint}
        </p>
      )}
      {payload.detail && (
        <button
          type="button"
          onClick={() => setShowDetails((prev) => !prev)}
          title={showDetails ? toggleLabels.hide : toggleLabels.show}
          className="mt-2 text-xs font-medium text-danger underline hover:text-danger">
          {showDetails ? toggleLabels.hide : toggleLabels.show}
        </button>
      )}
      {showDetails && payload.detail && (
        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-danger/10 p-2 text-xs text-danger">
          {payload.detail}
        </pre>
      )}
      {recoveryActions.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <span className="sr-only">
            Recommended next actions:{" "}
            {recoveryActions.map((action) => action.label).join(", ")}
          </span>
          {recoveryActions.map((action) => (
            <button
              key={action.id}
              type="button"
              onClick={action.onClick}
              className="rounded border border-danger/40 bg-surface px-2 py-1 text-[11px] font-medium text-danger transition hover:bg-danger/10"
            >
              {action.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export interface MessageContentProps {
  t: TFunction
  message: string
  message_type?: string
  isBot: boolean
  isStreaming: boolean
  editMode: boolean
  onEditFormSubmit: (value: string, isSend: boolean) => void
  onCloseEdit: () => void

  // Display state (from useMessageState)
  errorPayload: ChatErrorPayload | null
  shouldRenderStreamingPlainText: boolean
  renderGreetingMarkdown: boolean
  assistantTextClass: string
  chatTextClass: string
  searchQuery?: string

  // Error recovery
  errorRecoveryActions: Array<{ id: string; label: string; onClick: () => void }>

  // Reasoning
  openReasoning?: boolean
  reasoningTimeTaken?: number

  // Images
  images?: string[]
  messageId?: string
  metadataExtra?: MessageMetadataExtra
  dynamicUISurface?: DynamicUISurface
  onDynamicUIAction?: (payload: unknown) => void
  showInlineImageActions: boolean
  canRegenerateImage: boolean
  imageGenerationMetadata: { request: ImageGenerationRequestSnapshot | null } | null
  onRegenerateImage?: (payload: {
    messageId?: string
    imageIndex: number
    imageUrl: string
    request: ImageGenerationRequestSnapshot | null
  }) => void | Promise<void>
  onDeleteImage?: (payload: {
    messageId?: string
    imageIndex: number
    imageUrl: string
  }) => void
}

export const MessageContent = React.memo(function MessageContent(
  props: MessageContentProps
) {
  const {
    t,
    message,
    message_type,
    isBot,
    isStreaming,
    editMode,
    onEditFormSubmit,
    onCloseEdit,
    errorPayload,
    shouldRenderStreamingPlainText,
    renderGreetingMarkdown,
    assistantTextClass,
    chatTextClass,
    searchQuery,
    errorRecoveryActions,
    openReasoning,
    reasoningTimeTaken,
    images,
    messageId,
    metadataExtra,
    dynamicUISurface,
    onDynamicUIAction,
    showInlineImageActions,
    canRegenerateImage,
    imageGenerationMetadata,
    onRegenerateImage,
    onDeleteImage,
  } = props
  const [renderMermaidDiagrams] = useStorage(
    "renderMermaidDiagrams",
    DEFAULT_CHAT_SETTINGS.renderMermaidDiagrams
  )
  const enableAssistantMermaidDiagrams =
    isBot && renderMermaidDiagrams !== false && !isStreaming
  const dynamicUIEnvelope = normalizeDynamicUIEnvelope(metadataExtra?.dynamic_ui)
  const resolvedDynamicUISurface = dynamicUISurface ?? "artifact"

  return (
    <>
      <div className="flex flex-grow flex-col">
        {!editMode ? (
          isBot ? (
            errorPayload ? (
              <ErrorBubble
                payload={errorPayload}
                toggleLabels={{
                  show: t(
                    "error.showDetails",
                    "Show technical details"
                  ) as string,
                  hide: t(
                    "error.hideDetails",
                    "Hide technical details"
                  ) as string
                }}
                recoveryActions={errorRecoveryActions}
              />
            ) : shouldRenderStreamingPlainText ? (
              <p
                data-testid="playground-streaming-plain-text"
                className={`text-body text-text-muted whitespace-pre-wrap ${assistantTextClass}`}
              >
                {message}
              </p>
            ) : !isStreaming && dynamicUIEnvelope ? (
              messageId ? (
                <DynamicMessageRenderer
                  envelope={dynamicUIEnvelope}
                  sourceMessageId={messageId}
                  sourceText={message}
                  surface={resolvedDynamicUISurface}
                  onAction={onDynamicUIAction}
                />
              ) : (
                <DynamicUISourceFallback
                  source={message}
                  error="Dynamic UI actions require a saved assistant message id."
                />
              )
            ) : renderGreetingMarkdown ? (
              <React.Suspense
                fallback={
                  <p
                    className={`text-body text-text-muted ${assistantTextClass}`}>
                    {t("loading.content")}
                  </p>
                }>
                <Markdown
                  message={message}
                  className={`${MARKDOWN_BASE_CLASSES} ${assistantTextClass}`}
                  searchQuery={searchQuery}
                  codeBlockVariant="compact"
                  enableMermaidDiagrams={enableAssistantMermaidDiagrams}
                />
              </React.Suspense>
            ) : (
              <>
                {parseReasoning(message).map((e, i) => {
                  if (e.type === "reasoning") {
                    return (
                      <ReasoningBlock
                        key={`reasoning-${i}`}
                        content={e.content}
                        isStreaming={isStreaming}
                        reasoningRunning={e.reasoning_running}
                        openReasoning={openReasoning}
                        reasoningTimeTaken={reasoningTimeTaken}
                        assistantTextClass={assistantTextClass}
                        markdownBaseClasses={MARKDOWN_BASE_CLASSES}
                        searchQuery={searchQuery}
                        t={t}
                        enableMermaidDiagrams={enableAssistantMermaidDiagrams}
                      />
                    )
                  }

                  return (
                    <React.Suspense
                      key={`message-${i}`}
                      fallback={
                        <p
                          className={`text-body text-text-muted ${assistantTextClass}`}>
                          {t("loading.content")}
                        </p>
                      }>
                      <Markdown
                        message={e.content}
                        className={`${MARKDOWN_BASE_CLASSES} ${assistantTextClass}`}
                        searchQuery={searchQuery}
                        codeBlockVariant="github"
                        enableMermaidDiagrams={enableAssistantMermaidDiagrams}
                      />
                    </React.Suspense>
                  )
                })}
              </>
            )
          ) : (
            <p
              className={`prose max-w-none dark:prose-invert whitespace-pre-line prose-p:leading-relaxed prose-pre:p-0 dark:prose-dark ${chatTextClass} ${
                message_type &&
                "italic text-text-muted text-body"
              }
              `}>
              {searchQuery
                ? highlightText(message, searchQuery)
                : message}
            </p>
          )
        ) : (
          <EditMessageForm
            value={message}
            onSumbit={onEditFormSubmit}
            onClose={onCloseEdit}
            isBot={isBot}
          />
        )}
      </div>
      {/* images if available */}
      {images &&
        images.filter((img) => img.length > 0).length > 0 && (
          <div className="mt-2 flex flex-wrap gap-3">
            {images
              .filter((image) => image.length > 0)
              .map((image, index) => (
                <div key={index} className="group relative">
                  <Image
                    src={image}
                    alt="Uploaded Image"
                    width={180}
                    className="rounded-md relative"
                  />
                  {showInlineImageActions && (
                    <div className="pointer-events-none absolute right-2 top-2 flex items-center gap-1 rounded-full border border-border/70 bg-surface/90 px-1 py-1 opacity-0 shadow-sm transition group-hover:opacity-100 group-focus-within:opacity-100">
                      {canRegenerateImage && (
                        <button
                          type="button"
                          className="pointer-events-auto inline-flex h-7 w-7 items-center justify-center rounded-full text-text-muted transition hover:bg-surface2 hover:text-text"
                          aria-label={t(
                            "playground:imageGeneration.regenerateImage",
                            "Regenerate image"
                          ) as string}
                          title={t(
                            "playground:imageGeneration.regenerateImage",
                            "Regenerate image"
                          ) as string}
                          onClick={() => {
                            void onRegenerateImage?.({
                              messageId,
                              imageIndex: index,
                              imageUrl: image,
                              request: imageGenerationMetadata?.request ?? null
                            })
                          }}
                        >
                          <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                        </button>
                      )}
                      {onDeleteImage && (
                        <button
                          type="button"
                          className="pointer-events-auto inline-flex h-7 w-7 items-center justify-center rounded-full text-text-muted transition hover:bg-danger/10 hover:text-danger"
                          aria-label={t(
                            "playground:imageGeneration.deleteImage",
                            "Delete image"
                          ) as string}
                          title={t(
                            "playground:imageGeneration.deleteImage",
                            "Delete image"
                          ) as string}
                          onClick={() => {
                            onDeleteImage?.({
                              messageId,
                              imageIndex: index,
                              imageUrl: image
                            })
                          }}
                        >
                          <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                        </button>
                      )}
                    </div>
                  )}
                </div>
              ))}
          </div>
        )}
    </>
  )
})
