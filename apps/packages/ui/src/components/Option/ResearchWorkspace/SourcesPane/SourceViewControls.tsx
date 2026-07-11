import React from "react"
import {
  AlertTriangle,
  Bookmark,
  ChevronDown,
  RefreshCw,
  RotateCcw,
  Save,
  Trash2
} from "lucide-react"
import { Button, Dropdown, Input, Modal, Tooltip } from "antd"
import type { MenuProps } from "antd"
import type { WorkspaceSourceSavedViewResponse } from "@/services/tldw/domains/workspace-api"
import type { SourceListViewState } from "./source-list-view"
import {
  SOURCE_VIEW_PRESETS,
  applySavedSourceViewState,
  serializeSourceListViewState
} from "./source-saved-views"
import type { SourceSavedViewsController } from "./use-source-saved-views"

const SOURCE_VIEW_PRESET_ORDER = [
  "needsReview",
  "unreviewed",
  "failedIngest",
  "partiallyIndexed",
  "pdfs",
  "webCaptures",
  "largeFiles"
] as const

let sourceViewOverlayRequestSequence = 0

type SourceViewOverlayKind = "save" | "replace" | "reset" | "delete"

export interface SourceViewOverlayRequest {
  id: number
  kind: SourceViewOverlayKind
  generation: number
  invoker: HTMLElement
  view?: WorkspaceSourceSavedViewResponse
}

interface SourceViewControlsProps {
  controller: SourceSavedViewsController
  sourceListViewState: SourceListViewState
  onApplySourceListViewState: (state: SourceListViewState) => void
  onOpenOverlay: (request: SourceViewOverlayRequest) => void
}

interface SourceViewOverlayHostProps {
  controller: SourceSavedViewsController
  request: SourceViewOverlayRequest | null
  onRequestHandled: () => void
}

const isFocusable = (element: HTMLElement | null): element is HTMLElement => {
  if (!element?.isConnected) return false
  if (element.matches(":disabled, [aria-disabled='true']")) return false
  if (element.closest("[hidden], [aria-hidden='true']")) return false
  let current: HTMLElement | null = element
  while (current) {
    const style = window.getComputedStyle(current)
    if (style.display === "none" || style.visibility === "hidden") return false
    current = current.parentElement
  }
  return true
}

const restoreOverlayFocus = (invoker: HTMLElement | null) => {
  window.setTimeout(() => {
    if (isFocusable(invoker)) {
      invoker.focus()
      return
    }

    const visibleTrigger = Array.from(
      document.querySelectorAll<HTMLElement>("[data-source-view-trigger]")
    ).find(isFocusable)
    if (visibleTrigger) {
      visibleTrigger.focus()
      return
    }

    const sourcesLandmark = document.querySelector<HTMLElement>(
      "[role='complementary'][aria-label*='Sources' i], [data-testid='workspace-sources-pane-root']"
    )
    if (sourcesLandmark?.isConnected) sourcesLandmark.focus()
  }, 0)
}

const issueFieldLabel = (field: string): string =>
  field
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replaceAll("_", " ")
    .replace(/^./, (character) => character.toUpperCase())

const SavedViewActions: React.FC<{
  view: WorkspaceSourceSavedViewResponse
  controller: SourceSavedViewsController
  openOverlay: (
    kind: Exclude<SourceViewOverlayKind, "save">,
    invoker: HTMLElement,
    view: WorkspaceSourceSavedViewResponse
  ) => void
}> = ({ view, controller, openOverlay }) => (
  <span className="ml-auto inline-flex shrink-0 items-center gap-0.5">
    {view.valid ? (
      <Tooltip title={`Replace saved view ${view.name}`}>
        <button
          type="button"
          aria-label={`Replace saved view ${view.name}`}
          disabled={!controller.available || controller.busy}
          onClick={(event) => {
            event.preventDefault()
            event.stopPropagation()
            openOverlay("replace", event.currentTarget, view)
          }}
          className="inline-flex h-7 w-7 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Save className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
      </Tooltip>
    ) : (
      <Tooltip title={`Reset saved view ${view.name}`}>
        <button
          type="button"
          aria-label={`Reset saved view ${view.name}`}
          disabled={!controller.available || controller.busy}
          onClick={(event) => {
            event.preventDefault()
            event.stopPropagation()
            openOverlay("reset", event.currentTarget, view)
          }}
          className="inline-flex h-7 w-7 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
      </Tooltip>
    )}
    <Tooltip title={`Delete saved view ${view.name}`}>
      <button
        type="button"
        aria-label={`Delete saved view ${view.name}`}
        disabled={!controller.available || controller.busy}
        onClick={(event) => {
          event.preventDefault()
          event.stopPropagation()
          openOverlay("delete", event.currentTarget, view)
        }}
        className="inline-flex h-7 w-7 items-center justify-center rounded text-text-muted hover:bg-error/10 hover:text-error focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-error disabled:cursor-not-allowed disabled:opacity-50"
      >
        <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
      </button>
    </Tooltip>
  </span>
)

export const SourceViewControls: React.FC<SourceViewControlsProps> = ({
  controller,
  sourceListViewState,
  onApplySourceListViewState,
  onOpenOverlay
}) => {
  const [menuOpen, setMenuOpen] = React.useState(false)

  const openOverlay = React.useCallback(
    (
      kind: SourceViewOverlayKind,
      invoker: HTMLElement,
      view?: WorkspaceSourceSavedViewResponse
    ) => {
      if (!controller.available) return
      sourceViewOverlayRequestSequence += 1
      setMenuOpen(false)
      onOpenOverlay({
        id: sourceViewOverlayRequestSequence,
        kind,
        generation: controller.generation,
        invoker,
        view
      })
    },
    [controller.available, controller.generation, onOpenOverlay]
  )

  const menuItems = React.useMemo<MenuProps["items"]>(
    () => [
      {
        key: "built-in",
        type: "group",
        label: "Built-in views",
        children: SOURCE_VIEW_PRESET_ORDER.map((key) => ({
          key: `preset:${key}`,
          label: SOURCE_VIEW_PRESETS[key].label
        }))
      },
      {
        key: "saved",
        type: "group",
        label: "Saved views",
        children:
          controller.views.length > 0
            ? controller.views.map((view) => ({
                key: `saved:${view.id}`,
                disabled: !view.valid,
                label: (
                  <span className="flex min-w-[15rem] items-center gap-2">
                    {!view.valid && (
                      <AlertTriangle
                        className="h-3.5 w-3.5 shrink-0 text-warning"
                        aria-hidden="true"
                      />
                    )}
                    <span className="min-w-0 flex-1 truncate">{view.name}</span>
                    {controller.activeViewId === view.id && controller.modified && (
                      <span className="text-[11px] font-medium text-warning">
                        Modified
                      </span>
                    )}
                    <SavedViewActions
                      view={view}
                      controller={controller}
                      openOverlay={openOverlay}
                    />
                  </span>
                )
              }))
            : [
                {
                  key: "saved-empty",
                  disabled: true,
                  label: controller.loading ? "Loading saved views..." : "No saved views"
                }
              ]
      }
    ],
    [controller, openOverlay]
  )

  const handleMenuClick: MenuProps["onClick"] = ({ key }) => {
    if (key.startsWith("preset:")) {
      const presetKey = key.slice("preset:".length) as keyof typeof SOURCE_VIEW_PRESETS
      const preset = SOURCE_VIEW_PRESETS[presetKey]
      if (preset) {
        const serialized = serializeSourceListViewState(preset.state)
        if (serialized.ok) {
          onApplySourceListViewState(
            applySavedSourceViewState(sourceListViewState, serialized.state)
          )
        }
      }
    } else if (key.startsWith("saved:")) {
      const view = controller.views.find(
        (candidate) => candidate.id === key.slice("saved:".length)
      )
      if (view?.valid) controller.applyView(view)
    }
    setMenuOpen(false)
  }

  const handleMenuKeyDown: React.KeyboardEventHandler<HTMLElement> = (event) => {
    const items = Array.from(
      event.currentTarget.querySelectorAll<HTMLElement>(
        "[role='menuitem']:not([aria-disabled='true'])"
      )
    )
    const activeItem = (document.activeElement as HTMLElement | null)?.closest(
      "[role='menuitem']"
    ) as HTMLElement | null
    const activeIndex = activeItem ? items.indexOf(activeItem) : -1

    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault()
      const delta = event.key === "ArrowDown" ? 1 : -1
      const nextIndex =
        activeIndex < 0
          ? event.key === "ArrowDown"
            ? 0
            : items.length - 1
          : (activeIndex + delta + items.length) % items.length
      items[nextIndex]?.focus()
      return
    }

    if ((event.key === "Enter" || event.key === " ") && activeIndex >= 0) {
      event.preventDefault()
      activeItem?.click()
    }
  }

  const unavailableDescriptionId = React.useId()

  return (
    <div className="mt-2 flex min-w-0 items-center gap-1.5" aria-label="Source view controls">
      <Dropdown
        open={menuOpen}
        onOpenChange={setMenuOpen}
        trigger={["click"]}
        menu={{
          items: menuItems,
          onClick: handleMenuClick,
          onKeyDown: handleMenuKeyDown,
          selectable: false
        }}
      >
        <Button
          size="small"
          data-source-view-trigger
          aria-label="Source views"
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          icon={<Bookmark className="h-3.5 w-3.5" aria-hidden="true" />}
          onKeyDown={(event) => {
            if (["Enter", " ", "ArrowDown"].includes(event.key)) {
              event.preventDefault()
              setMenuOpen(true)
            }
          }}
        >
          Views
          <ChevronDown className="ml-1 h-3 w-3" aria-hidden="true" />
        </Button>
      </Dropdown>

      <Tooltip title={controller.available ? "Save source view" : "Select a workspace"}>
        <span>
          <Button
            size="small"
            aria-label="Save source view"
            aria-describedby={
              controller.available ? undefined : unavailableDescriptionId
            }
            aria-busy={controller.busy}
            disabled={!controller.available || controller.busy}
            icon={<Save className="h-3.5 w-3.5" aria-hidden="true" />}
            onClick={(event) => openOverlay("save", event.currentTarget)}
          >
            Save
          </Button>
        </span>
      </Tooltip>
      {!controller.available && (
        <span id={unavailableDescriptionId} className="sr-only">
          Select a workspace
        </span>
      )}

      {controller.activeViewId && controller.modified && (
        <span className="text-[11px] font-medium text-warning">Modified</span>
      )}
    </div>
  )
}

export const SourceViewOverlayHost: React.FC<SourceViewOverlayHostProps> = ({
  controller,
  request,
  onRequestHandled
}) => {
  const [activeRequest, setActiveRequest] =
    React.useState<SourceViewOverlayRequest | null>(null)
  const [name, setName] = React.useState("")
  const [nameTouched, setNameTouched] = React.useState(false)
  const inputRef = React.useRef<React.ComponentRef<typeof Input>>(null)
  const handledRequestIdRef = React.useRef<number | null>(null)
  const announcementAtOpenRef = React.useRef<string | null>(null)
  const mutationCycleObservedRef = React.useRef(false)

  const close = React.useCallback((restoreFocus = true) => {
    setActiveRequest((current) => {
      if (restoreFocus) restoreOverlayFocus(current?.invoker ?? null)
      return null
    })
    setName("")
    setNameTouched(false)
  }, [])

  React.useLayoutEffect(() => {
    if (!request || handledRequestIdRef.current === request.id) return
    handledRequestIdRef.current = request.id
    if (
      request.generation !== controller.generation ||
      !controller.available
    ) {
      onRequestHandled()
      return
    }
    setActiveRequest(request)
    announcementAtOpenRef.current = controller.announcement
    mutationCycleObservedRef.current = false
    setName("")
    setNameTouched(false)
    onRequestHandled()
  }, [
    controller.announcement,
    controller.available,
    controller.generation,
    onRequestHandled,
    request
  ])

  React.useLayoutEffect(() => {
    if (
      activeRequest &&
      (activeRequest.generation !== controller.generation || !controller.available)
    ) {
      close()
    }
  }, [activeRequest, close, controller.available, controller.generation])

  React.useEffect(() => {
    if (!activeRequest) return
    if (controller.busy || controller.announcement === null) {
      mutationCycleObservedRef.current = true
      return
    }
    const completed =
      (activeRequest.kind === "save" &&
        ["Saved view created.", "Saved view replaced."].includes(
          controller.announcement
        )) ||
      (activeRequest.kind === "replace" &&
        controller.announcement === "Saved view replaced.") ||
      (activeRequest.kind === "reset" &&
        controller.announcement === "Saved view reset.") ||
      (activeRequest.kind === "delete" &&
        controller.announcement === "Saved view deleted.")
    if (
      completed &&
      (controller.announcement !== announcementAtOpenRef.current ||
        mutationCycleObservedRef.current)
    ) {
      close()
    }
  }, [activeRequest, close, controller.announcement, controller.busy])

  const isRequestCurrent = React.useCallback(
    () =>
      activeRequest !== null &&
      controller.available &&
      activeRequest.generation === controller.generation,
    [activeRequest, controller.available, controller.generation]
  )

  const trimmedName = name.trim()
  const nameInvalid = trimmedName.length === 0 || trimmedName.length > 120
  const saveInvalid = nameInvalid || controller.serializationIssues.length > 0
  const duplicate =
    activeRequest?.kind === "save" ? controller.duplicateConflict : null
  const view = activeRequest?.view

  const submit = async () => {
    if (!isRequestCurrent() || !activeRequest) return
    if (duplicate) {
      await controller.confirmReplace()
      return
    }
    if (activeRequest.kind === "save") {
      setNameTouched(true)
      if (saveInvalid) return
      await controller.createView(trimmedName)
      return
    }
    if (!view) return
    if (activeRequest.kind === "replace") await controller.replaceView(view)
    if (activeRequest.kind === "reset") await controller.resetView(view)
    if (activeRequest.kind === "delete") await controller.deleteView(view)
  }

  const title = duplicate
    ? "Replace saved view?"
    : activeRequest?.kind === "save"
      ? "Save source view"
      : activeRequest?.kind === "replace"
        ? "Replace saved view?"
        : activeRequest?.kind === "reset"
          ? "Reset saved view?"
          : "Delete saved view?"

  const primaryLabel = duplicate
    ? "Replace"
    : activeRequest?.kind === "save"
      ? "Save"
      : activeRequest?.kind === "replace"
        ? "Replace"
        : activeRequest?.kind === "reset"
          ? "Reset"
          : "Delete"

  return (
    <>
      <div className="sr-only" aria-live="polite" role="status">
        {controller.announcement ?? ""}
      </div>

      {(controller.listError ||
        controller.mutationError ||
        controller.limitState ||
        controller.versionConflict) && (
        <div className="fixed bottom-12 left-1/2 z-[1100] w-[min(24rem,calc(100vw-2rem))] -translate-x-1/2 space-y-2">
          {controller.listError && (
            <div
              role="alert"
              className="flex items-center justify-between gap-3 rounded-md border border-error/30 bg-surface px-3 py-2 text-xs text-text shadow-card"
            >
              <span>{controller.listError.message}</span>
              <Button size="small" onClick={() => void controller.retry()}>
                <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
                Retry saved views
              </Button>
            </div>
          )}
          {controller.mutationError && (
            <div
              role="alert"
              className="flex items-center justify-between gap-3 rounded-md border border-error/30 bg-surface px-3 py-2 text-xs text-text shadow-card"
            >
              <span>{controller.mutationError.message}</span>
              {controller.canRetryMutation && (
                <Button size="small" onClick={() => void controller.retryMutation()}>
                  Retry saved view action
                </Button>
              )}
            </div>
          )}
          {controller.limitState && (
            <div
              role="alert"
              className="rounded-md border border-warning/30 bg-surface px-3 py-2 text-xs text-text shadow-card"
            >
              Saved view limit of {controller.limitState.limit} reached. {" "}
              {controller.limitState.guidance}
            </div>
          )}
          {controller.versionConflict && (
            <div
              role="alert"
              className="flex items-center justify-between gap-3 rounded-md border border-warning/30 bg-surface px-3 py-2 text-xs text-text shadow-card"
            >
              <span>This saved view changed on the server.</span>
              {controller.canRetryVersion && (
                <Button
                  size="small"
                  onClick={() => void controller.retryVersionConflict()}
                >
                  Retry with latest version
                </Button>
              )}
            </div>
          )}
        </div>
      )}

      <Modal
        title={title}
        open={activeRequest !== null}
        onCancel={() => close()}
        destroyOnHidden
        focusable={{ focusTriggerAfterClose: false }}
        modalRender={(node) =>
          activeRequest?.kind !== "save" || duplicate
            ? React.cloneElement(node as React.ReactElement<Record<string, unknown>>, {
                role: "alertdialog",
                "aria-label": title
              })
            : node
        }
        afterOpenChange={(open) => {
          if (open && activeRequest?.kind === "save" && !duplicate) {
            window.setTimeout(() => inputRef.current?.focus(), 0)
          }
        }}
        footer={
          activeRequest
            ? [
                <Button key="cancel" onClick={() => close()} disabled={controller.busy}>
                  Cancel
                </Button>,
                <Button
                  key="submit"
                  type="primary"
                  danger={activeRequest.kind === "delete"}
                  loading={controller.busy}
                  aria-busy={controller.busy}
                  disabled={
                    controller.busy ||
                    !isRequestCurrent() ||
                    (activeRequest.kind === "save" && !duplicate && saveInvalid)
                  }
                  onClick={() => void submit()}
                >
                  {primaryLabel}
                </Button>
              ]
            : null
        }
      >
        {activeRequest?.kind === "save" && !duplicate ? (
          <div className="space-y-3">
            <label className="block text-sm font-medium text-text" htmlFor="source-view-name">
              View name
            </label>
            <Input
              ref={inputRef}
              id="source-view-name"
              aria-label="View name"
              aria-invalid={nameTouched && nameInvalid}
              maxLength={121}
              value={name}
              onBlur={() => setNameTouched(true)}
              onChange={(event) => {
                setName(event.target.value)
                setNameTouched(true)
              }}
              onPressEnter={() => void submit()}
            />
            {nameTouched && nameInvalid && (
              <p role="alert" className="text-xs text-error">
                Name must contain between 1 and 120 characters.
              </p>
            )}
            {controller.serializationIssues.length > 0 && (
              <ul role="alert" className="space-y-1 text-xs text-error">
                {controller.serializationIssues.map((issue) => (
                  <li key={`${issue.field}:${issue.message}`}>
                    {issueFieldLabel(issue.field)}: {issue.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ) : (
          <p className="text-sm text-text-muted">
            {duplicate
              ? `A saved view named ${duplicate.name} already exists. Replace it with the current filters and sort?`
              : activeRequest?.kind === "replace"
                ? `Replace ${view?.name ?? "this saved view"} with the current filters and sort?`
                : activeRequest?.kind === "reset"
                  ? `Reset ${view?.name ?? "this saved view"} to the default source view?`
                  : `Delete ${view?.name ?? "this saved view"}?`}
          </p>
        )}
      </Modal>
    </>
  )
}
