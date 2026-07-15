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
import type { MenuItemType } from "antd/es/menu/interface"
import type { WorkspaceSourceSavedViewResponse } from "@/services/tldw/domains/workspace-api"
import type { SourceListViewState } from "./source-list-view"
import {
  SOURCE_VIEW_PRESETS,
  applySavedSourceViewState,
  serializeSourceListViewState
} from "./source-saved-views"
import type { SourceViewStateValidationIssue } from "./source-saved-views"
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

const INVALID_VIEW_REASON_LABELS = {
  invalid_json: "Invalid saved state JSON",
  invalid_state: "Invalid saved state",
  unsupported_schema_version: "Unsupported schema version"
} as const

let sourceViewOverlayRequestSequence = 0

type SourceViewOverlayKind = "save" | "replace" | "reset" | "delete"

export interface SourceViewOverlayRequest {
  id: number
  kind: SourceViewOverlayKind
  generation: number
  invoker: HTMLElement
  view?: WorkspaceSourceSavedViewResponse
  validationIssues?: SourceViewStateValidationIssue[]
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
  if (
    !element.matches(
      "a[href], button, input, select, textarea, summary, [contenteditable='true'], [tabindex]"
    )
  ) {
    return false
  }
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

    const sourcesTab = Array.from(
      document.querySelectorAll<HTMLElement>("[role='tab']")
    ).find(
      (candidate) =>
        /\bsources\b/i.test(candidate.textContent ?? "") &&
        isFocusable(candidate)
    )
    if (sourcesTab) {
      sourcesTab.focus()
      return
    }

    const restoreSources = document.querySelector<HTMLElement>(
      "[data-testid='workspace-restore-sources']"
    )
    if (isFocusable(restoreSources)) {
      restoreSources.focus()
      return
    }

    const sourcesLandmark = Array.from(
      document.querySelectorAll<HTMLElement>(
        "[data-sources-focus-target], [role='complementary'][aria-label*='Sources' i]"
      )
    ).find(isFocusable)
    if (sourcesLandmark) sourcesLandmark.focus()
  }, 0)
}

const issueFieldLabel = (field: string): string =>
  field
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replaceAll("_", " ")
    .replace(/^./, (character) => character.toUpperCase())

type SpaceActivatableMenuItem = MenuItemType &
  Pick<React.LiHTMLAttributes<HTMLLIElement>, "onKeyUp">

const menuCommand = (item: MenuItemType): SpaceActivatableMenuItem => ({
  ...item,
  onKeyUp: (event) => {
    if (
      event.key !== " " ||
      event.currentTarget.getAttribute("aria-disabled") === "true"
    ) {
      return
    }
    event.preventDefault()
    event.currentTarget.click()
  }
})

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
      const validation =
        kind === "replace"
          ? serializeSourceListViewState(sourceListViewState)
          : null
      setMenuOpen(false)
      onOpenOverlay({
        id: sourceViewOverlayRequestSequence,
        kind,
        generation: controller.generation,
        invoker,
        view,
        validationIssues:
          validation?.ok === false ? validation.issues : undefined
      })
    },
    [
      controller.available,
      controller.generation,
      onOpenOverlay,
      sourceListViewState
    ]
  )

  const menuItems = React.useMemo<MenuProps["items"]>(
    () => [
      {
        key: "built-in",
        type: "group",
        label: "Built-in views",
        children: SOURCE_VIEW_PRESET_ORDER.map((key) =>
          menuCommand({
            key: `preset:${key}`,
            label: SOURCE_VIEW_PRESETS[key].label
          })
        )
      },
      {
        key: "saved",
        type: "group",
        label: "Saved views",
        children:
          controller.views.length > 0
            ? controller.views.map((view) => ({
                key: `saved-view:${view.id}`,
                label: (
                  <span className="flex min-w-[15rem] items-center gap-2">
                    {!view.valid && (
                      <AlertTriangle
                        className="h-3.5 w-3.5 shrink-0 text-warning"
                        aria-hidden="true"
                      />
                    )}
                    <span className="min-w-0 flex-1 truncate">{view.name}</span>
                    {!view.valid && (
                      <>
                        <span className="text-[11px] font-medium text-warning">
                          Invalid
                        </span>
                        <span className="sr-only">
                          {INVALID_VIEW_REASON_LABELS[view.invalid_reason]}. Apply unavailable.
                        </span>
                      </>
                    )}
                    {controller.activeViewId === view.id && controller.modified && (
                      <span className="text-[11px] font-medium text-warning">
                        Modified
                      </span>
                    )}
                  </span>
                ),
                children: [
                  ...(view.valid
                    ? [
                        menuCommand({
                          key: `saved:${view.id}:apply`,
                          label: `Apply saved view ${view.name}`,
                          disabled: controller.busy
                        }),
                        menuCommand({
                          key: `saved:${view.id}:replace`,
                          label: `Replace saved view ${view.name}`,
                          icon: <Save className="h-3.5 w-3.5" aria-hidden="true" />,
                          disabled: !controller.available || controller.busy
                        })
                      ]
                    : [
                        menuCommand({
                          key: `saved:${view.id}:reset`,
                          label: `Reset saved view ${view.name}`,
                          icon: (
                            <RotateCcw
                              className="h-3.5 w-3.5"
                              aria-hidden="true"
                            />
                          ),
                          disabled: !controller.available || controller.busy
                        })
                      ]),
                  menuCommand({
                    key: `saved:${view.id}:delete`,
                    label: `Delete saved view ${view.name}`,
                    icon: <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />,
                    danger: true,
                    disabled: !controller.available || controller.busy
                  })
                ]
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
    [controller]
  )

  const handleMenuClick: MenuProps["onClick"] = ({ key, domEvent }) => {
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
    } else {
      const view = controller.views.find((candidate) =>
        (["apply", "replace", "reset", "delete"] as const).some(
          (command) => key === `saved:${candidate.id}:${command}`
        )
      )
      if (!view) return
      if (key.endsWith(":apply") && view.valid) {
        controller.applyView(view)
      } else if (key.endsWith(":replace") && view.valid) {
        openOverlay("replace", domEvent.currentTarget as HTMLElement, view)
        return
      } else if (key.endsWith(":reset") && !view.valid) {
        openOverlay("reset", domEvent.currentTarget as HTMLElement, view)
        return
      } else if (key.endsWith(":delete")) {
        openOverlay("delete", domEvent.currentTarget as HTMLElement, view)
        return
      }
    }
    setMenuOpen(false)
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
  const pendingFocusRestoreRef = React.useRef<{
    token: number
    invoker: HTMLElement | null
  } | null>(null)
  const focusRestoreSequenceRef = React.useRef(0)
  const lastRestoredFocusTokenRef = React.useRef(0)
  const [focusRestoreToken, setFocusRestoreToken] = React.useState(0)

  const close = React.useCallback(
    (restoreFocus = true) => {
      if (!activeRequest) return
      if (restoreFocus) {
        focusRestoreSequenceRef.current += 1
        const token = focusRestoreSequenceRef.current
        pendingFocusRestoreRef.current = {
          token,
          invoker: activeRequest.invoker
        }
        setFocusRestoreToken(token)
      }
      setActiveRequest(null)
      setName("")
      setNameTouched(false)
    },
    [activeRequest]
  )

  React.useEffect(() => {
    const pending = pendingFocusRestoreRef.current
    if (
      !pending ||
      pending.token !== focusRestoreToken ||
      pending.token <= lastRestoredFocusTokenRef.current
    ) {
      return
    }
    lastRestoredFocusTokenRef.current = pending.token
    restoreOverlayFocus(pending.invoker)
  }, [focusRestoreToken])

  React.useLayoutEffect(() => {
    if (!request || handledRequestIdRef.current === request.id) return
    handledRequestIdRef.current = request.id
    if (
      request.generation !== controller.generation ||
      !controller.available
    ) {
      restoreOverlayFocus(request.invoker)
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
  const replacementIssues =
    activeRequest?.kind === "replace"
      ? [
          ...(activeRequest.validationIssues ?? []),
          ...controller.serializationIssues
        ].filter(
          (issue, index, issues) =>
            issues.findIndex(
              (candidate) =>
                candidate.field === issue.field && candidate.message === issue.message
            ) === index
        )
      : []
  const replaceInvalid = replacementIssues.length > 0

  const cancel = () => {
    if (controller.busy) return
    if (duplicate) controller.dismissDuplicateConflict()
    else if (controller.mutationError || controller.versionConflict) {
      controller.dismissMutationFailure()
    }
    close()
  }

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
    if (activeRequest.kind === "replace") {
      if (replaceInvalid) return
      await controller.replaceView(view)
    }
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
      <span
        data-testid="source-view-overlay-host"
        className="sr-only"
        aria-hidden="true"
      />
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
              <Button
                size="small"
                disabled={controller.busy}
                onClick={() => void controller.retry()}
              >
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
        onCancel={cancel}
        closable={!controller.busy}
        maskClosable={!controller.busy}
        keyboard={!controller.busy}
        destroyOnHidden
        focusable={{ focusTriggerAfterClose: false }}
        afterOpenChange={(open) => {
          if (open && activeRequest?.kind === "save" && !duplicate) {
            window.setTimeout(() => inputRef.current?.focus(), 0)
          }
        }}
        footer={
          activeRequest
            ? [
                <Button key="cancel" onClick={cancel} disabled={controller.busy}>
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
                    (activeRequest.kind === "save" && !duplicate && saveInvalid) ||
                    (activeRequest.kind === "replace" && replaceInvalid)
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
          <div className="space-y-3">
            <p className="text-sm text-text-muted">
              {duplicate
                ? `A saved view named ${duplicate.name} already exists. Replace it with the current filters and sort?`
                : activeRequest?.kind === "replace"
                  ? `Replace ${view?.name ?? "this saved view"} with the current filters and sort?`
                  : activeRequest?.kind === "reset"
                    ? `Reset ${view?.name ?? "this saved view"} to the default source view?`
                    : `Delete ${view?.name ?? "this saved view"}?`}
            </p>
            {replacementIssues.length > 0 && (
              <ul role="alert" className="space-y-1 text-xs text-error">
                {replacementIssues.map((issue) => (
                  <li key={`${issue.field}:${issue.message}`}>
                    {issueFieldLabel(issue.field)}: {issue.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </Modal>
    </>
  )
}
