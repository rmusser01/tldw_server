import React from "react"
import { Button, InputNumber, Skeleton, Typography } from "antd"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"
import type {
  LlamacppSnapshotSlotsResponse,
  LlamacppSnapshotCatalogResponse,
  LlamacppSnapshotOperationResponse
} from "@/types/llamacpp-admin"

export interface LlamacppSnapshotsPanelProps {
  enabled: boolean
  retention: number
  slots: LlamacppSnapshotSlotsResponse | null
  catalog: LlamacppSnapshotCatalogResponse | null
  operation?: LlamacppSnapshotOperationResponse | null
  loading?: boolean
  mutating?: boolean
  error?: string | null
  outcomeUnknown?: boolean
  onEnable: (enabled: boolean) => void
  onRetention: (retention: number) => void
  onRefresh: () => void
  onPage: (offset: number) => void
  onSave: (slotId: number) => void
  onRestore: (snapshotId: string, slotId: number) => void
  onDelete: (snapshotId: string) => void
  onStop: () => void
}

export const snapshotOperationActive = (
  operation?: LlamacppSnapshotOperationResponse | null,
  launchGeneration?: string | null
) =>
  Boolean(
    operation &&
    launchGeneration &&
    operation.launch_generation === launchGeneration &&
    !["complete", "failed", "outcome_unknown"].includes(operation.state)
  )

export const LlamacppSnapshotsPanel = (props: LlamacppSnapshotsPanelProps) => {
  const { t } = useTranslation()
  const copy = (key: string, fallback: string) =>
    t(`settings:admin.snapshots.${key}`, fallback)
  const [confirmation, setConfirmation] = React.useState<{
    kind: "restore" | "delete" | "stop"
    id: string
  } | null>(null)
  const [destination, setDestination] = React.useState<number | null>(null)
  const [retention, setRetention] = React.useState(props.retention)
  const trigger = React.useRef<HTMLElement | null>(null)
  const destinationRef = React.useRef<HTMLSelectElement>(null)
  const cancelRef = React.useRef<React.ComponentRef<typeof Button>>(null)
  React.useEffect(() => setRetention(props.retention), [props.retention])
  React.useEffect(() => {
    if (confirmation?.kind === "restore") destinationRef.current?.focus()
    else if (confirmation) cancelRef.current?.focus()
  }, [confirmation])
  const close = () => {
    setConfirmation(null)
    trigger.current?.focus()
  }
  const open = (
    kind: "restore" | "delete" | "stop",
    id: string,
    element: HTMLElement
  ) => {
    trigger.current = element
    setDestination(
      props.slots?.slots.find((slot) => !slot.busy)?.slot_id ?? null
    )
    setConfirmation({ kind, id })
  }
  const currentOperation = Boolean(
    props.operation &&
    props.slots?.launch_generation &&
    props.operation.launch_generation === props.slots.launch_generation
  )
  const active = snapshotOperationActive(
    props.operation,
    props.slots?.launch_generation
  )
  const unknown =
    props.outcomeUnknown ||
    (currentOperation &&
      props.operation?.state === "outcome_unknown" &&
      props.slots?.capability !== "stopped")
  const blocked = Boolean(
    props.loading || props.mutating || props.error || active || unknown
  )
  const ready = props.enabled && props.slots?.capability === "ready" && !blocked
  const idleSlots = props.slots?.slots.filter((slot) => !slot.busy) || []
  const selected = props.catalog?.snapshots.find(
    (snapshot) => snapshot.snapshot_id === confirmation?.id
  )
  const canRestore =
    ready &&
    selected?.compatibility === "compatible" &&
    idleSlots.some((slot) => slot.slot_id === destination)
  const stateLabels = {
    ready: copy("ready", "Ready"),
    stopped: copy(
      "stopped",
      "Start this profile to inspect slots and save or restore."
    ),
    disabled: copy(
      "disabled",
      "Enable snapshots, then start or restart this profile explicitly."
    ),
    restart_required: copy(
      "restartRequired",
      "Restart required. Stop and start this profile when callers are idle."
    ),
    busy: copy("busy", "An operation is in progress. Wait for completion."),
    unsupported: copy(
      "unsupported",
      "This runtime configuration or build is not supported for snapshots."
    ),
    unavailable: copy(
      "unavailable",
      "The runtime owner is unavailable. Refresh to check its state."
    )
  }
  const operationLabels = {
    validating: copy("validating", "Validating"),
    saving: copy("saving", "Saving"),
    verifying: copy("verifying", "Verifying"),
    restoring: copy("restoring", "Restoring"),
    complete: copy("complete", "Complete"),
    failed: copy("failed", "Failed"),
    outcome_unknown: copy("outcomeUnknown", "Outcome unknown")
  }
  const time = (date: string) =>
    new Date(date).toLocaleString(undefined, { timeZoneName: "short" })
  const bytes = (count: number) =>
    `${(count / 1024 / 1024).toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB`
  return (
    <section
      aria-label={copy("title", "Slot snapshots")}
      className="space-y-5 rounded-lg border border-border bg-surface p-4 text-text"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold">
            {copy("title", "Slot snapshots")}
          </h3>
          <p className="max-w-prose">
            {copy(
              "intro",
              "Save processed context to reuse later. Conversations are unchanged."
            )}
          </p>
          <p className="text-text-muted">
            {copy(
              "sensitivity",
              "Sensitive runtime context. Administrators only."
            )}
          </p>
        </div>
        <Button
          disabled={blocked}
          onClick={() => props.onEnable(!props.enabled)}
        >
          {props.enabled
            ? copy("disable", "Disable snapshots")
            : copy("enable", "Enable snapshots")}
        </Button>
      </div>
      <p className="max-w-prose text-text-muted">
        {copy(
          "enableHelp",
          "Changing enablement does not restart the runtime. Existing saved copies remain available. Quiesce all callers before saving or restoring; an idle slot is only an observation."
        )}
      </p>
      {props.error && (
        <Alert variant="error" title={props.error} role="alert" />
      )}
      {props.slots && (
        <p>
          {stateLabels[props.slots.capability]}
          {props.slots.reason && <> ({props.slots.reason})</>}
        </p>
      )}
      <div>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h4 className="font-semibold">{copy("slots", "Slots")}</h4>
          <Button onClick={props.onRefresh} disabled={props.loading}>
            {copy("refresh", "Refresh")}
          </Button>
        </div>
        {props.loading && !props.slots ? (
          <Skeleton active={false} paragraph={{ rows: 2 }} />
        ) : (
          props.slots?.slots.map((slot) => (
            <div
              key={slot.slot_id}
              className="flex flex-wrap items-center justify-between gap-3 border-t border-border py-3"
            >
              <p>
                {copy("slot", "Slot")} {slot.slot_id}:{" "}
                {slot.busy ? copy("slotBusy", "Busy") : copy("idle", "Idle")} ·{" "}
                {copy("processedTokens", "Processed tokens")}:{" "}
                {slot.token_count.toLocaleString()}
              </p>
              <div>
                <Button
                  disabled={!ready || slot.busy || slot.token_count === 0}
                  onClick={() => props.onSave(slot.slot_id)}
                >
                  {copy("save", "Save snapshot")}
                </Button>
                {slot.busy && (
                  <p className="text-text-muted">
                    {copy("saveBusy", "Save unavailable: busy")}
                  </p>
                )}
                {!slot.busy && slot.token_count === 0 && (
                  <p className="text-text-muted">
                    {copy("saveEmpty", "Save unavailable: no processed tokens")}
                  </p>
                )}
              </div>
            </div>
          ))
        )}
      </div>
      <div>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
          <h4 className="font-semibold">
            {copy("catalog", "Saved snapshots")}
          </h4>
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="snapshot-retention">
              {copy("keepNewest", "Keep newest")}
            </label>
            <InputNumber
              id="snapshot-retention"
              min={1}
              max={1000}
              precision={0}
              value={retention}
              disabled={blocked}
              onChange={(value) => {
                if (value !== null) setRetention(value)
              }}
            />
            <Button
              disabled={
                blocked ||
                retention === props.retention ||
                retention < 1 ||
                retention > 1000 ||
                !Number.isInteger(retention)
              }
              onClick={() => props.onRetention(retention)}
            >
              {copy("applyRetention", "Apply retention")}
            </Button>
          </div>
        </div>
        <p className="text-text-muted">
          {copy(
            "retentionHelp",
            "Pruning happens only after a successful save. Changing this limit alone deletes nothing."
          )}
        </p>
        {props.catalog && (
          <p>
            {copy("total", "Total")}: {bytes(props.catalog.total_bytes)} ·{" "}
            {props.catalog.total} {copy("savedCopies", "saved copies")}
          </p>
        )}
        {props.catalog?.total === 0 && (
          <p className="py-4">
            {copy(
              "empty",
              "No saved snapshots. Send a text prompt to this runtime, wait for an idle slot, then save its processed context."
            )}
          </p>
        )}
        {[...(props.catalog?.snapshots || [])]
          .sort((a, b) => b.commit_sequence - a.commit_sequence)
          .map((snapshot) => (
            <div
              key={snapshot.snapshot_id}
              className="flex flex-wrap items-start justify-between gap-3 border-t border-border py-3"
            >
              <div className="min-w-0 space-y-1">
                <p>
                  <time dateTime={snapshot.created_at}>
                    {time(snapshot.created_at)}
                  </time>
                </p>
                <p>
                  {copy("processedTokens", "Processed tokens")}:{" "}
                  {snapshot.token_count.toLocaleString()} ·{" "}
                  {bytes(snapshot.byte_count)}
                </p>
                <p>
                  {snapshot.compatibility === "compatible"
                    ? copy("compatible", "Compatible")
                    : snapshot.compatibility === "incompatible"
                      ? copy("incompatible", "Incompatible")
                      : copy("compatibilityUnknown", "Compatibility unknown")}
                  {snapshot.reasons.length > 0 &&
                    `: ${snapshot.reasons.join(", ")}`}
                </p>
                <Typography.Text
                  copyable
                  className="break-all font-mono text-xs"
                >
                  {snapshot.snapshot_id}
                </Typography.Text>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  disabled={
                    !ready ||
                    !idleSlots.length ||
                    snapshot.compatibility !== "compatible"
                  }
                  onClick={(event) =>
                    open("restore", snapshot.snapshot_id, event.currentTarget)
                  }
                >
                  {copy("restore", "Restore")}
                </Button>
                <Button
                  danger
                  disabled={blocked}
                  onClick={(event) =>
                    open("delete", snapshot.snapshot_id, event.currentTarget)
                  }
                >
                  {copy("delete", "Delete")}
                </Button>
              </div>
            </div>
          ))}
        {ready && !idleSlots.length && (
          <p>
            {copy("noIdle", "Restore unavailable: no idle destination slot.")}
          </p>
        )}
        {props.catalog && props.catalog.total > props.catalog.limit && (
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={props.loading || props.catalog.offset === 0}
              onClick={() =>
                props.onPage(
                  Math.max(0, props.catalog!.offset - props.catalog!.limit)
                )
              }
            >
              {copy("previous", "Previous snapshots")}
            </Button>
            <Button
              disabled={
                props.loading ||
                props.catalog.offset + props.catalog.limit >=
                  props.catalog.total
              }
              onClick={() =>
                props.onPage(props.catalog!.offset + props.catalog!.limit)
              }
            >
              {copy("next", "Next snapshots")}
            </Button>
          </div>
        )}
      </div>
      {confirmation && (
        <div
          className="space-y-3 rounded-md border border-border p-3"
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              event.stopPropagation()
              close()
            }
          }}
        >
          {confirmation.kind === "restore" && (
            <>
              <h4 className="break-all font-semibold">
                {copy("restore", "Restore")} {confirmation.id}
              </h4>
              <label className="block" htmlFor="snapshot-destination">
                {copy("destination", "Destination slot")}
              </label>
              <select
                ref={destinationRef}
                id="snapshot-destination"
                value={destination ?? ""}
                onChange={(event) => setDestination(Number(event.target.value))}
                className="min-w-[8rem] rounded-md border border-border bg-surface2 p-2 pr-8 text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
              >
                {idleSlots.map((slot) => (
                  <option key={slot.slot_id} value={slot.slot_id}>
                    {slot.slot_id}: {copy("idle", "Idle")}
                  </option>
                ))}
              </select>
              <p>
                {copy(
                  "restoreWarning",
                  "This replaces the destination cache. Failure may also clear it. Messages and tool state will not be restored."
                )}
              </p>
              <Button
                danger
                disabled={!canRestore}
                onClick={() => {
                  if (canRestore && destination !== null) {
                    props.onRestore(confirmation.id, destination)
                    close()
                  }
                }}
              >
                {t("settings:admin.snapshots.restoreTarget", {
                  defaultValue: "Restore into slot {{slot}}",
                  slot: destination
                })}
              </Button>
            </>
          )}
          {confirmation.kind === "delete" && (
            <>
              <p className="break-all">
                {copy(
                  "deleteWarning",
                  "Permanently delete this saved copy. This does not erase an active slot."
                )}{" "}
                {confirmation.id}
              </p>
              <Button
                danger
                disabled={blocked}
                onClick={() => {
                  props.onDelete(confirmation.id)
                  close()
                }}
              >
                {t("settings:admin.snapshots.deleteTarget", {
                  defaultValue: "Permanently delete {{id}}",
                  id: confirmation.id
                })}
              </Button>
            </>
          )}
          {confirmation.kind === "stop" && (
            <>
              <p>
                {copy(
                  "stopWarning",
                  "Stopping ends inference for every caller. Confirm Stop to recover from an unknown snapshot outcome; then start the runtime explicitly."
                )}
              </p>
              <Button
                danger
                disabled={props.mutating || active}
                onClick={() => {
                  props.onStop()
                  close()
                }}
              >
                {copy("stopConfirm", "Stop runtime and inference")}
              </Button>
            </>
          )}
          <Button ref={cancelRef} className="ml-2" onClick={close}>
            {copy("cancel", "Cancel")}
          </Button>
        </div>
      )}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="space-y-2"
      >
        {props.outcomeUnknown && (
          <p>
            {copy(
              "transportUnknown",
              "Outcome unknown. Refresh to recover the receipt, or use Stop recovery. Do not repeat the mutation."
            )}
          </p>
        )}
        {props.operation && (
          <>
            {!currentOperation && (
              <p>
                {copy(
                  "previousLaunch",
                  "Receipt from a previous launch. It does not describe the current slot state."
                )}
              </p>
            )}
            <p>
              {copy("latest", "Latest operation")}:{" "}
              {operationLabels[props.operation.state]}
              {props.operation.token_count != null &&
                ` · ${props.operation.token_count.toLocaleString()} ${copy("tokens", "tokens")}`}
            </p>
            {active && (
              <p>
                {copy(
                  "working",
                  "Do not stop the runtime until this operation completes. Closing this page does not cancel execution."
                )}
              </p>
            )}
            {props.operation.state === "complete" &&
              props.operation.kind === "restore" && (
                <p>
                  {copy(
                    "continue",
                    "Open the original conversation in Chatbook to continue."
                  )}
                </p>
              )}
            {props.operation.state === "failed" &&
              props.operation.recovery_action === "retry_manually" && (
                <p>
                  {copy(
                    "failedHelp",
                    "The operation failed before dispatch. Review Details, resolve the cause and refresh before trying a new manual action."
                  )}
                </p>
              )}
            {unknown && (
              <p>
                {copy(
                  "unknownHelp",
                  "The runtime may have changed. Do not repeat the restore. Stop the runtime explicitly before any new snapshot mutation."
                )}
              </p>
            )}
            <details>
              <summary>{copy("details", "Details")}</summary>
              <Typography.Text copyable className="break-all">
                {props.operation.operation_id}
              </Typography.Text>
              {props.operation.error_code && (
                <p>{props.operation.error_code}</p>
              )}
              {props.operation.warning_code && (
                <p>{props.operation.warning_code}</p>
              )}
            </details>
          </>
        )}
      </div>
      {unknown && (
        <Button
          danger
          disabled={props.mutating}
          onClick={(event) => open("stop", "", event.currentTarget)}
        >
          {copy("stopRecovery", "Stop recovery")}
        </Button>
      )}
    </section>
  )
}
