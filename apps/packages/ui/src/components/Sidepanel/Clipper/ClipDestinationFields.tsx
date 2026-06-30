import type { ChangeEvent } from "react"
import type { WebClipperDestination } from "@/services/web-clipper/types"
import { useTranslation } from "react-i18next"

export type WorkspacePickerOption = {
  id: string
  name: string | null
}

export type FolderPickerOption = {
  id: number
  name: string | null
  path: string
}

type ClipDestinationFieldsProps = {
  destinationMode: WebClipperDestination
  folderId: string
  folderOptions: FolderPickerOption[]
  folderOptionsError: string | null
  folderValidation: string | null
  isFolderOptionsLoading: boolean
  isWorkspaceOptionsLoading: boolean
  workspaceOptions: WorkspacePickerOption[]
  workspaceOptionsError: string | null
  workspaceId: string
  workspaceValidation: string | null
  onDestinationChange: (nextValue: WebClipperDestination) => void
  onFolderIdChange: (nextValue: string) => void
  onWorkspaceIdChange: (nextValue: string) => void
}

const destinationOptions: WebClipperDestination[] = [
  "note",
  "workspace",
  "both"
]

const ClipDestinationFields = ({
  destinationMode,
  folderId,
  folderOptions,
  folderOptionsError,
  folderValidation,
  isFolderOptionsLoading,
  isWorkspaceOptionsLoading,
  workspaceOptions,
  workspaceOptionsError,
  workspaceId,
  workspaceValidation,
  onDestinationChange,
  onFolderIdChange,
  onWorkspaceIdChange
}: ClipDestinationFieldsProps) => {
  const { t } = useTranslation()
  const hasFolderOptions = folderOptions.length > 0
  const hasWorkspaceOptions = workspaceOptions.length > 0
  const folderInput = (
    <input
      id="clip-folder-id"
      type="number"
      inputMode="numeric"
      min="1"
      step="1"
      value={folderId}
      onChange={(event) => onFolderIdChange(event.target.value)}
      className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-text"
      placeholder={t(
        "sidepanel:clipper.folderPlaceholder",
        "42"
      )}
      aria-invalid={folderValidation ? "true" : "false"}
    />
  )
  const workspaceInput = (
    <input
      id="clip-workspace-id"
      type="text"
      value={workspaceId}
      onChange={(event) => onWorkspaceIdChange(event.target.value)}
      className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-text"
      placeholder={t(
        "sidepanel:clipper.workspacePlaceholder",
        "workspace-alpha"
      )}
      aria-invalid={workspaceValidation ? "true" : "false"}
    />
  )

  return (
    <section className="panel-card p-3">
      <fieldset className="space-y-3">
        <legend className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-muted">
          {t("sidepanel:clipper.destinationLegend", "Destination")}
        </legend>

        <div className="grid grid-cols-3 gap-2">
          {destinationOptions.map((option) => {
            const label =
              option === "note"
                ? t("sidepanel:clipper.destinationNote", "Note")
                : option === "workspace"
                  ? t("sidepanel:clipper.destinationWorkspace", "Workspace")
                  : t("sidepanel:clipper.destinationBoth", "Both")

            return (
              <label
                key={option}
                className={`flex cursor-pointer items-center justify-center rounded-lg border px-3 py-2 text-sm font-medium transition ${
                  destinationMode === option
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border bg-background text-text"
                }`}
              >
                <input
                  type="radio"
                  name="clip-destination-mode"
                  value={option}
                  checked={destinationMode === option}
                  onChange={(event: ChangeEvent<HTMLInputElement>) =>
                    onDestinationChange(
                      event.target.value as WebClipperDestination
                    )
                  }
                  className="sr-only"
                />
                {label}
              </label>
            )
          })}
        </div>

        {destinationMode !== "workspace" ? (
          <div className="space-y-2">
            {isFolderOptionsLoading ? (
              <p className="text-sm text-text-muted">
                {t("sidepanel:clipper.folderLoading", "Loading folders...")}
              </p>
            ) : null}
            {folderOptionsError ? (
              <p className="text-sm text-amber-700">
                {folderOptionsError}
              </p>
            ) : null}
            {hasFolderOptions ? (
              <>
                <label className="block text-sm font-medium text-text" htmlFor="clip-folder-picker">
                  {t("sidepanel:clipper.folderPickerLabel", "Folder")}
                </label>
                <select
                  id="clip-folder-picker"
                  value={folderId}
                  onChange={(event) => onFolderIdChange(event.target.value)}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-text"
                  aria-invalid={folderValidation ? "true" : "false"}
                >
                  <option value="">
                    {t("sidepanel:clipper.folderPickerPlaceholder", "No folder")}
                  </option>
                  {folderOptions.map((folder) => (
                    <option key={folder.id} value={String(folder.id)}>
                      {folder.path || folder.name || folder.id}
                    </option>
                  ))}
                </select>
                <details className="rounded-lg border border-border bg-surface2 px-3 py-2">
                  <summary className="cursor-pointer text-sm font-medium text-text-muted">
                    {t(
                      "sidepanel:clipper.folderAdvancedSummary",
                      "Advanced: enter folder ID manually"
                    )}
                  </summary>
                  <div className="mt-2 space-y-2">
                    <label className="block text-sm font-medium text-text" htmlFor="clip-folder-id">
                      {t("sidepanel:clipper.folderLabel", "Folder ID")}
                    </label>
                    {folderInput}
                  </div>
                </details>
              </>
            ) : !isFolderOptionsLoading ? (
              <>
                <label className="block text-sm font-medium text-text" htmlFor="clip-folder-id">
                  {t("sidepanel:clipper.folderLabel", "Folder ID")}
                </label>
                {folderInput}
              </>
            ) : null}
            {folderValidation ? (
              <p className="text-sm text-red-600">{folderValidation}</p>
            ) : null}
          </div>
        ) : null}

        {destinationMode !== "note" ? (
          <div className="space-y-2">
            {isWorkspaceOptionsLoading ? (
              <p className="text-sm text-text-muted">
                {t("sidepanel:clipper.workspaceLoading", "Loading workspaces...")}
              </p>
            ) : null}
            {workspaceOptionsError ? (
              <p className="text-sm text-amber-700">
                {workspaceOptionsError}
              </p>
            ) : null}
            {hasWorkspaceOptions ? (
              <>
                <label className="block text-sm font-medium text-text" htmlFor="clip-workspace-picker">
                  {t("sidepanel:clipper.workspacePickerLabel", "Workspace")}
                </label>
                <select
                  id="clip-workspace-picker"
                  value={workspaceId}
                  onChange={(event) => onWorkspaceIdChange(event.target.value)}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-text"
                  aria-invalid={workspaceValidation ? "true" : "false"}
                >
                  <option value="">
                    {t("sidepanel:clipper.workspacePickerPlaceholder", "Select a workspace")}
                  </option>
                  {workspaceOptions.map((workspace) => (
                    <option key={workspace.id} value={workspace.id}>
                      {workspace.name
                        ? `${workspace.name} (${workspace.id})`
                        : workspace.id}
                    </option>
                  ))}
                </select>
                <details className="rounded-lg border border-border bg-surface2 px-3 py-2">
                  <summary className="cursor-pointer text-sm font-medium text-text-muted">
                    {t(
                      "sidepanel:clipper.workspaceAdvancedSummary",
                      "Advanced: enter workspace ID manually"
                    )}
                  </summary>
                  <div className="mt-2 space-y-2">
                    <label className="block text-sm font-medium text-text" htmlFor="clip-workspace-id">
                      {t("sidepanel:clipper.workspaceLabel", "Workspace ID")}
                    </label>
                    {workspaceInput}
                  </div>
                </details>
              </>
            ) : !isWorkspaceOptionsLoading ? (
              <>
                <label className="block text-sm font-medium text-text" htmlFor="clip-workspace-id">
                  {t("sidepanel:clipper.workspaceLabel", "Workspace ID")}
                </label>
                {workspaceInput}
              </>
            ) : null}
            {workspaceValidation ? (
              <p className="text-sm text-red-600">{workspaceValidation}</p>
            ) : null}
          </div>
        ) : null}
      </fieldset>
    </section>
  )
}

export default ClipDestinationFields
