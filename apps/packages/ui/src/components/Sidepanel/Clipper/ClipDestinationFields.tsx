import type { ChangeEvent } from "react"
import type { WebClipperDestination } from "@/services/web-clipper/types"
import { useTranslation } from "react-i18next"

type ClipDestinationFieldsProps = {
  destinationMode: WebClipperDestination
  folderId: string
  folderValidation: string | null
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
  folderValidation,
  workspaceId,
  workspaceValidation,
  onDestinationChange,
  onFolderIdChange,
  onWorkspaceIdChange
}: ClipDestinationFieldsProps) => {
  const { t } = useTranslation()

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
            <label className="block text-sm font-medium text-text" htmlFor="clip-folder-id">
              {t("sidepanel:clipper.folderLabel", "Folder ID")}
            </label>
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
            {folderValidation ? (
              <p className="text-sm text-red-600">{folderValidation}</p>
            ) : null}
          </div>
        ) : null}

        {destinationMode !== "note" ? (
          <div className="space-y-2">
            <label className="block text-sm font-medium text-text" htmlFor="clip-workspace-id">
              {t("sidepanel:clipper.workspaceLabel", "Workspace ID")}
            </label>
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
