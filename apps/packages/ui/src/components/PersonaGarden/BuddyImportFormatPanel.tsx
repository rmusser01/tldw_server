import React from "react"
import { Tag } from "antd"
import { FileArchive, PackageOpen } from "lucide-react"
import { useTranslation } from "react-i18next"

import type { BuddyBuilderSource } from "./buddyBuilderState"
import {
  CODEX_PET_ARCHIVE_EXTENSION,
  NATIVE_PERSONA_VISUAL_PACK_EXTENSION
} from "./buddyBuilderArchive"

export type BuddyImportFormatPanelProps = {
  source: BuddyBuilderSource | null
  importPreviewPanel: React.ReactNode
}

export const BuddyImportFormatPanel: React.FC<BuddyImportFormatPanelProps> = ({
  source,
  importPreviewPanel
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  if (source !== "codex_import" && source !== "native_import") return null

  const isCodex = source === "codex_import"
  const title = isCodex
    ? t("sidepanel:personaGarden.visuals.builder.codexImportSource", {
        defaultValue: "Import Codex/Petdex pet"
      })
    : t("sidepanel:personaGarden.visuals.builder.nativeImportSource", {
        defaultValue: "Import Persona Visual pack"
      })
  const detail = isCodex
    ? t("sidepanel:personaGarden.visuals.builder.codexImportDetail", {
        defaultValue:
          "Use a Codex/Petdex .zip with pet.json or petjson.json and a compatible spritesheet."
      })
    : t("sidepanel:personaGarden.visuals.builder.nativeImportDetail", {
        defaultValue:
          "Use a .tldw-persona-vpack archive exported from Persona Visual Packs."
      })

  return (
    <section
      data-testid="buddy-builder-import-panel"
      className="rounded-md border border-border bg-bg p-3"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2 text-sm font-medium text-text">
            {isCodex ? (
              <PackageOpen className="h-4 w-4" />
            ) : (
              <FileArchive className="h-4 w-4" />
            )}
            {title}
          </div>
          <div className="mt-1 text-xs leading-5 text-text-muted">{detail}</div>
        </div>
        <Tag>
          {isCodex
            ? CODEX_PET_ARCHIVE_EXTENSION
            : NATIVE_PERSONA_VISUAL_PACK_EXTENSION}
        </Tag>
      </div>
      <div className="mt-3">{importPreviewPanel}</div>
    </section>
  )
}

export default BuddyImportFormatPanel
