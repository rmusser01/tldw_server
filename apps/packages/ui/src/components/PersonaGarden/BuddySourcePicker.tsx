import React from "react"
import { Tag } from "antd"
import { Box, CopyPlus, FolderOpen, PackageOpen, PenLine, Upload } from "lucide-react"
import { useTranslation } from "react-i18next"

import {
  CODEX_PET_ARCHIVE_EXTENSION,
  NATIVE_PERSONA_VISUAL_PACK_EXTENSION
} from "./buddyBuilderArchive"
import type { BuddyBuilderSource } from "./buddyBuilderState"

export type BuddySourcePickerProps = {
  selectedSource: BuddyBuilderSource | null
  onSelectSource: (source: BuddyBuilderSource) => void
  onStartBlank?: () => void
  onOpenLibrary?: () => void
  onOpenDuplicate?: () => void
}

type SourceOption = {
  source: BuddyBuilderSource
  label: string
  description: string
  icon: React.ReactNode
  tag?: string
}

export const BuddySourcePicker: React.FC<BuddySourcePickerProps> = ({
  selectedSource,
  onSelectSource,
  onStartBlank,
  onOpenLibrary,
  onOpenDuplicate
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const options: SourceOption[] = [
    {
      source: "bundled",
      label: t("sidepanel:personaGarden.visuals.builder.bundledSource", {
        defaultValue: "Bundled Buddy"
      }),
      description: t("sidepanel:personaGarden.visuals.builder.bundledSourceHelp", {
        defaultValue: "Start from one of the reviewed Basic defaults."
      }),
      icon: <PackageOpen className="h-4 w-4" />,
      tag: t("sidepanel:personaGarden.visuals.builder.recommendedTag", {
        defaultValue: "recommended"
      })
    },
    {
      source: "codex_import",
      label: t("sidepanel:personaGarden.visuals.builder.codexImportSource", {
        defaultValue: "Import Codex/Petdex pet"
      }),
      description: t("sidepanel:personaGarden.visuals.builder.codexImportHelp", {
        defaultValue: "Reuse a Codex-compatible pet archive."
      }),
      icon: <Upload className="h-4 w-4" />,
      tag: CODEX_PET_ARCHIVE_EXTENSION
    },
    {
      source: "native_import",
      label: t("sidepanel:personaGarden.visuals.builder.nativeImportSource", {
        defaultValue: "Import Persona Visual pack"
      }),
      description: t("sidepanel:personaGarden.visuals.builder.nativeImportHelp", {
        defaultValue: "Restore or move a native Persona Visual archive."
      }),
      icon: <Upload className="h-4 w-4" />,
      tag: NATIVE_PERSONA_VISUAL_PACK_EXTENSION
    },
    {
      source: "library",
      label: t("sidepanel:personaGarden.visuals.builder.librarySource", {
        defaultValue: "Use library pack"
      }),
      description: t("sidepanel:personaGarden.visuals.builder.librarySourceHelp", {
        defaultValue: "Attach a saved visual pack already in your library."
      }),
      icon: <FolderOpen className="h-4 w-4" />
    },
    {
      source: "duplicate",
      label: t("sidepanel:personaGarden.visuals.builder.duplicateSource", {
        defaultValue: "Duplicate from persona"
      }),
      description: t("sidepanel:personaGarden.visuals.builder.duplicateSourceHelp", {
        defaultValue: "Copy a pack from another persona into this one."
      }),
      icon: <CopyPlus className="h-4 w-4" />
    },
    {
      source: "blank",
      label: t("sidepanel:personaGarden.visuals.builder.blankSource", {
        defaultValue: "Start blank"
      }),
      description: t("sidepanel:personaGarden.visuals.builder.blankSourceHelp", {
        defaultValue: "Create a draft when the buddy needs custom states."
      }),
      icon: <PenLine className="h-4 w-4" />
    }
  ]

  const handleSelect = (source: BuddyBuilderSource) => {
    onSelectSource(source)
    if (source === "blank") onStartBlank?.()
    if (source === "library") onOpenLibrary?.()
    if (source === "duplicate") onOpenDuplicate?.()
  }

  return (
    <section data-testid="buddy-builder-source-picker" className="space-y-2">
      <div className="flex items-center gap-2 text-sm font-medium text-text">
        <Box className="h-4 w-4" />
        {t("sidepanel:personaGarden.visuals.builder.sourceStep", {
          defaultValue: "Choose a source"
        })}
      </div>
      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
        {options.map((option) => {
          const isSelected = selectedSource === option.source
          return (
            <button
              type="button"
              key={option.source}
              aria-label={option.label}
              aria-pressed={isSelected}
              className={`min-h-[4.5rem] rounded-md border px-3 py-2 text-left transition-colors hover:border-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary ${
                isSelected ? "border-primary bg-primary/10" : "border-border bg-bg"
              }`}
              onClick={() => handleSelect(option.source)}
            >
              <span className="flex w-full items-start gap-2">
                <span className="mt-0.5 text-text-muted">{option.icon}</span>
                <span className="min-w-0 flex-1">
                  <span className="flex flex-wrap items-center gap-1">
                    <span className="font-medium text-text">{option.label}</span>
                    {option.tag ? <Tag>{option.tag}</Tag> : null}
                  </span>
                  <span className="mt-1 block text-xs leading-5 text-text-muted">
                    {option.description}
                  </span>
                </span>
              </span>
            </button>
          )
        })}
      </div>
    </section>
  )
}

export default BuddySourcePicker
