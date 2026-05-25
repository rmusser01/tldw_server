import React from "react"
import { Card, Typography } from "antd"
import { useTranslation } from "react-i18next"

import type { FlashcardsGenerateIntent } from "@/services/tldw/flashcards-generate-handoff"
import type { StudyPackIntent } from "@/services/tldw/study-pack-handoff"

import { StudyPackCreateDrawer } from "../components/StudyPackCreateDrawer"
import { useImportLimitsQuery } from "../hooks"
import { ImageOcclusionTransferPanel } from "./ImageOcclusionTransferPanel"
import { ExportPanel } from "./ImportExport/ExportPanel"
import { GeneratePanel } from "./ImportExport/GeneratePanel"
import { ImportPanel } from "./ImportExport/ImportPanel"
import { StudyPackPanel } from "./ImportExport/StudyPackPanel"
import {
  normalizeImportLimits,
  type TransferActionSummary,
  type TransferActionSummaryInput
} from "./ImportExport/shared"

const { Text } = Typography

/**
 * Import/Export tab for flashcards.
 */
type ImportExportTabProps = {
  generateIntent?: FlashcardsGenerateIntent | null
  studyPackIntent?: StudyPackIntent | null
}

export const ImportExportTab: React.FC<ImportExportTabProps> = ({
  generateIntent,
  studyPackIntent
}) => {
  const { t } = useTranslation(["option", "common"])
  const limitsQuery = useImportLimitsQuery()
  const [lastTransferAction, setLastTransferAction] =
    React.useState<TransferActionSummary | null>(null)
  const [studyPackDrawerOpen, setStudyPackDrawerOpen] = React.useState(false)
  const importLimitsText = React.useMemo(() => {
    const importLimits = normalizeImportLimits(limitsQuery.data)
    if (!importLimits) return null
    return t("option:flashcards.transferSummaryLimitsValue", {
      defaultValue:
        "{{lines}} lines / {{lineBytes}} bytes per line / {{fieldBytes}} bytes per field",
      lines: importLimits.maxLines.toLocaleString(),
      lineBytes: importLimits.maxLineLengthBytes.toLocaleString(),
      fieldBytes: importLimits.maxFieldLengthBytes.toLocaleString()
    })
  }, [limitsQuery.data, t])

  React.useEffect(() => {
    if (studyPackIntent) {
      setStudyPackDrawerOpen(true)
    }
  }, [studyPackIntent])

  const handleTransferAction = React.useCallback((summary: TransferActionSummaryInput) => {
    setLastTransferAction({
      ...summary,
      at: new Date().toISOString()
    })
  }, [])

  const lastTransferActionText = React.useMemo(() => {
    if (!lastTransferAction) {
      return t("option:flashcards.transferSummaryNoAction", {
        defaultValue: "No transfer actions yet in this session."
      })
    }
    const areaLabel =
      lastTransferAction.area === "import"
        ? t("option:flashcards.importTitle", { defaultValue: "Import Flashcards" })
        : lastTransferAction.area === "export"
          ? t("option:flashcards.exportTitle", { defaultValue: "Export Flashcards" })
          : lastTransferAction.area === "occlusion"
            ? t("option:flashcards.occlusionTitle", {
                defaultValue: "Image Occlusion"
              })
            : t("option:flashcards.generateTitle", {
                defaultValue: "Generate Flashcards"
              })
    return t("option:flashcards.transferSummaryLastAction", {
      defaultValue: "{{area}} · {{message}} · {{time}}",
      area: areaLabel,
      message: lastTransferAction.message,
      time: new Date(lastTransferAction.at).toLocaleTimeString()
    })
  }, [lastTransferAction, t])

  return (
    <div className="grid gap-4 grid-cols-1 xl:grid-cols-4">
      <Card
        className="xl:col-span-4"
        title={t("option:flashcards.studyPackLauncherTitle", {
          defaultValue: "Study packs"
        })}
        data-testid="flashcards-study-pack-launcher"
      >
        <StudyPackPanel onLaunch={() => setStudyPackDrawerOpen(true)} />
      </Card>
      <Card
        className="xl:col-span-4"
        title={t("option:flashcards.transferSummaryTitle", {
          defaultValue: "Transfer summary"
        })}
        data-testid="flashcards-transfer-summary"
      >
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <div data-testid="flashcards-transfer-summary-formats">
            <Text strong className="block">
              {t("option:flashcards.transferSummaryFormatsLabel", {
                defaultValue: "Supported formats"
              })}
            </Text>
            <Text type="secondary">
              {t("option:flashcards.transferSummaryFormatsValue", {
                defaultValue:
                  "Import: CSV, TSV, JSON, JSONL, Structured Q&A, APKG · Author: Generate, Image Occlusion · Export: TSV, CSV, JSON, APKG"
              })}
            </Text>
          </div>
          <div data-testid="flashcards-transfer-summary-limits">
            <Text strong className="block">
              {t("option:flashcards.transferSummaryLimitsLabel", {
                defaultValue: "Current import limits"
              })}
            </Text>
            <Text type="secondary">
              {importLimitsText
                ? importLimitsText
                : t("option:flashcards.transferSummaryLimitsUnknown", {
                    defaultValue: "Limits unavailable"
                  })}
            </Text>
          </div>
          <div data-testid="flashcards-transfer-summary-last-action">
            <Text strong className="block">
              {t("option:flashcards.transferSummaryLastActionLabel", {
                defaultValue: "Last action"
              })}
            </Text>
            <Text
              type={
                lastTransferAction?.status === "error"
                  ? "danger"
                  : lastTransferAction?.status === "warning"
                    ? "warning"
                    : "secondary"
              }
            >
              {lastTransferActionText}
            </Text>
          </div>
        </div>
      </Card>
      <Card
        title={t("option:flashcards.importTitle", {
          defaultValue: "Import Flashcards"
        })}
      >
        <ImportPanel onTransferAction={handleTransferAction} />
      </Card>
      <Card
        title={t("option:flashcards.exportTitle", {
          defaultValue: "Export Flashcards"
        })}
      >
        <ExportPanel onTransferAction={handleTransferAction} />
      </Card>
      <Card
        title={t("option:flashcards.generateTitle", {
          defaultValue: "Generate Flashcards"
        })}
      >
        <GeneratePanel
          initialIntent={generateIntent || null}
          onTransferAction={handleTransferAction}
        />
      </Card>
      <Card
        title={t("option:flashcards.occlusionTitle", {
          defaultValue: "Image Occlusion"
        })}
      >
        <ImageOcclusionTransferPanel onTransferAction={handleTransferAction} />
      </Card>
      <StudyPackCreateDrawer
        open={studyPackDrawerOpen}
        onClose={() => setStudyPackDrawerOpen(false)}
        initialIntent={studyPackIntent || null}
      />
    </div>
  )
}

export default ImportExportTab
