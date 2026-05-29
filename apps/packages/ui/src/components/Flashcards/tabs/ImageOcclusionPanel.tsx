import { Button, Card, Input, Typography } from "antd"
import React from "react"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"

import {
  finalizeNormalizedOcclusionRect,
  formatNormalizedOcclusionRect,
  resolveNextSelectedOcclusionId,
  type NormalizedOcclusionRect
} from "../utils/image-occlusion"

const MIN_REGION_SIZE_PX = 8

export interface ImageOcclusionRegion extends NormalizedOcclusionRect {
  id: string
  label: string
}

interface ImageOcclusionPanelState {
  sourceFile: File | null
  sourceUrl: string | null
  regions: ImageOcclusionRegion[]
  selectedRegionId: string | null
}

interface ImageOcclusionPanelProps {
  onChange?: (state: ImageOcclusionPanelState) => void
}

interface DragState {
  startClientX: number
  startClientY: number
  currentClientX: number
  currentClientY: number
  bounds: {
    left: number
    top: number
    width: number
    height: number
  }
}

const { Text } = Typography

export const ImageOcclusionPanel: React.FC<ImageOcclusionPanelProps> = ({ onChange }) => {
  const { t } = useTranslation(["option", "common"])
  const [sourceFile, setSourceFile] = React.useState<File | null>(null)
  const [sourceUrl, setSourceUrl] = React.useState<string | null>(null)
  const [regions, setRegions] = React.useState<ImageOcclusionRegion[]>([])
  const [selectedRegionId, setSelectedRegionId] = React.useState<string | null>(null)
  const [dragState, setDragState] = React.useState<DragState | null>(null)
  const nextRegionIdRef = React.useRef(1)

  React.useEffect(() => {
    onChange?.({
      sourceFile,
      sourceUrl,
      regions,
      selectedRegionId
    })
  }, [onChange, regions, selectedRegionId, sourceFile, sourceUrl])

  React.useEffect(() => {
    return () => {
      if (sourceUrl) {
        URL.revokeObjectURL(sourceUrl)
      }
    }
  }, [sourceUrl])

  const handleSourceFileChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const nextFile = event.target.files?.[0] ?? null
      event.target.value = ""
      if (!nextFile) return

      setSourceFile(nextFile)
      setRegions([])
      setSelectedRegionId(null)
      nextRegionIdRef.current = 1

      setSourceUrl((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous)
        }
        return URL.createObjectURL(nextFile)
      })
    },
    []
  )

  const handlePointerDown = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect()
    if (bounds.width <= 0 || bounds.height <= 0) {
      return
    }
    setDragState({
      startClientX: event.clientX,
      startClientY: event.clientY,
      currentClientX: event.clientX,
      currentClientY: event.clientY,
      bounds: {
        left: bounds.left,
        top: bounds.top,
        width: bounds.width,
        height: bounds.height
      }
    })
  }, [])

  const handlePointerMove = React.useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    setDragState((previous) =>
      previous
        ? {
            ...previous,
            currentClientX: event.clientX,
            currentClientY: event.clientY
          }
        : previous
    )
  }, [])

  const handlePointerUp = React.useCallback(() => {
    setDragState((previous) => {
      if (!previous) {
        return previous
      }

      const normalized = finalizeNormalizedOcclusionRect({
        startClientX: previous.startClientX,
        startClientY: previous.startClientY,
        endClientX: previous.currentClientX,
        endClientY: previous.currentClientY,
        bounds: previous.bounds,
        minSizePx: MIN_REGION_SIZE_PX
      })

      if (!normalized) {
        return null
      }

      const regionId = `region-${nextRegionIdRef.current++}`
      setRegions((current) => [
        ...current,
        {
          id: regionId,
          label: "",
          ...normalized
        }
      ])
      setSelectedRegionId(regionId)
      return null
    })
  }, [])

  const handleLabelChange = React.useCallback((regionId: string, label: string) => {
    setRegions((current) =>
      current.map((region) => (region.id === regionId ? { ...region, label } : region))
    )
  }, [])

  const handleRemoveRegion = React.useCallback((regionId: string) => {
    setRegions((current) => {
      setSelectedRegionId((currentSelected) => {
        if (currentSelected !== regionId) {
          return currentSelected
        }
        return resolveNextSelectedOcclusionId(
          current.map((region) => region.id),
          regionId
        )
      })
      return current.filter((region) => region.id !== regionId)
    })
  }, [])

  const draftRect = React.useMemo(() => {
    if (!dragState) {
      return null
    }
    return finalizeNormalizedOcclusionRect({
      startClientX: dragState.startClientX,
      startClientY: dragState.startClientY,
      endClientX: dragState.currentClientX,
      endClientY: dragState.currentClientY,
      bounds: dragState.bounds,
      minSizePx: 0
    })
  }, [dragState])

  const selectedRegionLabel = React.useMemo(() => {
    const selectedIndex = regions.findIndex((region) => region.id === selectedRegionId)
    if (selectedIndex === -1) {
      return t("option:flashcards.occlusionNoRegionSelected", {
        defaultValue: "No region selected"
      })
    }
    return t("option:flashcards.occlusionSelectedRegion", {
      defaultValue: "Region {{index}}",
      index: selectedIndex + 1
    })
  }, [regions, selectedRegionId, t])

  return (
    <div className="flex flex-col gap-3">
      <Text type="secondary">
        {t("option:flashcards.occlusionHelp", {
          defaultValue:
            "Upload one image, drag rectangular occlusions, label each region, then generate drafts."
        })}
      </Text>

      <label className="flex flex-col gap-2">
        <span className="font-medium">
          {t("option:flashcards.occlusionSourceLabel", {
            defaultValue: "Source image"
          })}
        </span>
        <input
          type="file"
          accept="image/png,image/jpeg,image/webp,image/gif"
          aria-label={t("option:flashcards.occlusionUploadSourceAria", {
            defaultValue: "Upload image occlusion source"
          })}
          onChange={handleSourceFileChange}
        />
      </label>

      {!sourceUrl ? (
        <Alert
          variant="info"
          title={t("option:flashcards.occlusionAwaitingImage", {
            defaultValue: "Choose an image to begin authoring occlusions."
          })}
        />
      ) : (
        <Card size="small" styles={{ body: { padding: 12 } }}>
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_320px]">
            <div className="space-y-2">
              <div className="relative overflow-hidden rounded-md border border-border bg-black/5">
                <img
                  src={sourceUrl}
                  alt={t("option:flashcards.occlusionPreviewAlt", {
                    defaultValue: "Image occlusion source preview"
                  })}
                  className="block w-full max-h-[420px] object-contain select-none"
                  draggable={false}
                />
                <div
                  data-testid="image-occlusion-overlay"
                  className="absolute inset-0 cursor-crosshair touch-none"
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  onPointerLeave={handlePointerUp}
                >
                  {regions.map((region, index) => (
                    <button
                      key={region.id}
                      type="button"
                      data-testid={`image-occlusion-region-${region.id}`}
                      className={`absolute border-2 text-[11px] font-medium ${
                        selectedRegionId === region.id
                          ? "border-amber-400 bg-amber-300/20"
                          : "border-sky-500 bg-sky-400/15"
                      }`}
                      style={{
                        left: `${region.x * 100}%`,
                        top: `${region.y * 100}%`,
                        width: `${region.width * 100}%`,
                        height: `${region.height * 100}%`
                      }}
                      onClick={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                        setSelectedRegionId(region.id)
                      }}
                    >
                      <span className="absolute left-1 top-1 rounded bg-black/70 px-1 py-0.5 text-white">
                        {index + 1}
                      </span>
                    </button>
                  ))}
                  {draftRect && (
                    <div
                      className="absolute border-2 border-dashed border-white/90 bg-white/10"
                      style={{
                        left: `${draftRect.x * 100}%`,
                        top: `${draftRect.y * 100}%`,
                        width: `${draftRect.width * 100}%`,
                        height: `${draftRect.height * 100}%`
                      }}
                    />
                  )}
                </div>
              </div>
              <Text type="secondary">
                {t("option:flashcards.occlusionCanvasHint", {
                  defaultValue:
                    "Drag on the image to create a region. Click a region to select it."
                })}
              </Text>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Text strong>
                  {t("option:flashcards.occlusionRegionsTitle", {
                    defaultValue: "Regions"
                  })}
                </Text>
                <Text data-testid="image-occlusion-selected-region">{selectedRegionLabel}</Text>
              </div>

              {regions.length === 0 ? (
                <Alert
                  variant="info"
                  title={t("option:flashcards.occlusionNoRegions", {
                    defaultValue: "No regions yet. Draw on the image to add one."
                  })}
                />
              ) : (
                <div className="space-y-2">
                  {regions.map((region, index) => (
                    <Card
                      key={region.id}
                      size="small"
                      data-testid={`image-occlusion-region-row-${region.id}`}
                      className={
                        selectedRegionId === region.id ? "border-amber-400 shadow-sm" : undefined
                      }
                      title={t("option:flashcards.occlusionRegionTitle", {
                        defaultValue: "Region {{index}}",
                        index: index + 1
                      })}
                      extra={
                        <Button
                          type="text"
                          danger
                          size="small"
                          data-testid={`image-occlusion-remove-region-${region.id}`}
                          onClick={(event) => {
                            event.stopPropagation()
                            handleRemoveRegion(region.id)
                          }}
                        >
                          {t("common:remove", { defaultValue: "Remove" })}
                        </Button>
                      }
                      onClick={() => setSelectedRegionId(region.id)}
                    >
                      <div className="space-y-2">
                        <Input
                          value={region.label}
                          data-testid={`image-occlusion-region-label-${region.id}`}
                          placeholder={t("option:flashcards.occlusionLabelPlaceholder", {
                            defaultValue: "Answer label"
                          })}
                          onChange={(event) => handleLabelChange(region.id, event.target.value)}
                        />
                        <Text
                          type="secondary"
                          data-testid={`image-occlusion-region-geometry-${region.id}`}
                        >
                          {formatNormalizedOcclusionRect(region)}
                        </Text>
                      </div>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}

export default ImageOcclusionPanel
