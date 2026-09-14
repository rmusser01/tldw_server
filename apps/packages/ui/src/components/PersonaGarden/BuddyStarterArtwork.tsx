import React from "react"
import { Button } from "antd"
import { useTranslation } from "react-i18next"
import { getPersonaVisualStarterPack } from "@/services/persona-visuals"
import { normalizeFrames } from "@/components/Common/PersonaBuddy/personaVisualAssets"
import { usePersonaVisualAssetUrls } from "@/components/Common/PersonaBuddy/usePersonaVisualAssetUrls"
import type {
  PersonaVisualAsset,
  PersonaVisualFrame,
  PersonaVisualManifest
} from "@/types/persona-visuals"

type Artwork = { asset: PersonaVisualAsset; frame: PersonaVisualFrame }

const ArtworkImage = ({
  artwork,
  label,
  onReady,
  onError
}: {
  artwork: Artwork
  label: string
  onReady: () => void
  onError: () => void
}) => {
  const { asset, frame } = artwork
  const resolveUrl = usePersonaVisualAssetUrls({ [asset.id]: asset }, asset)
  const url = resolveUrl(asset)
  React.useEffect(() => {
    if (url === null) onError()
  }, [url, onError])
  if (!url) return null
  if (!frame.region)
    return (
      <img
        src={url}
        alt={label}
        className="h-40 w-full object-contain"
        onLoad={onReady}
        onError={onError}
      />
    )
  const { x, y, width, height } = frame.region
  /* eslint-disable react/no-unknown-property -- SVGImageElement supports load/error events to confirm the cropped preview. */
  return (
    <svg
      role="img"
      aria-label={label}
      viewBox={`${x} ${y} ${width} ${height}`}
      className="h-40 w-full"
      preserveAspectRatio="xMidYMid meet"
    >
      <image
        href={url}
        width={asset.width ?? undefined}
        height={asset.height ?? undefined}
        onLoad={onReady}
        onError={onError}
      />
    </svg>
  )
  /* eslint-enable react/no-unknown-property */
}

export const BuddyStarterArtwork = ({
  starterId,
  title,
  onReadyChange
}: {
  starterId: string
  title: string
  onReadyChange: (ready: boolean) => void
}) => {
  const { t } = useTranslation("sidepanel")
  const [artwork, setArtwork] = React.useState<Artwork | null>(null)
  const [error, setError] = React.useState(false)
  const [loaded, setLoaded] = React.useState(false)
  const [attempt, setAttempt] = React.useState(0)
  const ready = React.useCallback(() => {
    setLoaded(true)
    onReadyChange(true)
  }, [onReadyChange])
  const failed = React.useCallback(() => {
    setError(true)
    onReadyChange(false)
  }, [onReadyChange])
  React.useEffect(() => {
    let cancelled = false
    setArtwork(null)
    setError(false)
    setLoaded(false)
    onReadyChange(false)
    void getPersonaVisualStarterPack(starterId)
      .then((detail) => {
        if (cancelled) return
        const manifest = detail.manifest as PersonaVisualManifest
        const animationId = manifest.states?.idle?.animation_id
        const animation = animationId
          ? manifest.animations?.[animationId]
          : null
        const frames = normalizeFrames(animation)
        const frame = frames[animation?.preview_frame ?? 0] ?? frames[0]
        const source = detail.assets.find(
          (asset) => asset.asset_key === frame?.asset_id
        )
        if (!frame || !source) throw new Error("No preview artwork")
        setArtwork({
          frame,
          asset: {
            id: source.asset_key,
            asset_role: source.asset_role,
            mime_type: source.mime_type,
            width: source.width,
            height: source.height,
            url: `/api/v1/persona/visual-starter-packs/${encodeURIComponent(starterId)}/assets/${encodeURIComponent(source.asset_key)}/content`
          }
        })
      })
      .catch(() => {
        if (!cancelled) failed()
      })
    return () => {
      cancelled = true
    }
  }, [starterId, attempt, onReadyChange, failed])
  return (
    <div className="flex min-h-40 flex-col items-center justify-center gap-2 bg-surface2 p-2 text-sm text-text-muted">
      {error ? (
        <>
          <p role="status">
            {t("personaGarden.visuals.builder.previewUnavailable", {
              defaultValue: "Preview unavailable. Retry to view this Buddy."
            })}
          </p>
          <Button size="small" onClick={() => setAttempt((value) => value + 1)}>
            {t("personaGarden.visuals.builder.retryPreview", {
              defaultValue: "Retry preview"
            })}
          </Button>
        </>
      ) : (
        <>
          {!loaded ? (
            <span role="status">
              {t("personaGarden.visuals.builder.loadingPreview", {
                defaultValue: "Loading artwork…"
              })}
            </span>
          ) : null}
          {artwork ? (
            <ArtworkImage
              key={attempt}
              artwork={artwork}
              label={`${title} ${t("personaGarden.visuals.builder.previewLabel", { defaultValue: "preview" })}`}
              onReady={ready}
              onError={failed}
            />
          ) : null}
        </>
      )}
    </div>
  )
}
