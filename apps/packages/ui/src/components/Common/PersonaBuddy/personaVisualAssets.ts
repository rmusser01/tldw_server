import type {
  PersonaVisualAnimation,
  PersonaVisualAsset,
  PersonaVisualFrame,
  PersonaVisualPack
} from "@/types/persona-visuals"

export const getAssetsById = (
  pack: PersonaVisualPack | null | undefined
): Record<string, PersonaVisualAsset> => {
  if (!pack) return {}
  if (pack.assets_by_id && Object.keys(pack.assets_by_id).length > 0) {
    return pack.assets_by_id
  }
  const assets: Record<string, PersonaVisualAsset> = {}
  for (const asset of pack.assets || []) {
    if (asset?.id) assets[asset.id] = asset
  }
  return assets
}

export const normalizeFrames = (
  animation: PersonaVisualAnimation | null | undefined
): PersonaVisualFrame[] => {
  if (!animation) return []
  if (Array.isArray(animation.frames) && animation.frames.length > 0) {
    return animation.frames.filter((frame) => Boolean(frame?.asset_id))
  }
  return (animation.asset_ids || [])
    .filter((assetId) => Boolean(String(assetId || "").trim()))
    .map((assetId) => ({ asset_id: String(assetId) }))
}
