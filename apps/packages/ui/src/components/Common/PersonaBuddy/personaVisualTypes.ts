export type PersonaVisualRenderError =
  | "missing_animation"
  | "missing_asset"
  | "unsupported_region"
  | "asset_load_failed"
  | "static_asset_unsupported"

export type PersonaVisualRenderErrorHandler = (
  error: PersonaVisualRenderError | null
) => void
