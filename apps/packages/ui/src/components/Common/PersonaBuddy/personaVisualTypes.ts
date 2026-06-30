export type PersonaVisualRenderError =
  | "missing_animation"
  | "missing_asset"
  | "unsupported_region"

export type PersonaVisualRenderErrorHandler = (
  error: PersonaVisualRenderError | null
) => void
