import { DynamicUISourceFallback } from "../DynamicUISourceFallback"
import type { DynamicUIRendererProps } from "../registry"

const OpenUIRenderer = ({ envelope }: DynamicUIRendererProps) => (
  <DynamicUISourceFallback
    source={envelope.source}
    error="OpenUI runtime is not enabled yet."
  />
)

export default OpenUIRenderer
