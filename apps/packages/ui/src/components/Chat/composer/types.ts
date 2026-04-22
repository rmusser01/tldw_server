/**
 * Shared types for the chat composer module. Both the Playground
 * (`/chat`) and Sidepanel (extension) composers consume these types. Variants
 * (V1/V3/V5) receive a `ChatComposerContext` — the hook family's return value
 * — and render against it without knowing which surface they're on.
 *
 * Populated incrementally across Phase 2 as hooks are lifted from
 * Playground/Sidepanel into the shared `hooks/` directory.
 */

/** Which parent surface is hosting the composer. */
export type ChatComposerSurface = "playground" | "sidepanel"

/** Which variant is currently rendered. V1 default, V3 and V5 selectable. */
export type ChatComposerVariant = "v1" | "v3" | "v5"

/**
 * Canonical doc-attachment shape understood by both composers. Both
 * `sendMessage({ docs: [...] })` calls forward this verbatim to parent
 * `onSubmit` handlers (see Phase-0 spike 6 for provenance).
 */
export interface ChatComposerDoc {
  type: "tab"
  tabId: string
  title: string
  url: string
  favIconUrl?: string
}

/**
 * Superset submit payload — the shared hook accepts this and the per-surface
 * adapter (see `useComposerSubmit(surface)`) strips fields the parent
 * `onSubmit` doesn't understand.
 *
 * Shared core: image, message, docs, imageBackendOverride.
 * Sidepanel-only: uploadedFiles, requestOverrides (queued-message replay).
 * Playground-only: userMessageType, assistantMessageType,
 * imageGenerationSource, researchContext.
 */
export interface ChatComposerSubmitPayload {
  // Shared core — both surfaces accept
  image: string
  message: string
  docs: ChatComposerDoc[]
  imageBackendOverride?: unknown

  // Sidepanel-only
  uploadedFiles?: unknown[]
  requestOverrides?: {
    chatMode?: string
    selectedModel?: string | null
    selectedSystemPrompt?: string | null
    toolChoice?: "auto" | "required" | "none" | null
    useOCR?: boolean
    webSearch?: boolean
  }

  // Playground-only
  userMessageType?: string
  assistantMessageType?: string
  imageGenerationSource?: string
  researchContext?: unknown
}

/**
 * What a variant component gets from the hook family. Grows over Phase 2 as
 * concerns are extracted. Variant components should subscribe only to the
 * slices they render.
 */
export interface ChatComposerContext {
  /** Which surface we're rendering for. */
  surface: ChatComposerSurface

  /** Variant chosen by the user's preference. */
  variant: ChatComposerVariant

  // Additional slices land here as hook extraction continues.
}
