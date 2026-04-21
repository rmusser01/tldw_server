import React, { Suspense } from "react"
import type { TerminalStackV1Props } from "./variants/TerminalStackV1"
import type { SplitBriefV3Props } from "./variants/SplitBriefV3"
import type { RadialCommandV5Props } from "./variants/RadialCommandV5"
import type { ChatComposerVariant } from "./types"

/**
 * Top-level chat composer dispatcher. Renders one of V1 / V3 / V5 based on
 * the `variant` prop. Surfaces (Playground, Sidepanel) read their variant
 * preference via `useComposerVariantPreference()` and pass it along with
 * the variant-specific props bag.
 *
 * Using a discriminated union so TypeScript enforces the correct props
 * shape at each call site — you can't pass V3's `briefSections` when
 * `variant` is `"v1"`. Callers that want fallback behavior should branch
 * on variant themselves before constructing the props object.
 *
 * Variants are loaded lazily via `React.lazy` (bundler-agnostic; works
 * for both the Next.js web app and the WXT extension build, unlike
 * `next/dynamic`). Only the chosen variant's chunk loads on entry; an
 * in-session switch triggers a second async import with `null` fallback
 * (the slot content, including draft text, lives in the hook layer
 * outside the variant — once the new shell loads, the existing draft
 * re-attaches to the variant's `textareaSlot` and survives the swap).
 *
 * A chunk-load failure (network drop mid-import) bubbles up through
 * Suspense as a render error. We catch it in a local error boundary
 * and show an inline message rather than crashing the whole composer
 * tree — the parent surface's other UI (chat history, settings panels)
 * stays interactive.
 */

interface VariantErrorBoundaryState {
  failed: boolean
}

class VariantErrorBoundary extends React.Component<
  { children: React.ReactNode; variant: ChatComposerVariant },
  VariantErrorBoundaryState
> {
  state: VariantErrorBoundaryState = { failed: false }

  static getDerivedStateFromError(): VariantErrorBoundaryState {
    return { failed: true }
  }

  componentDidUpdate(prevProps: { variant: ChatComposerVariant }) {
    // Reset when the user picks a different variant — a fresh chunk
    // import gets a fresh chance to succeed.
    if (prevProps.variant !== this.props.variant && this.state.failed) {
      this.setState({ failed: false })
    }
  }

  render() {
    if (!this.state.failed) return this.props.children
    return (
      <div
        role="alert"
        data-testid="composer-variant-load-error"
        className="rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn"
      >
        Couldn't load the {this.props.variant.toUpperCase()} composer
        layout. Refresh the page to try again, or pick a different style
        in Settings.
      </div>
    )
  }
}

export type ChatComposerProps =
  | ({ variant: "v1" } & TerminalStackV1Props)
  | ({ variant: "v3" } & SplitBriefV3Props)
  | ({ variant: "v5" } & RadialCommandV5Props)

const TerminalStackV1Lazy = React.lazy(() =>
  import("./variants/TerminalStackV1").then((m) => ({
    default: m.TerminalStackV1,
  }))
)
const SplitBriefV3Lazy = React.lazy(() =>
  import("./variants/SplitBriefV3").then((m) => ({
    default: m.SplitBriefV3,
  }))
)
const RadialCommandV5Lazy = React.lazy(() =>
  import("./variants/RadialCommandV5").then((m) => ({
    default: m.RadialCommandV5,
  }))
)

/**
 * Minimal skeleton while a variant's chunk loads. Reserves roughly the
 * composer's height so content doesn't jump in when the shell hydrates,
 * and provides an `aria-live="polite"` status for screen-reader users.
 */
const VariantLoadingSkeleton: React.FC = () => (
  <div
    className="rounded-lg border border-border/60 bg-surface/30 p-3 animate-pulse"
    role="status"
    aria-live="polite"
    data-testid="composer-variant-loading"
  >
    <div className="h-10 rounded bg-surface2/50" />
    <span className="sr-only">Loading composer…</span>
  </div>
)

export const ChatComposer: React.FC<ChatComposerProps> = (props) => {
  switch (props.variant) {
    case "v1": {
      const { variant: _variant, ...v1Props } = props
      return (
        <VariantErrorBoundary variant="v1">
          <Suspense fallback={<VariantLoadingSkeleton />}>
            <TerminalStackV1Lazy {...v1Props} />
          </Suspense>
        </VariantErrorBoundary>
      )
    }
    case "v3": {
      const { variant: _variant, ...v3Props } = props
      return (
        <VariantErrorBoundary variant="v3">
          <Suspense fallback={<VariantLoadingSkeleton />}>
            <SplitBriefV3Lazy {...v3Props} />
          </Suspense>
        </VariantErrorBoundary>
      )
    }
    case "v5": {
      const { variant: _variant, ...v5Props } = props
      return (
        <VariantErrorBoundary variant="v5">
          <Suspense fallback={<VariantLoadingSkeleton />}>
            <RadialCommandV5Lazy {...v5Props} />
          </Suspense>
        </VariantErrorBoundary>
      )
    }
    default: {
      const _exhaustive: never = props
      return null
    }
  }
}

/** Re-export helpers consumers will often pair with `<ChatComposer>`. */
export {
  useComposerVariantPreference,
  COMPOSER_VARIANT_PREFERENCE_KEY,
} from "./hooks/useComposerVariantPreference"
export type { ChatComposerVariant }
