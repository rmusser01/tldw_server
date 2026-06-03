import React from "react"
import type { DynamicUISurface } from "@/types/dynamic-ui"
import { DynamicUISourceFallback } from "./DynamicUISourceFallback"

export class DynamicUIErrorBoundary extends React.Component<
  { source: string; surface?: DynamicUISurface; children: React.ReactNode },
  { error: string | null }
> {
  state = { error: null }

  static getDerivedStateFromError(error: unknown) {
    return {
      error:
        error instanceof Error ? error.message : "Dynamic UI render failed."
    }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("[DynamicUI] render error", error, errorInfo, {
      source: this.props.source
    })
  }

  componentDidUpdate(prevProps: { source: string }) {
    if (prevProps.source !== this.props.source && this.state.error) {
      this.setState({ error: null })
    }
  }

  render() {
    if (this.state.error) {
      return (
        <DynamicUISourceFallback
          source={this.props.source}
          error={this.state.error}
          surface={this.props.surface}
        />
      )
    }
    return this.props.children
  }
}
