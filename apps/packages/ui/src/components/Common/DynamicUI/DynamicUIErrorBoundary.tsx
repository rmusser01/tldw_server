import React from "react"
import { DynamicUISourceFallback } from "./DynamicUISourceFallback"

export class DynamicUIErrorBoundary extends React.Component<
  { source: string; children: React.ReactNode },
  { error: string | null }
> {
  state = { error: null }

  static getDerivedStateFromError(error: unknown) {
    return {
      error:
        error instanceof Error ? error.message : "Dynamic UI render failed."
    }
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
        />
      )
    }
    return this.props.children
  }
}
