import React from "react"
import { AlertTriangle } from "lucide-react"

interface ChatErrorBoundaryProps extends React.PropsWithChildren {
  onRetry?: () => void;
  fallback?: React.ReactNode;
}

interface ChatErrorBoundaryState {
  hasError: boolean
  error: Error | null
  resetKey: number
}

export class ChatErrorBoundary extends React.Component<
  ChatErrorBoundaryProps,
  ChatErrorBoundaryState
> {
  state: ChatErrorBoundaryState = {
    hasError: false,
    error: null,
    resetKey: 0
  }

  static getDerivedStateFromError(error: Error): Partial<ChatErrorBoundaryState> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error("[ChatErrorBoundary]", error, errorInfo.componentStack)
  }

  handleRetry = (): void => {
    this.setState((prev) => ({
      hasError: false,
      error: null,
      resetKey: prev.resetKey + 1
    }))
    this.props.onRetry?.()
  }

  render(): React.ReactNode {
    if (!this.state.hasError) {
      return <React.Fragment key={this.state.resetKey}>{this.props.children}</React.Fragment>
    }

    if (this.props.fallback) return this.props.fallback;

    return (
      <div
        role="alert"
        className="flex flex-col items-center justify-center gap-3 py-16 text-center"
      >
        <AlertTriangle className="h-8 w-8 text-warn" aria-hidden="true" />
        <p className="text-sm font-medium text-text">
          Something went wrong displaying the chat.
        </p>
        {process.env.NODE_ENV !== "production" && this.state.error && (
          <pre className="max-w-md whitespace-pre-wrap rounded-md bg-surface2 p-3 text-left text-xs text-danger">
            {this.state.error.message}
          </pre>
        )}
        <button
          type="button"
          className="mt-2 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          onClick={this.handleRetry}
        >
          Try again
        </button>
      </div>
    )
  }
}
