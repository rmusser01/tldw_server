import React from "react"
import { useTranslation } from "react-i18next"
import { AlertTriangle, RefreshCw } from "lucide-react"
import { EmptyState } from "@/components/ui/feedback/EmptyState"

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

interface ErrorBoundaryProps {
  children: React.ReactNode
  fallback?: React.ReactNode
}

/**
 * Error Boundary for the Document Workspace
 *
 * Catches JavaScript errors in child components and displays
 * a fallback UI instead of crashing the entire page.
 */
export class DocumentWorkspaceErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    // Log the error for debugging
    console.error("DocumentWorkspace Error:", error)
    console.error("Component Stack:", errorInfo.componentStack)
  }

  handleRetry = (): void => {
    this.setState({ hasError: false, error: null })
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return <ErrorFallback onRetry={this.handleRetry} error={this.state.error} />
    }

    return this.props.children
  }
}

interface ErrorFallbackProps {
  onRetry: () => void
  error: Error | null
}

/**
 * Default fallback UI shown when an error occurs
 */
const ErrorFallback: React.FC<ErrorFallbackProps> = ({ onRetry, error }) => {
  const { t } = useTranslation(["option", "common"])

  return (
    <div className="flex h-full items-center justify-center bg-bg p-8">
      <EmptyState
        variant="card"
        size="lg"
        icon={AlertTriangle}
        iconClassName="text-warning"
        title={t(
          "option:documentWorkspace.errorTitle",
          "Something went wrong"
        )}
        description={
          <div className="space-y-2">
            <p className="text-text-secondary">
              {t(
                "option:documentWorkspace.errorDescription",
                "An error occurred while loading the document workspace."
              )}
            </p>
            {error && process.env.NODE_ENV === "development" && (
              <details className="mt-4 text-left">
                <summary className="cursor-pointer text-sm text-muted hover:text-text">
                  {t("common:showDetails", "Show details")}
                </summary>
                <pre className="mt-2 overflow-auto rounded bg-surface p-3 text-xs text-error">
                  {error.message}
                  {error.stack && (
                    <>
                      {"\n\n"}
                      {error.stack}
                    </>
                  )}
                </pre>
              </details>
            )}
          </div>
        }
        primaryAction={{
          label: t("common:tryAgain", "Try again"),
          icon: <RefreshCw className="h-4 w-4" />,
          onClick: onRetry
        }}
      />
    </div>
  )
}

export default DocumentWorkspaceErrorBoundary
