'use client';

import React, { Component, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { AlertTriangle, RefreshCw, Home, ChevronDown, ChevronUp } from 'lucide-react';
import { logger } from '@/lib/logger';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  onGoHome?: () => void;
  onReload?: () => void;
  maxRetries?: number;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  showDetails: boolean;
  retryCount: number;
  maxRetries: number;
  retriesExhausted: boolean;
}

class ErrorBoundaryBase extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  private static instanceCounter = 0;
  private detailsId: string;

  constructor(props: ErrorBoundaryProps) {
    super(props);
    ErrorBoundaryBase.instanceCounter += 1;
    this.detailsId = `error-boundary-details-${ErrorBoundaryBase.instanceCounter}`;
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      retryCount: 0,
      maxRetries: props.maxRetries ?? 3,
      retriesExhausted: false,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({ errorInfo });

    // Log error
    logger.error('Error Boundary caught an error', { component: 'ErrorBoundary', error: error.message, componentStack: errorInfo?.componentStack ?? undefined });

    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  handleReset = () => {
    this.setState((prev) => {
      const nextCount = prev.retryCount + 1;
      if (nextCount >= prev.maxRetries) {
        return {
          ...prev,
          hasError: true,
          retryCount: nextCount,
          retriesExhausted: true,
        };
      }
      return {
        hasError: false,
        error: null,
        errorInfo: null,
        showDetails: false,
        retryCount: nextCount,
        maxRetries: prev.maxRetries,
        retriesExhausted: false,
      };
    });
  };

  handleReload = () => {
    if (this.props.onReload) {
      this.props.onReload();
      return;
    }
    this.handleReset();
  };

  handleGoHome = () => {
    if (this.props.onGoHome) {
      this.props.onGoHome();
      return;
    }
    this.handleReset();
  };

  toggleDetails = () => {
    this.setState((prev) => ({ showDetails: !prev.showDetails }));
  };

  render() {
    if (this.state.hasError) {
      const retriesExhausted =
        this.state.retriesExhausted || this.state.retryCount >= this.state.maxRetries;
      // Custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <div className="min-h-screen flex items-center justify-center bg-background p-4">
          <Card className="max-w-lg w-full">
            <CardHeader className="text-center">
              <div className="mx-auto mb-4 h-12 w-12 rounded-full bg-destructive/10 flex items-center justify-center">
                <AlertTriangle className="h-6 w-6 text-destructive" aria-hidden="true" />
              </div>
              <CardTitle>Something went wrong</CardTitle>
              <CardDescription>
                {retriesExhausted
                  ? 'Retry limit reached. Please reload the page or return to the dashboard.'
                  : 'An unexpected error occurred. Don&apos;t worry, your data is safe.'}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Error message */}
              <div className="rounded-lg bg-muted p-3">
                <p className="text-sm font-medium text-destructive">
                  {this.state.error?.message || 'Unknown error'}
                </p>
              </div>

              {/* Action buttons */}
              <div className="flex flex-col sm:flex-row gap-2">
                <Button onClick={this.handleReset} className="flex-1" disabled={retriesExhausted}>
                  <RefreshCw className="mr-2 h-4 w-4" aria-hidden="true" />
                  {retriesExhausted ? 'Retry limit reached' : 'Try Again'}
                </Button>
                <Button variant="outline" onClick={this.handleGoHome} className="flex-1">
                  <Home className="mr-2 h-4 w-4" aria-hidden="true" />
                  Go to Dashboard
                </Button>
              </div>

              {/* Technical details toggle */}
              <button
                onClick={this.toggleDetails}
                aria-expanded={this.state.showDetails}
                aria-controls={this.detailsId}
                aria-label="Toggle technical details"
                className="flex items-center justify-center w-full text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                {this.state.showDetails ? (
                  <>
                    <ChevronUp className="mr-1 h-4 w-4" aria-hidden="true" />
                    Hide technical details
                  </>
                ) : (
                  <>
                    <ChevronDown className="mr-1 h-4 w-4" aria-hidden="true" />
                    Show technical details
                  </>
                )}
              </button>

              {/* Technical details */}
              {this.state.showDetails && (
                <div
                  id={this.detailsId}
                  role="region"
                  aria-label="Technical details"
                  className="rounded-lg bg-muted p-3 overflow-auto max-h-48"
                >
                  <p className="text-xs font-mono text-muted-foreground whitespace-pre-wrap">
                    {this.state.error?.stack || 'No stack trace available'}
                  </p>
                  {this.state.errorInfo && (
                    <div className="mt-2 pt-2 border-t border-border">
                      <p className="text-xs font-mono text-muted-foreground whitespace-pre-wrap">
                        Component Stack:
                        {this.state.errorInfo.componentStack}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Help text */}
              <p className="text-xs text-center text-muted-foreground">
                If this problem persists, please contact your administrator.
              </p>
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

function ErrorBoundary(props: ErrorBoundaryProps) {
  const router = useRouter();
  const { onGoHome, onReload, ...rest } = props;
  const handleGoHome = onGoHome ?? (() => router.push('/'));
  const handleReload = onReload ?? (() => router.refresh());

  return (
    <ErrorBoundaryBase
      {...rest}
      onGoHome={handleGoHome}
      onReload={handleReload}
    />
  );
}

type ErrorHandlerProps = Pick<
  ErrorBoundaryProps,
  'onError' | 'onGoHome' | 'onReload' | 'maxRetries'
>;

export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options?: { fallback?: ReactNode } & ErrorHandlerProps
) {
  function WithErrorBoundary(props: P) {
    return (
      <ErrorBoundary {...options}>
        <WrappedComponent {...props} />
      </ErrorBoundary>
    );
  }

  const wrappedName = WrappedComponent.displayName || WrappedComponent.name || 'Component';
  WithErrorBoundary.displayName = `withErrorBoundary(${wrappedName})`;

  return WithErrorBoundary;
}

// Page-level error boundary with simpler UI
export function PageErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      onError={(error) => {
        // Could send to error reporting service here
        logger.error('Page error', { component: 'PageErrorBoundary', error: error.message });
      }}
    >
      {children}
    </ErrorBoundary>
  );
}

// Card-level error boundary for smaller sections
interface CardErrorBoundaryProps {
  children: ReactNode;
  title?: string;
  onReload?: () => void;
}

export function CardErrorBoundary({
  children,
  title = 'This section',
  onReload,
}: CardErrorBoundaryProps) {
  const router = useRouter();
  const handleReload = onReload ?? (() => router.refresh());
  return (
    <ErrorBoundary
      onReload={handleReload}
      onError={(error) => {
        logger.error('Card section error', { component: 'CardErrorBoundary', error: error.message });
      }}
      fallback={
        <Card className="border-destructive/50">
          <CardContent className="py-8 text-center">
            <AlertTriangle className="h-8 w-8 text-destructive mx-auto mb-2" aria-hidden="true" />
            <p className="text-sm text-muted-foreground">
              {title} failed to load.
            </p>
            <Button
              variant="outline"
              size="sm"
              className="mt-3"
              onClick={handleReload}
            >
              <RefreshCw className="mr-2 h-3 w-3" aria-hidden="true" />
              Reload
            </Button>
          </CardContent>
        </Card>
      }
    >
      {children}
    </ErrorBoundary>
  );
}

export default ErrorBoundary;
