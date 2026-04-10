import React, { startTransition } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { trackRouteAliasRedirect } from '@/utils/route-alias-telemetry';

type RouteRedirectProps = {
  to: string;
  title?: string;
  description?: string;
  preserveParams?: boolean;
};

export const resolveRedirectTarget = (
  currentPath: string,
  to: string,
  preserveParams: boolean
): string => {
  if (!preserveParams) return to;
  if (to.includes('?') || to.includes('#')) return to;

  const queryIndex = currentPath.indexOf('?');
  const hashIndex = currentPath.indexOf('#');
  const suffixStartCandidates = [queryIndex, hashIndex].filter((index) => index >= 0);

  if (suffixStartCandidates.length === 0) return to;

  const suffixStart = Math.min(...suffixStartCandidates);
  const suffix = currentPath.slice(suffixStart);
  return suffix ? `${to}${suffix}` : to;
};

const toPathLabel = (path: string): string => {
  const input = String(path || '').trim();
  if (!input) return '/';

  try {
    const parsed = new URL(input);
    const combined = `${parsed.pathname}${parsed.search}${parsed.hash}`;
    return combined || '/';
  } catch {
    return input.startsWith('/') ? input : `/${input}`;
  }
};

export const RouteRedirect: React.FC<RouteRedirectProps> = ({
  to,
  title = 'This route has moved',
  description = 'We are sending you to the updated page.',
  preserveParams = true,
}) => {
  const router = useRouter();
  const redirectedRef = React.useRef(false);

  const destination = React.useMemo(
    () => resolveRedirectTarget(router.asPath || '', to, preserveParams),
    [preserveParams, router.asPath, to]
  );
  const sourcePath = React.useMemo(
    () => toPathLabel(router.asPath || router.pathname || ''),
    [router.asPath, router.pathname]
  );

  React.useEffect(() => {
    if (redirectedRef.current) return;
    redirectedRef.current = true;
    let cancelled = false;

    const redirect = async () => {
      try {
        void trackRouteAliasRedirect({
          sourcePath: router.asPath || router.pathname || '',
          destinationPath: destination,
          preserveParams,
        });
      } catch {
        // Keep the redirect moving even if telemetry setup throws synchronously.
      }

      if (typeof router.prefetch === 'function') {
        try {
          void router.prefetch(destination).catch(() => {
            // Keep the redirect moving even when prefetch is unavailable.
          });
        } catch {
          // Keep the redirect moving even when prefetch is unavailable.
        }
      }

      if (cancelled) return;

      startTransition(() => {
        void router.replace(destination);
      });
    };

    void redirect();

    return () => {
      cancelled = true;
    };
  }, [destination, preserveParams, router]);

  return (
    <div className="flex min-h-[60vh] w-full items-center justify-center px-6">
      <div
        className="w-full max-w-lg rounded-xl border border-border bg-surface p-6 text-center shadow-sm"
        role="status"
        aria-live="polite"
        data-testid="route-redirect-panel"
      >
        <div className="mx-auto mb-4 h-7 w-7 animate-spin rounded-full border-2 border-border border-t-primary" />
        <h1 className="text-lg font-semibold text-text">{title}</h1>
        <p className="mt-2 text-sm text-text-muted">{description}</p>
        <p className="mt-2 text-xs text-text-muted">
          Redirecting from <code className="rounded bg-surface2 px-1 py-0.5">{sourcePath}</code> to{' '}
          <code className="rounded bg-surface2 px-1 py-0.5">{toPathLabel(destination)}</code>.
        </p>
        <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
          <Link
            href={destination}
            className="inline-flex items-center rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong"
            data-testid="route-redirect-open-updated-page"
          >
            Open updated page
          </Link>
          <Link
            href="/"
            className="inline-flex items-center rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
            data-testid="route-redirect-go-chat"
          >
            Go to Chat
          </Link>
          <Link
            href="/settings"
            className="inline-flex items-center rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
            data-testid="route-redirect-open-settings"
          >
            Open Settings
          </Link>
        </div>
      </div>
    </div>
  );
};

export default RouteRedirect;
