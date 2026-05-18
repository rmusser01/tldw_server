import React, { startTransition } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';

export type RoutePlaceholderProps = {
  title: string;
  description: string;
  plannedPath?: string;
  primaryCtaHref?: string;
  primaryCtaLabel?: string;
};

export const RoutePlaceholder: React.FC<RoutePlaceholderProps> = ({
  title,
  description,
  plannedPath,
  primaryCtaHref = '/',
  primaryCtaLabel = 'Go to Chat',
}) => {
  const router = useRouter();
  const routeLabel = String(router.asPath || '/');
  const showSettingsShortcut = !primaryCtaHref.startsWith('/settings');

  return (
    <div className="flex min-h-[70vh] w-full items-center justify-center px-6 py-12">
      <div
        className="w-full max-w-xl rounded-xl border border-border bg-surface p-8 shadow-sm"
        data-testid="route-placeholder-panel"
      >
        <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
          Coming Soon
        </p>
        <h1 className="mt-2 text-2xl font-semibold text-text">{title}</h1>
        <p className="mt-3 text-sm text-text-muted">{description}</p>
        <p className="mt-2 text-xs text-text-muted">
          Requested route:{' '}
          <code className="rounded bg-surface2 px-1 py-0.5">{routeLabel}</code>
        </p>
        {plannedPath ? (
          <p className="mt-1 text-xs text-text-muted">
            Planned route:{' '}
            <code className="rounded bg-surface2 px-1 py-0.5">
              {plannedPath}
            </code>
          </p>
        ) : null}

        <div className="mt-6 flex flex-wrap gap-2">
          <Link
            href={primaryCtaHref}
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong"
            data-testid="route-placeholder-primary"
          >
            {primaryCtaLabel}
          </Link>
          {showSettingsShortcut ? (
            <Link
              href="/settings"
              className="rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
              data-testid="route-placeholder-open-settings"
            >
              Open Settings
            </Link>
          ) : null}
          <button
            type="button"
            className="rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
            onClick={() => {
              startTransition(() => {
                router.back();
              });
            }}
            data-testid="route-placeholder-go-back"
          >
            Go back
          </button>
        </div>
      </div>
    </div>
  );
};

export default RoutePlaceholder;
