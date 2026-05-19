import Link from 'next/link';
import { startTransition } from 'react';
import { useRouter } from 'next/router';

export default function NotFoundPage() {
  const router = useRouter();
  const routeLabel = String(router.asPath || '/');

  return (
    <div className="flex min-h-[70vh] w-full items-center justify-center px-6 py-12">
      <div
        className="w-full max-w-xl rounded-xl border border-border bg-surface p-8 shadow-sm"
        data-testid="not-found-recovery-panel"
      >
        <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">404</p>
        <h1 className="mt-2 text-2xl font-semibold text-text">We could not find that route</h1>
        <p className="mt-3 text-sm text-text-muted">
          The link may be out of date, moved, or typed incorrectly. Choose a route below to
          continue.
        </p>
        <p className="mt-2 text-xs text-text-muted">
          Route not found: <code className="rounded bg-surface2 px-1 py-0.5">{routeLabel}</code>
        </p>

        <div className="mt-6 flex flex-wrap gap-2">
          <button
            type="button"
            className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong"
            onClick={() => {
              startTransition(() => {
                void router.push('/');
              });
            }}
            data-testid="not-found-open-home"
          >
            Open Home
          </button>
          <Link
            href="/research"
            className="rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
            data-testid="not-found-open-research"
          >
            Open Research
          </Link>
          <Link
            href="/media"
            className="rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
            data-testid="not-found-open-media"
          >
            Open Media
          </Link>
          <Link
            href="/settings"
            className="rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
            data-testid="not-found-open-settings"
          >
            Open Settings
          </Link>
          <button
            type="button"
            className="rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
            onClick={() => {
              startTransition(() => {
                router.back();
              });
            }}
            data-testid="not-found-go-back"
          >
            Go back
          </button>
        </div>
      </div>
    </div>
  );
}
