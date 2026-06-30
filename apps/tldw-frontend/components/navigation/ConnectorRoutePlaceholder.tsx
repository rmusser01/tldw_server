import React from 'react';
import Link from 'next/link';
import {
  getOperationsRouteJob,
  type OperationsRouteJob,
} from '@tldw/ui/routes/operations-route-jobs';

type ConnectorRoute = '/connectors' | '/connectors/browse' | '/connectors/jobs' | '/connectors/sources';

type AlternativeLink = {
  href: string;
  label: string;
  description: string;
};

type ConnectorPlaceholderCopy = {
  route: ConnectorRoute;
  unavailableMessage: string;
  primaryHref: string;
  primaryLabel: string;
  alternatives: AlternativeLink[];
};

type ConnectorRoutePlaceholderProps = {
  route: string;
};

const CONNECTOR_COPY: Record<ConnectorRoute, ConnectorPlaceholderCopy> = {
  '/connectors': {
    route: '/connectors',
    unavailableMessage:
      'Connector management is not active in this build. Use the supported routes below for current setup and automation work.',
    primaryHref: '/settings',
    primaryLabel: 'Open Settings',
    alternatives: [
      {
        href: '/integrations',
        label: 'Integrations',
        description: 'Manage personal Slack and Discord connections.',
      },
      {
        href: '/sources',
        label: 'Sources',
        description: 'Manage ingestion sources and sync status.',
      },
      {
        href: '/scheduled-tasks',
        label: 'Scheduled Tasks',
        description: 'Manage reminder tasks and endpoint availability.',
      },
    ],
  },
  '/connectors/browse': {
    route: '/connectors/browse',
    unavailableMessage:
      'Connector catalog browsing is not active in this build. Use integrations and source setup for currently supported connection work.',
    primaryHref: '/integrations',
    primaryLabel: 'Open Integrations',
    alternatives: [
      {
        href: '/sources',
        label: 'Sources',
        description: 'Set up source ingestion where the server supports it.',
      },
      {
        href: '/settings',
        label: 'Settings',
        description: 'Review server connection and configuration options.',
      },
    ],
  },
  '/connectors/jobs': {
    route: '/connectors/jobs',
    unavailableMessage:
      'Connector job orchestration is not active in this build. Use scheduled tasks and watchlists for currently supported automation.',
    primaryHref: '/scheduled-tasks',
    primaryLabel: 'Open Scheduled Tasks',
    alternatives: [
      {
        href: '/watchlists',
        label: 'Watchlists',
        description: 'Operate feed monitoring, runs, and reports.',
      },
      {
        href: '/settings',
        label: 'Settings',
        description: 'Review server connection and configuration options.',
      },
    ],
  },
  '/connectors/sources': {
    route: '/connectors/sources',
    unavailableMessage:
      'Source-specific connector workflows are not active in this build. Use Sources for current ingestion-source setup.',
    primaryHref: '/sources',
    primaryLabel: 'Open Sources',
    alternatives: [
      {
        href: '/integrations',
        label: 'Integrations',
        description: 'Manage personal Slack and Discord connections.',
      },
      {
        href: '/settings',
        label: 'Settings',
        description: 'Review server connection and configuration options.',
      },
    ],
  },
};

const normalizeConnectorRoute = (route: string): ConnectorRoute => {
  const [pathname] = route.split('?');
  const normalized = pathname.endsWith('/') && pathname !== '/' ? pathname.slice(0, -1) : pathname;

  if (
    normalized === '/connectors/browse' ||
    normalized === '/connectors/jobs' ||
    normalized === '/connectors/sources'
  ) {
    return normalized;
  }

  return '/connectors';
};

const resolveConnectorJob = (route: ConnectorRoute): OperationsRouteJob | undefined =>
  getOperationsRouteJob(route);

export const ConnectorRoutePlaceholder: React.FC<ConnectorRoutePlaceholderProps> = ({
  route,
}) => {
  const connectorRoute = normalizeConnectorRoute(route);
  const copy = CONNECTOR_COPY[connectorRoute];
  const job = resolveConnectorJob(connectorRoute);
  const label = job?.label ?? 'Connectors';
  const requestedRoute = route || connectorRoute;

  return (
    <main className="flex min-h-[70vh] w-full items-center justify-center px-6 py-12">
      <section
        aria-labelledby="connector-placeholder-title"
        className="w-full max-w-2xl rounded-xl border border-border bg-surface p-8 shadow-sm"
        data-placeholder-kind="connector"
        data-testid="route-placeholder-panel"
      >
        <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
          Connector placeholder
        </p>
        <h1 id="connector-placeholder-title" className="mt-2 text-2xl font-semibold text-text">
          {label}
        </h1>
        <p className="mt-3 text-sm text-text-muted">{copy.unavailableMessage}</p>

        {job ? (
          <dl className="mt-5 grid gap-3 rounded-lg border border-border bg-surface2 p-4 text-sm sm:grid-cols-2">
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                Route state
              </dt>
              <dd className="mt-1 font-medium text-text">Placeholder</dd>
            </div>
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-text-muted">
                Current job
              </dt>
              <dd className="mt-1 text-text">{job.primaryJob}</dd>
            </div>
          </dl>
        ) : null}

        <p className="mt-4 text-xs text-text-muted">
          Requested route:{' '}
          <code className="rounded bg-surface2 px-1 py-0.5">{requestedRoute}</code>
        </p>

        <div className="mt-6">
          <Link
            href={copy.primaryHref}
            className="inline-flex rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong"
            data-testid="connector-placeholder-primary"
          >
            {copy.primaryLabel}
          </Link>
        </div>

        <div className="mt-6" data-testid="connector-placeholder-alternatives">
          <h2 className="text-sm font-semibold text-text">Supported alternatives</h2>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            {copy.alternatives.map((alternative) => (
              <Link
                key={alternative.href}
                href={alternative.href}
                className="rounded-md border border-border px-3 py-2 text-sm text-text hover:bg-surface2"
              >
                <span className="block font-medium">{alternative.label}</span>
                <span className="mt-1 block text-xs text-text-muted">
                  {alternative.description}
                </span>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
};

export default ConnectorRoutePlaceholder;
