'use client';

import { Fragment } from 'react';
import {
  AlertTriangle,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Copy,
  Edit3,
  History,
  KeyRound,
  Link2,
  Play,
  Plus,
  RefreshCw,
  RotateCw,
  Send,
  Trash2,
  Webhook,
} from 'lucide-react';

import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { EmptyState } from '@/components/ui/empty-state';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { formatDateTime } from '@/lib/format';
import type {
  WebhookDelivery,
  WebhookDeliveryComponent,
  WebhookStatus,
} from '@/types';
import {
  activationBlockReason,
  useWebhooksPageController,
  WEBHOOK_PAGE_SIZE,
} from './use-webhooks-page-controller';

const componentLabel = (component: WebhookDeliveryComponent): string => (
  `${component.component[0]?.toUpperCase()}${component.component.slice(1)} ${
    component.ready ? 'ready' : component.reason_code?.replaceAll('_', ' ') ?? 'unavailable'
  }`
);

const terminalDelivery = (delivery: WebhookDelivery): boolean => (
  ['succeeded', 'dead', 'canceled', 'superseded'].includes(delivery.state)
);

const httpStatusLabel = (statusCode: number | null): string => (
  statusCode === null ? 'No HTTP status' : `HTTP ${statusCode} (${Math.floor(statusCode / 100)}xx)`
);

function RuntimeStatus({ status }: { status: WebhookStatus }) {
  const delivery = status.delivery;
  const backlog = Object.values(delivery.backlog).reduce((sum, count) => sum + count, 0);
  return (
    <section
      className="grid gap-3 border-y py-3 text-sm md:grid-cols-[minmax(0,1fr)_auto] md:items-center"
      aria-label="Webhook delivery runtime"
    >
      <div className="flex flex-wrap gap-2">
        <Badge
          variant={delivery.key_ready && delivery.key_primary_match ? 'outline' : 'destructive'}
        >
          {delivery.key_ready && delivery.key_primary_match
            ? 'Signing key ready'
            : delivery.key_ready
              ? 'Signing key primary mismatch'
              : 'Signing key unavailable'}
        </Badge>
        {[delivery.worker, delivery.reconciler, delivery.retention].map((component) => (
          <Badge key={component.component} variant={component.ready ? 'outline' : 'destructive'}>
            {componentLabel(component)}
            {component.heartbeat_age_seconds !== null && ` (${component.heartbeat_age_seconds}s)`}
          </Badge>
        ))}
        <Badge variant={delivery.acquisition_ready ? 'outline' : 'destructive'}>
          {delivery.acquisition_ready
            ? 'Acquisition ready'
            : `Acquisition ${delivery.acquisition_reason_code?.replaceAll('_', ' ') ?? 'blocked'}`}
        </Badge>
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted-foreground md:justify-end">
        <span>{backlog} nonterminal {backlog === 1 ? 'delivery' : 'deliveries'}</span>
        <span>
          {delivery.oldest_nonterminal_age_seconds === null
            ? 'No outstanding work'
            : `Oldest work ${delivery.oldest_nonterminal_age_seconds}s`}
        </span>
        <span>Jobs: {delivery.jobs_backend}</span>
      </div>
    </section>
  );
}

function StatusAlerts({ status }: { status: WebhookStatus }) {
  const required = status.migration.secret_rotation_required_count;
  return (
    <div className="space-y-3" aria-label="Webhook operational status">
      {status.mode === 'off' && (
        <Alert variant="destructive">
          <AlertDescription>
            The webhook control plane is off. Enable it in deployment configuration before managing registrations.
          </AlertDescription>
        </Alert>
      )}
      {status.mode === 'migrate' && (
        <Alert variant="destructive">
          <AlertDescription>
            Webhooks are in migration mode. Complete the predeploy migration and switch to on before managing registrations.
          </AlertDescription>
        </Alert>
      )}
      {status.migration.phase !== 'complete' && (
        <Alert variant="destructive">
          <AlertDescription>
            Webhook migration is not complete ({status.migration.phase}). Complete or resume migration before using canonical registrations.
          </AlertDescription>
        </Alert>
      )}
      {status.key_state !== 'available' && (
        <Alert variant="destructive">
          <AlertDescription>
            The webhook signing key is unavailable. Creation, rotation, and activation are blocked.
          </AlertDescription>
        </Alert>
      )}
      {status.limits.registrations_over_limit && (
        <Alert variant="destructive">
          <AlertDescription>
            The webhook registration limit is exceeded. Delete registrations or raise the configured limit.
          </AlertDescription>
        </Alert>
      )}
      {status.limits.active_registrations_over_limit && (
        <Alert variant="destructive">
          <AlertDescription>The active webhook limit is exceeded. New activations are blocked.</AlertDescription>
        </Alert>
      )}
      {!status.delivery_capability_ready && (
        <Alert>
          <AlertDescription>
            Webhook delivery capability is unavailable. Activation and delivery commands are disabled.
          </AlertDescription>
        </Alert>
      )}
      {required > 0 && (
        <Alert>
          <AlertDescription>
            {required} registration{required === 1 ? '' : 's'} require{required === 1 ? 's' : ''} a new signing secret before activation.
          </AlertDescription>
        </Alert>
      )}
      {status.migration.legacy_file_restore_permitted ? (
        status.migration.rollback_window_expires_at && (
          <Alert>
            <AlertDescription>
              Legacy restore remains available until {formatDateTime(status.migration.rollback_window_expires_at)}.
            </AlertDescription>
          </Alert>
        )
      ) : (
        <Alert>
          <AlertDescription>
            Legacy restore is unavailable. Continue with the forward-fix runbook for any registration issue.
          </AlertDescription>
        </Alert>
      )}
      {status.mode === 'on' && <RuntimeStatus status={status} />}
    </div>
  );
}

function WebhooksPageContent() {
  const controller = useWebhooksPageController();
  const {
    status,
    catalog,
    canonicalPage,
    offset,
    loading,
    statusError,
    ready,
    createOpen,
    createUrl,
    createUrlError,
    createDescription,
    createTimeout,
    createEvents,
    creating,
    editor,
    editDescription,
    editTimeout,
    editEvents,
    replacementUrl,
    replacementUrlError,
    mutatingId,
    conflict,
    secretState,
    secretCopied,
    secretAcknowledged,
    secretWarning,
    commandError,
    commandBusy,
    pendingOperation,
    sensitiveCommandLocked,
    expandedId,
    deliveryPage,
    deliveryLoading,
    deliveryError,
    testingId,
    testStatus,
    testRetryAvailable,
    redeliveringId,
    redeliveryStatus,
    redeliveryRetryAvailable,
    addDisabled,
    visibleTotal,
    visibleCount,
    hasPrevious,
    hasNext,
  } = controller;

  return (
    <ResponsiveLayout>
      <div className="space-y-6 p-4 lg:p-8">
        <header className="flex flex-col gap-4 border-b pb-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Webhooks</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Manage outgoing event registrations, delivery history, and signing secrets.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => void controller.loadControlPlane(offset)}
              loading={loading}
              loadingText="Refreshing"
            >
              <RefreshCw className="h-4 w-4" aria-hidden="true" />
              Refresh
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={controller.openCreate}
              disabled={addDisabled || sensitiveCommandLocked}
            >
              <Plus className="h-4 w-4" aria-hidden="true" />
              Add webhook
            </Button>
          </div>
        </header>

        {statusError && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" aria-hidden="true" />
            <AlertDescription className="space-y-3">
              <p>{statusError.message}</p>
              {statusError.requestId && (
                <p className="font-mono text-xs">Request ID: {statusError.requestId}</p>
              )}
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void controller.loadControlPlane(0)}
              >
                <RefreshCw className="h-4 w-4" aria-hidden="true" />
                Retry status
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {status && <StatusAlerts status={status} />}

        {conflict && (
          <Alert variant="destructive">
            <AlertDescription>
              <p>
                Review the current webhook before retrying {conflict.action}. The previous command was not retried automatically.
              </p>
              <p className="mt-2">
                Current revision {conflict.registration.revision}: {conflict.registration.description || 'No description'} at {conflict.registration.target_display}.
              </p>
            </AlertDescription>
          </Alert>
        )}

        {pendingOperation === 'rotate' && commandError && (
          <Alert variant="destructive">
            <AlertDescription className="space-y-3">
              <p>{commandError}</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  size="sm"
                  onClick={() => void controller.retrySecretCommand()}
                  loading={commandBusy}
                  loadingText="Retrying"
                >
                  <RotateCw className="h-4 w-4" aria-hidden="true" />
                  Retry same command
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    controller.clearSensitiveCommandState(false);
                    void controller.loadControlPlane(offset);
                  }}
                >
                  Reload registrations
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        )}

        {testStatus && (
          <Alert>
            <AlertDescription className="flex flex-wrap items-center justify-between gap-3">
              <span>{testStatus}</span>
              {testRetryAvailable && (
                <Button type="button" size="sm" onClick={() => void controller.retrySameTest()}>
                  <RotateCw className="h-4 w-4" aria-hidden="true" />
                  Retry same test
                </Button>
              )}
            </AlertDescription>
          </Alert>
        )}

        {redeliveryStatus && (
          <Alert>
            <AlertDescription className="flex flex-wrap items-center justify-between gap-3">
              <span>{redeliveryStatus}</span>
              {redeliveryRetryAvailable && (
                <Button type="button" size="sm" onClick={() => void controller.retrySameRedelivery()}>
                  <RotateCw className="h-4 w-4" aria-hidden="true" />
                  Retry same redelivery
                </Button>
              )}
            </AlertDescription>
          </Alert>
        )}

        {loading && visibleCount === 0 && !statusError ? (
          <div className="min-h-40 py-12 text-center text-sm text-muted-foreground" role="status">
            Loading webhooks...
          </div>
        ) : ready ? (
          canonicalPage.items.length === 0 ? (
            <EmptyState
              icon={Webhook}
              title="No webhooks configured"
              description="Create an inactive registration, store its signing secret, then enable it when delivery is ready."
            />
          ) : (
            <div className="overflow-x-auto border-y">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Destination</TableHead>
                    <TableHead>Description</TableHead>
                    <TableHead>Events</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Updated</TableHead>
                    <TableHead className="min-w-80 text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {canonicalPage.items.map((registration) => {
                    const activationReason = status
                      ? activationBlockReason(registration, status)
                      : 'Webhook status is unavailable';
                    const busy = mutatingId === registration.id;
                    const rowActionsDisabled = busy || sensitiveCommandLocked;
                    const rotationBlocked = registration.active
                      ? 'Disable the webhook before generating a new secret'
                      : status?.key_state !== 'available'
                        ? 'Webhook signing key is unavailable'
                        : null;
                    const expanded = expandedId === registration.id;
                    return (
                      <Fragment key={registration.id}>
                        <TableRow>
                          <TableCell>
                            <div className="max-w-64">
                              <p className="truncate font-mono text-sm" title={registration.target_display}>
                                {registration.target_display}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                ID {registration.id}, revision {registration.revision}
                              </p>
                            </div>
                          </TableCell>
                          <TableCell className="max-w-56">
                            <span className="line-clamp-2 text-sm">
                              {registration.description || 'No description'}
                            </span>
                          </TableCell>
                          <TableCell>
                            <div className="flex max-w-64 flex-wrap gap-1">
                              {registration.event_types.map((eventType) => (
                                <Badge key={eventType} variant="secondary" className="text-xs">
                                  {eventType}
                                </Badge>
                              ))}
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="space-y-1">
                              <Badge variant={registration.active ? 'default' : 'outline'}>
                                {registration.active ? 'Active' : 'Inactive'}
                              </Badge>
                              {registration.secret_rotation_required && (
                                <p className="text-xs text-amber-700">Secret rotation required</p>
                              )}
                            </div>
                          </TableCell>
                          <TableCell className="whitespace-nowrap text-sm">
                            {formatDateTime(registration.updated_at)}
                          </TableCell>
                          <TableCell>
                            <div className="flex min-h-9 flex-wrap justify-end gap-1">
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                className="h-9 w-9"
                                onClick={() => controller.openMetadataEditor(registration)}
                                disabled={rowActionsDisabled}
                                aria-label="Edit metadata"
                                title="Edit metadata"
                              >
                                <Edit3 className="h-4 w-4" aria-hidden="true" />
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                className="h-9 w-9"
                                onClick={() => controller.openDestinationEditor(registration)}
                                disabled={rowActionsDisabled}
                                aria-label="Replace destination"
                                title="Replace destination"
                              >
                                <Link2 className="h-4 w-4" aria-hidden="true" />
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => void controller.testCanonicalRegistration(registration)}
                                disabled={rowActionsDisabled || !status?.delivery_capability_ready}
                                loading={testingId === registration.id}
                                loadingText="Testing"
                              >
                                <Play className="h-4 w-4" aria-hidden="true" />
                                {testRetryAvailable ? 'Run new test' : 'Run test'}
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => void controller.rotateCanonicalSecret(registration)}
                                disabled={rowActionsDisabled || Boolean(rotationBlocked)}
                                title={rotationBlocked ?? undefined}
                              >
                                <KeyRound className="h-4 w-4" aria-hidden="true" />
                                Generate a new secret
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => void controller.toggleCanonicalRegistration(registration)}
                                disabled={rowActionsDisabled || Boolean(activationReason)}
                                title={activationReason ?? undefined}
                              >
                                {registration.active ? 'Disable' : 'Enable'}
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => void controller.toggleDeliveryHistory(registration)}
                                disabled={rowActionsDisabled}
                                aria-label={expanded ? 'Hide delivery history' : 'Show delivery history'}
                              >
                                <History className="h-4 w-4" aria-hidden="true" />
                                History
                                {expanded
                                  ? <ChevronUp className="h-4 w-4" aria-hidden="true" />
                                  : <ChevronDown className="h-4 w-4" aria-hidden="true" />}
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                className="h-9 w-9"
                                onClick={() => void controller.deleteCanonicalRegistration(registration)}
                                disabled={rowActionsDisabled}
                                aria-label="Delete webhook"
                                title="Delete webhook"
                              >
                                <Trash2 className="h-4 w-4 text-destructive" aria-hidden="true" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                        {expanded && (
                          <TableRow>
                            <TableCell colSpan={6} className="bg-muted/20 p-0">
                              <section className="min-h-32 space-y-3 px-4 py-4" aria-label="Delivery history">
                                <div className="flex items-center justify-between gap-3">
                                  <h2 className="text-sm font-semibold">Delivery history</h2>
                                  <span className="text-xs text-muted-foreground">
                                    Sanitized metadata only
                                  </span>
                                </div>
                                {deliveryLoading ? (
                                  <p className="text-sm text-muted-foreground" role="status">
                                    Loading delivery history...
                                  </p>
                                ) : deliveryError ? (
                                  <p className="text-sm text-destructive">{deliveryError}</p>
                                ) : deliveryPage.items.length === 0 ? (
                                  <p className="text-sm text-muted-foreground">No deliveries recorded.</p>
                                ) : (
                                  <ul className="divide-y">
                                    {deliveryPage.items.map(({ delivery, attempts }) => (
                                      <li key={delivery.id} className="space-y-2 py-3 first:pt-0">
                                        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
                                          <span className="font-mono">{delivery.event_type}</span>
                                          <Badge variant="secondary">{delivery.kind}</Badge>
                                          <Badge variant={delivery.state === 'succeeded' ? 'outline' : 'secondary'}>
                                            {delivery.state.replaceAll('_', ' ')}
                                          </Badge>
                                          <span>{httpStatusLabel(delivery.status_code)}</span>
                                          <span>{delivery.latency_ms === null ? 'No latency' : `${delivery.latency_ms}ms`}</span>
                                          {delivery.reason_code && (
                                            <span>{delivery.reason_code.replaceAll('_', ' ')}</span>
                                          )}
                                          <span className="text-muted-foreground">
                                            {formatDateTime(delivery.created_at)}
                                          </span>
                                          {terminalDelivery(delivery) && delivery.kind !== 'test' && (
                                            <Button
                                              type="button"
                                              variant="outline"
                                              size="sm"
                                              className="ml-auto"
                                              onClick={() => void controller.redeliverWebhook(delivery)}
                                              loading={redeliveringId === delivery.id}
                                              loadingText="Redelivering"
                                              aria-label={`Redeliver ${delivery.event_type}`}
                                            >
                                              <Send className="h-4 w-4" aria-hidden="true" />
                                              Redeliver
                                            </Button>
                                          )}
                                        </div>
                                        <dl className="grid gap-x-5 gap-y-1 text-xs text-muted-foreground sm:grid-cols-2 lg:grid-cols-4">
                                          <div>
                                            <dt className="inline font-medium text-foreground">Delivery ID: </dt>
                                            <dd className="inline break-all font-mono">{delivery.id}</dd>
                                          </div>
                                          <div>
                                            <dt className="inline font-medium text-foreground">Event ID: </dt>
                                            <dd className="inline break-all font-mono">{delivery.event_id}</dd>
                                          </div>
                                          <div>
                                            <dt className="inline font-medium text-foreground">Versions: </dt>
                                            <dd className="inline">
                                              Config v{delivery.delivery_config_version}, secret v{delivery.secret_version}
                                            </dd>
                                          </div>
                                          <div>
                                            <dt className="inline font-medium text-foreground">Attempts: </dt>
                                            <dd className="inline">{delivery.attempt_count}</dd>
                                          </div>
                                          <div>
                                            <dt className="inline font-medium text-foreground">Updated: </dt>
                                            <dd className="inline">{formatDateTime(delivery.updated_at)}</dd>
                                          </div>
                                          <div>
                                            <dt className="inline font-medium text-foreground">Terminal: </dt>
                                            <dd className="inline">
                                              {delivery.terminal_at ? formatDateTime(delivery.terminal_at) : 'Not terminal'}
                                            </dd>
                                          </div>
                                          <div>
                                            <dt className="inline font-medium text-foreground">Expires: </dt>
                                            <dd className="inline">{formatDateTime(delivery.expires_at)}</dd>
                                          </div>
                                          <div>
                                            <dt className="inline font-medium text-foreground">Redelivery: </dt>
                                            <dd className="inline break-all font-mono">
                                              {delivery.redelivery_of_id
                                                ? `Of ${delivery.redelivery_of_id}`
                                                : 'Original delivery'}
                                            </dd>
                                          </div>
                                        </dl>
                                        {delivery.completed_after_config_change && (
                                          <p className="text-xs text-amber-700">
                                            Completed after the registration configuration changed.
                                          </p>
                                        )}
                                        {attempts.length > 0 && (
                                          <ol className="divide-y text-xs text-muted-foreground">
                                            {attempts.map((attempt) => (
                                              <li key={attempt.id} className="flex flex-wrap gap-x-4 gap-y-1 py-2">
                                                <span className="font-medium text-foreground">
                                                  Attempt {attempt.sequence}: {attempt.state.replaceAll('_', ' ')}
                                                </span>
                                                <span>{httpStatusLabel(attempt.status_code)}</span>
                                                <span>
                                                  {attempt.latency_ms === null ? 'No latency' : `${attempt.latency_ms}ms`}
                                                </span>
                                                <span>
                                                  {attempt.request_timeout_seconds === null
                                                    ? 'No request timeout'
                                                    : `${attempt.request_timeout_seconds}s timeout`}
                                                </span>
                                                <span>
                                                  {attempt.requested_retry_delay_seconds === null
                                                    ? 'No requested retry delay'
                                                    : `${attempt.requested_retry_delay_seconds}s requested retry delay`}
                                                </span>
                                                {attempt.reason_code && (
                                                  <span>Reason: {attempt.reason_code.replaceAll('_', ' ')}</span>
                                                )}
                                                <span>Started {formatDateTime(attempt.started_at)}</span>
                                                <span>
                                                  {attempt.finished_at
                                                    ? `Finished ${formatDateTime(attempt.finished_at)}`
                                                    : 'Not finished'}
                                                </span>
                                              </li>
                                            ))}
                                          </ol>
                                        )}
                                      </li>
                                    ))}
                                  </ul>
                                )}
                              </section>
                            </TableCell>
                          </TableRow>
                        )}
                      </Fragment>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )
        ) : null}

        {ready && visibleTotal > 0 && (
          <nav className="flex flex-wrap items-center justify-between gap-3 border-t pt-4" aria-label="Webhook pagination">
            <p className="text-sm text-muted-foreground">
              Showing {offset + 1}-{offset + visibleCount} of {visibleTotal}
            </p>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void controller.goToPage(offset - WEBHOOK_PAGE_SIZE)}
                disabled={!hasPrevious || loading}
                aria-label="Previous page"
              >
                <ChevronLeft className="h-4 w-4" aria-hidden="true" />
                Previous
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void controller.goToPage(offset + WEBHOOK_PAGE_SIZE)}
                disabled={!hasNext || loading}
                aria-label="Next page"
              >
                Next
                <ChevronRight className="h-4 w-4" aria-hidden="true" />
              </Button>
            </div>
          </nav>
        )}
      </div>

      <Dialog open={createOpen} onOpenChange={controller.handleCreateOpenChange}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Add webhook</DialogTitle>
            <DialogDescription>
              Create an inactive registration and store its generated signing secret.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="webhook-create-url">Destination URL</Label>
              <Input
                id="webhook-create-url"
                type="url"
                autoComplete="off"
                value={createUrl}
                onChange={(event) => {
                  controller.setCreateUrl(event.target.value);
                  controller.setCreateUrlError('');
                  controller.setCommandError('');
                }}
                placeholder="https://receiver.example/hooks/events"
                maxLength={2_048}
                aria-invalid={Boolean(createUrlError)}
                aria-describedby={createUrlError ? 'webhook-create-url-error' : undefined}
                disabled={sensitiveCommandLocked}
              />
              {createUrlError && (
                <p id="webhook-create-url-error" role="alert" className="text-sm text-destructive">
                  {createUrlError}
                </p>
              )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="webhook-create-description">Description</Label>
              <Input
                id="webhook-create-description"
                value={createDescription}
                onChange={(event) => controller.setCreateDescription(event.target.value)}
                maxLength={500}
                disabled={sensitiveCommandLocked}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="webhook-create-timeout">Timeout (seconds)</Label>
              <Input
                id="webhook-create-timeout"
                type="number"
                min={1}
                max={30}
                step={1}
                value={createTimeout}
                onChange={(event) => controller.setCreateTimeout(event.target.value)}
                disabled={sensitiveCommandLocked}
              />
            </div>
            <fieldset className="space-y-3">
              <legend className="text-sm font-medium">Events</legend>
              {catalog?.events.map((event) => (
                <label key={event.event_type} className="flex items-start gap-3 rounded-md border p-3">
                  <Checkbox
                    checked={createEvents.includes(event.event_type)}
                    onCheckedChange={() => controller.toggleCreateEvent(event.event_type)}
                    disabled={sensitiveCommandLocked}
                  />
                  <span className="min-w-0">
                    <span className="block break-all font-mono text-sm">{event.event_type}</span>
                    <span className="block text-xs text-muted-foreground">{event.description}</span>
                  </span>
                </label>
              ))}
            </fieldset>
            {commandError && (
              <Alert variant="destructive">
                <AlertDescription className="space-y-3">
                  <p>{commandError}</p>
                  {pendingOperation === 'create' && (
                    <div className="flex flex-wrap gap-2">
                      <Button
                        type="button"
                        size="sm"
                        onClick={() => void controller.retrySecretCommand()}
                        loading={commandBusy}
                        loadingText="Retrying"
                      >
                        <RotateCw className="h-4 w-4" aria-hidden="true" />
                        Retry same command
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          controller.clearSensitiveCommandState(false);
                          controller.setCreateOpen(false);
                          void controller.loadControlPlane(0);
                        }}
                      >
                        Reload registrations
                      </Button>
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            )}
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => controller.handleCreateOpenChange(false)}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void controller.beginCanonicalCreate()}
              disabled={creating || sensitiveCommandLocked || !createUrl.trim() || createEvents.length === 0}
              loading={creating}
              loadingText="Creating"
            >
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(editor)} onOpenChange={(open) => !open && controller.setEditor(null)}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>
              {editor?.kind === 'destination' ? 'Replace webhook destination' : 'Edit webhook metadata'}
            </DialogTitle>
            <DialogDescription>
              {editor?.kind === 'destination'
                ? `Current redacted destination: ${editor.registration.target_display}`
                : 'A fresh registration revision will be shown for confirmation before saving.'}
            </DialogDescription>
          </DialogHeader>
          {editor?.kind === 'destination' ? (
            <div className="space-y-2">
              <Label htmlFor="webhook-replacement-url">New destination URL</Label>
              <Input
                id="webhook-replacement-url"
                type="url"
                autoComplete="off"
                value={replacementUrl}
                onChange={(event) => {
                  controller.setReplacementUrl(event.target.value);
                  controller.setReplacementUrlError('');
                }}
                placeholder="https://receiver.example/hooks/new"
                maxLength={2_048}
                aria-invalid={Boolean(replacementUrlError)}
                aria-describedby={replacementUrlError ? 'webhook-replacement-url-error' : undefined}
              />
              {replacementUrlError && (
                <p id="webhook-replacement-url-error" role="alert" className="text-sm text-destructive">
                  {replacementUrlError}
                </p>
              )}
            </div>
          ) : editor ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="webhook-edit-description">Description</Label>
                <Input
                  id="webhook-edit-description"
                  value={editDescription}
                  onChange={(event) => controller.setEditDescription(event.target.value)}
                  maxLength={500}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="webhook-edit-timeout">Timeout (seconds)</Label>
                <Input
                  id="webhook-edit-timeout"
                  type="number"
                  min={1}
                  max={30}
                  value={editTimeout}
                  onChange={(event) => controller.setEditTimeout(event.target.value)}
                />
              </div>
              <fieldset className="space-y-2">
                <legend className="text-sm font-medium">Events</legend>
                {catalog?.events.map((event) => (
                  <label key={event.event_type} className="flex items-start gap-3 rounded-md border p-3">
                    <Checkbox
                      checked={editEvents.includes(event.event_type)}
                      onCheckedChange={() => controller.toggleEditEvent(event.event_type)}
                    />
                    <span>
                      <span className="block break-all font-mono text-sm">{event.event_type}</span>
                      <span className="block text-xs text-muted-foreground">{event.description}</span>
                    </span>
                  </label>
                ))}
              </fieldset>
            </div>
          ) : null}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => controller.setEditor(null)}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void controller.submitEditor()}
              loading={editor ? mutatingId === editor.registration.id : false}
              loadingText="Saving"
              disabled={editor?.kind === 'destination' ? !replacementUrl.trim() : editEvents.length === 0}
            >
              {editor?.kind === 'destination' ? 'Save destination' : 'Save changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(secretState)} onOpenChange={(open) => !open && controller.requestSecretClose()}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Signing secret</DialogTitle>
            <DialogDescription>
              This value is shown once. Store it before leaving this page.
            </DialogDescription>
          </DialogHeader>
          {secretState && (
            <div className="space-y-4">
              {secretState.replayed && (
                <Alert>
                  <AlertDescription>
                    This response was recovered from the original command using its idempotency key.
                  </AlertDescription>
                </Alert>
              )}
              <div className="flex items-center gap-2">
                <Label htmlFor="webhook-signing-secret" className="sr-only">Signing secret</Label>
                <Input
                  id="webhook-signing-secret"
                  readOnly
                  value={secretState.value}
                  className="min-w-0 font-mono text-sm"
                />
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={() => void controller.handleCopySecret()}
                  aria-label="Copy signing secret"
                >
                  <Copy className="h-4 w-4" aria-hidden="true" />
                </Button>
              </div>
              {secretCopied && <p className="text-sm text-emerald-700" role="status">Copied to clipboard.</p>}
              <label className="flex items-start gap-3 rounded-md border p-3 text-sm">
                <Checkbox
                  checked={secretAcknowledged}
                  onCheckedChange={(checked) => controller.setSecretAcknowledged(checked === true)}
                />
                <span>I have stored this signing secret in the destination service.</span>
              </label>
              {secretWarning && (
                <Alert variant="destructive"><AlertDescription>{secretWarning}</AlertDescription></Alert>
              )}
            </div>
          )}
          <DialogFooter>
            <Button
              type="button"
              onClick={controller.requestSecretClose}
              disabled={!secretCopied || !secretAcknowledged}
            >
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </ResponsiveLayout>
  );
}

export default function WebhooksPage() {
  return (
    <PermissionGuard variant="route" requireAuth role={['admin', 'super_admin', 'owner']}>
      <WebhooksPageContent />
    </PermissionGuard>
  );
}
