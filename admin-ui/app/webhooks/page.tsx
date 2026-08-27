'use client';

import { Fragment } from 'react';
import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Copy,
  Edit3,
  KeyRound,
  Link2,
  Play,
  Plus,
  RefreshCw,
  RotateCw,
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
import type { WebhookStatus } from '@/types';
import {
  activationBlockReason,
  canLoadCanonicalData,
  useWebhooksPageController,
  WEBHOOK_PAGE_SIZE,
} from './use-webhooks-page-controller';

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
          <AlertDescription>
            The active webhook limit is exceeded. New activations are blocked.
          </AlertDescription>
        </Alert>
      )}
      {!status.delivery_capability_ready && (
        <Alert>
          <AlertDescription>
            Webhook delivery capability is unavailable. Registrations can be prepared, but activation is disabled.
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
    </div>
  );
}

function WebhooksPageContent() {
  const {
    mode,
    status,
    catalog,
    canonicalPage,
    legacyItems,
    offset,
    loading,
    statusError,
    createOpen,
    createUrl,
    createDescription,
    createTimeout,
    createEvents,
    legacyEvents,
    legacyEnabled,
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
    hasPendingCommand,
    legacyExpandedId,
    legacyDeliveries,
    legacyDeliveryLoading,
    addDisabled,
    visibleTotal,
    visibleCount,
    hasPrevious,
    hasNext,
    setCommandError,
    setCreateDescription,
    setCreateOpen,
    setCreateTimeout,
    setCreateUrl,
    setEditDescription,
    setEditTimeout,
    setEditor,
    setLegacyEnabled,
    setLegacyEvents,
    setReplacementUrl,
    setReplacementUrlError,
    setSecretAcknowledged,
    clearSensitiveCommandState,
    loadControlPlane,
    retrySecretCommand,
    beginCanonicalCreate,
    beginLegacyCreate,
    openCreate,
    handleCreateOpenChange,
    toggleCreateEvent,
    toggleEditEvent,
    openMetadataEditor,
    openDestinationEditor,
    submitEditor,
    toggleCanonicalRegistration,
    deleteCanonicalRegistration,
    rotateCanonicalSecret,
    handleCopySecret,
    requestSecretClose,
    toggleLegacyEnabled,
    deleteLegacyRegistration,
    testLegacyRegistration,
    toggleLegacyDeliveries,
    goToPage,
  } = useWebhooksPageController();

  return (
    <ResponsiveLayout>
      <div className="space-y-6 p-4 lg:p-8">
        <header className="flex flex-col gap-4 border-b pb-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Webhooks</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Manage outgoing event registrations and signing secrets.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => void loadControlPlane(offset)}
              loading={loading}
              loadingText="Refreshing"
            >
              <RefreshCw className="h-4 w-4" aria-hidden="true" />
              Refresh
            </Button>
            <Button type="button" size="sm" onClick={openCreate} disabled={addDisabled}>
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
              <Button type="button" variant="outline" size="sm" onClick={() => void loadControlPlane(0)}>
                <RefreshCw className="h-4 w-4" aria-hidden="true" />
                Retry status
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {status && mode === 'canonical' && <StatusAlerts status={status} />}

        {mode === 'legacy' && (
          <Alert>
            <AlertDescription>
              <strong>Legacy compatibility mode.</strong> ETags and secret rotation are unavailable. Complete migration before switching to canonical management.
            </AlertDescription>
          </Alert>
        )}

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
                  onClick={() => void retrySecretCommand()}
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
                    clearSensitiveCommandState(false);
                    void loadControlPlane(offset);
                  }}
                >
                  Reload registrations
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        )}

        {loading && visibleCount === 0 && !statusError ? (
          <div className="py-12 text-center text-sm text-muted-foreground" role="status" aria-live="polite">
            Loading webhooks...
          </div>
        ) : mode === 'canonical' && status && canLoadCanonicalData(status) ? (
          canonicalPage.items.length === 0 ? (
            <EmptyState
              icon={Webhook}
              title="No webhooks configured"
              description="Create an inactive registration, store its signing secret, then enable it when delivery is ready."
            />
          ) : (
            <div className="overflow-x-auto rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Destination</TableHead>
                    <TableHead>Description</TableHead>
                    <TableHead>Events</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Updated</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {canonicalPage.items.map((registration) => {
                    const activationReason = activationBlockReason(registration, status);
                    const busy = mutatingId === registration.id;
                    const rotationBlocked = registration.active
                      ? 'Disable the webhook before generating a new secret'
                      : status.key_state !== 'available'
                        ? 'Webhook signing key is unavailable'
                        : null;
                    return (
                      <TableRow key={registration.id}>
                        <TableCell>
                          <div className="max-w-64">
                            <p className="truncate font-mono text-sm" title={registration.target_display}>
                              {registration.target_display}
                            </p>
                            <p className="text-xs text-muted-foreground">ID {registration.id}, revision {registration.revision}</p>
                          </div>
                        </TableCell>
                        <TableCell className="max-w-56">
                          <span className="line-clamp-2 text-sm">{registration.description || 'No description'}</span>
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
                              <Badge variant="destructive">Secret rotation required</Badge>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {formatDateTime(registration.updated_at, { fallback: 'Unknown' })}
                        </TableCell>
                        <TableCell>
                          <div className="flex min-w-max flex-wrap justify-end gap-1">
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => openMetadataEditor(registration)}
                              disabled={busy}
                            >
                              <Edit3 className="h-4 w-4" aria-hidden="true" />
                              Edit metadata
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => openDestinationEditor(registration)}
                              disabled={busy}
                            >
                              <Link2 className="h-4 w-4" aria-hidden="true" />
                              Replace destination
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => void rotateCanonicalSecret(registration)}
                              disabled={busy || Boolean(rotationBlocked)}
                              title={rotationBlocked ?? undefined}
                            >
                              <KeyRound className="h-4 w-4" aria-hidden="true" />
                              Generate a new secret
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => void toggleCanonicalRegistration(registration)}
                              disabled={busy || Boolean(activationReason)}
                              title={activationReason ?? undefined}
                            >
                              {registration.active ? 'Disable' : 'Enable'}
                            </Button>
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              className="h-9 w-9"
                              onClick={() => void deleteCanonicalRegistration(registration)}
                              disabled={busy}
                              aria-label="Delete webhook"
                            >
                              <Trash2 className="h-4 w-4 text-destructive" aria-hidden="true" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )
        ) : mode === 'legacy' ? (
          legacyItems.length === 0 ? (
            <EmptyState icon={Webhook} title="No legacy webhooks configured" />
          ) : (
            <div className="space-y-4">
              <div className="overflow-x-auto rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Destination</TableHead>
                      <TableHead>Events</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Legacy actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {legacyItems.map((registration) => (
                      <Fragment key={registration.id}>
                        <TableRow>
                          <TableCell className="max-w-72 truncate font-mono text-sm">
                            {registration.targetUrl}
                          </TableCell>
                          <TableCell>
                            <div className="flex flex-wrap gap-1">
                              {registration.eventTypes.map((eventType) => (
                                <Badge key={eventType} variant="secondary">{eventType}</Badge>
                              ))}
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant={registration.enabled ? 'default' : 'outline'}>
                              {registration.enabled ? 'Enabled' : 'Disabled'}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <div className="flex min-w-max justify-end gap-1">
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => void testLegacyRegistration(registration)}
                              >
                                <Play className="h-4 w-4" aria-hidden="true" />
                                Test
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => void toggleLegacyDeliveries(registration)}
                                aria-label={legacyExpandedId === registration.id
                                  ? 'Hide delivery history'
                                  : 'Show delivery history'}
                              >
                                Delivery history
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => void toggleLegacyEnabled(registration)}
                              >
                                {registration.enabled ? 'Disable' : 'Enable'}
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                className="h-9 w-9"
                                onClick={() => void deleteLegacyRegistration(registration)}
                                aria-label="Delete legacy webhook"
                              >
                                <Trash2 className="h-4 w-4 text-destructive" aria-hidden="true" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                        {legacyExpandedId === registration.id && (
                          <TableRow>
                            <TableCell colSpan={4}>
                              <section aria-label="Delivery history" className="space-y-2 py-2">
                                <h2 className="text-sm font-semibold">Delivery history</h2>
                                {legacyDeliveryLoading ? (
                                  <p className="text-sm text-muted-foreground">Loading delivery history...</p>
                                ) : legacyDeliveries.length === 0 ? (
                                  <p className="text-sm text-muted-foreground">No legacy deliveries recorded.</p>
                                ) : (
                                  <ul className="space-y-2">
                                    {legacyDeliveries.map((delivery) => (
                                      <li key={delivery.id} className="flex flex-wrap gap-x-4 gap-y-1 border-t pt-2 text-sm">
                                        <span className="font-mono">{delivery.eventType}</span>
                                        <span>{delivery.success ? 'Succeeded' : 'Failed'}</span>
                                        <span>{delivery.statusCode ?? 'No HTTP status'}</span>
                                        <span>{formatDateTime(delivery.attemptedAt, { fallback: 'Unknown time' })}</span>
                                      </li>
                                    ))}
                                  </ul>
                                )}
                              </section>
                            </TableCell>
                          </TableRow>
                        )}
                      </Fragment>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )
        ) : null}

        {mode && visibleTotal > 0 && (
          <nav className="flex flex-wrap items-center justify-between gap-3 border-t pt-4" aria-label="Webhook pagination">
            <p className="text-sm text-muted-foreground">
              Showing {offset + 1}-{offset + visibleCount} of {visibleTotal}
            </p>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void goToPage(offset - WEBHOOK_PAGE_SIZE)}
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
                onClick={() => void goToPage(offset + WEBHOOK_PAGE_SIZE)}
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

      <Dialog open={createOpen} onOpenChange={handleCreateOpenChange}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Add webhook</DialogTitle>
            <DialogDescription>
              {mode === 'canonical'
                ? 'Create an inactive registration and store its generated signing secret.'
                : 'Create a registration through the legacy compatibility API.'}
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
                  setCreateUrl(event.target.value);
                  setCommandError('');
                }}
                placeholder="https://receiver.example/hooks/events"
                maxLength={2_048}
                aria-invalid={Boolean(commandError)}
                disabled={hasPendingCommand}
              />
            </div>
            {mode === 'canonical' ? (
              <>
                <div className="space-y-2">
                  <Label htmlFor="webhook-create-description">Description</Label>
                  <Input
                    id="webhook-create-description"
                    value={createDescription}
                    onChange={(event) => setCreateDescription(event.target.value)}
                    maxLength={500}
                    disabled={hasPendingCommand}
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
                    onChange={(event) => setCreateTimeout(event.target.value)}
                    disabled={hasPendingCommand}
                  />
                </div>
                <fieldset className="space-y-3">
                  <legend className="text-sm font-medium">Events</legend>
                  {catalog?.events.map((event) => (
                    <label key={event.event_type} className="flex items-start gap-3 rounded-md border p-3">
                      <Checkbox
                        checked={createEvents.includes(event.event_type)}
                        onCheckedChange={() => toggleCreateEvent(event.event_type)}
                        disabled={hasPendingCommand}
                      />
                      <span className="min-w-0">
                        <span className="block break-all font-mono text-sm">{event.event_type}</span>
                        <span className="block text-xs text-muted-foreground">{event.description}</span>
                      </span>
                    </label>
                  ))}
                </fieldset>
              </>
            ) : (
              <>
                <div className="space-y-2">
                  <Label htmlFor="legacy-webhook-events">Events</Label>
                  <Input
                    id="legacy-webhook-events"
                    value={legacyEvents}
                    onChange={(event) => setLegacyEvents(event.target.value)}
                    placeholder="incident.created, user.created"
                  />
                </div>
                <label className="flex items-center gap-2 text-sm">
                  <Checkbox
                    checked={legacyEnabled}
                    onCheckedChange={(checked) => setLegacyEnabled(checked === true)}
                  />
                  Enabled
                </label>
              </>
            )}
            {commandError && (
              <Alert variant="destructive">
                <AlertDescription className="space-y-3">
                  <p>{commandError}</p>
                  {pendingOperation === 'create' && (
                    <div className="flex flex-wrap gap-2">
                      <Button
                        type="button"
                        size="sm"
                        onClick={() => void retrySecretCommand()}
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
                          clearSensitiveCommandState(false);
                          setCreateOpen(false);
                          void loadControlPlane(0);
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
            <Button type="button" variant="outline" onClick={() => handleCreateOpenChange(false)}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void (mode === 'canonical' ? beginCanonicalCreate() : beginLegacyCreate())}
              disabled={
                creating
                || hasPendingCommand
                || !createUrl.trim()
                || (mode === 'canonical' ? createEvents.length === 0 : !legacyEvents.trim())
              }
              loading={creating}
              loadingText="Creating"
            >
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(editor)} onOpenChange={(open) => !open && setEditor(null)}>
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
                  setReplacementUrl(event.target.value);
                  setReplacementUrlError('');
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
                  onChange={(event) => setEditDescription(event.target.value)}
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
                  onChange={(event) => setEditTimeout(event.target.value)}
                />
              </div>
              <fieldset className="space-y-2">
                <legend className="text-sm font-medium">Events</legend>
                {catalog?.events.map((event) => (
                  <label key={event.event_type} className="flex items-start gap-3 rounded-md border p-3">
                    <Checkbox
                      checked={editEvents.includes(event.event_type)}
                      onCheckedChange={() => toggleEditEvent(event.event_type)}
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
            <Button type="button" variant="outline" onClick={() => setEditor(null)}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void submitEditor()}
              loading={editor ? mutatingId === editor.registration.id : false}
              loadingText="Saving"
              disabled={editor?.kind === 'destination' ? !replacementUrl.trim() : editEvents.length === 0}
            >
              {editor?.kind === 'destination' ? 'Save destination' : 'Save changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(secretState)} onOpenChange={(open) => !open && requestSecretClose()}>
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
                  onClick={() => void handleCopySecret()}
                  aria-label="Copy signing secret"
                >
                  <Copy className="h-4 w-4" aria-hidden="true" />
                </Button>
              </div>
              {secretCopied && (
                <p className="text-sm text-emerald-700" role="status">Copied to clipboard.</p>
              )}
              <label className="flex items-start gap-3 rounded-md border p-3 text-sm">
                <Checkbox
                  checked={secretAcknowledged}
                  onCheckedChange={(checked) => setSecretAcknowledged(checked === true)}
                />
                <span>I have stored this signing secret in the destination service.</span>
              </label>
              {secretWarning && (
                <Alert variant="destructive">
                  <AlertDescription>{secretWarning}</AlertDescription>
                </Alert>
              )}
            </div>
          )}
          <DialogFooter>
            <Button
              type="button"
              onClick={requestSecretClose}
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
    <PermissionGuard
      role={['admin', 'super_admin', 'owner']}
      requireAuth
      variant="route"
    >
      <WebhooksPageContent />
    </PermissionGuard>
  );
}
