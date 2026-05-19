'use client';

import { useCallback, useEffect, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Checkbox } from '@/components/ui/checkbox';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/toast';
import { CreateRegistrationCodeDialog } from '@/components/dashboard/CreateRegistrationCodeDialog';
import { Clipboard, KeyRound, Trash2 } from 'lucide-react';
import { AccessibleIconButton } from '@/components/ui/accessible-icon-button';
import { api } from '@/lib/api-client';
import Link from 'next/link';
import { logger } from '@/lib/logger';
import type { RegistrationCode, RegistrationSettings } from '@/types';

const isRegistrationCodeActive = (code: RegistrationCode) => {
  if (!code.expires_at) {
    return code.times_used < code.max_uses;
  }
  const expiresAt = new Date(code.expires_at);
  if (Number.isNaN(expiresAt.getTime())) {
    return code.times_used < code.max_uses;
  }
  return expiresAt >= new Date() && code.times_used < code.max_uses;
};

const formatShortDate = (dateStr: string) => {
  const date = new Date(dateStr);
  if (Number.isNaN(date.getTime())) return '\u2014';
  return date.toLocaleDateString();
};

export default function RegistrationCodesPage() {
  const confirm = useConfirm();
  const { success, error: showError } = useToast();

  const [registrationCodes, setRegistrationCodes] = useState<RegistrationCode[]>([]);
  const [registrationSettings, setRegistrationSettings] = useState<RegistrationSettings | null>(null);
  const [registrationSettingsError, setRegistrationSettingsError] = useState('');
  const [savingRegistrationSettings, setSavingRegistrationSettings] = useState(false);
  const [showRegistrationDialog, setShowRegistrationDialog] = useState(false);
  const [registrationForm, setRegistrationForm] = useState({
    max_uses: 1,
    expiry_days: 7,
    role_to_grant: 'user',
  });
  const [registrationError, setRegistrationError] = useState('');
  const [creatingRegistration, setCreatingRegistration] = useState(false);
  const [deletingRegistrationId, setDeletingRegistrationId] = useState<string | null>(null);
  const [latestRegistrationCode, setLatestRegistrationCode] = useState<RegistrationCode | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      setRegistrationSettingsError('');

      const [settingsResult, codesResult] = await Promise.allSettled([
        api.getRegistrationSettings(),
        api.getRegistrationCodes(),
      ]);

      if (settingsResult.status === 'fulfilled' && settingsResult.value) {
        setRegistrationSettings(settingsResult.value as RegistrationSettings);
      } else {
        setRegistrationSettingsError('Failed to load registration settings');
      }

      if (codesResult.status === 'fulfilled') {
        setRegistrationCodes(
          Array.isArray(codesResult.value) ? codesResult.value : []
        );
      }

      const failures = [
        { key: 'settings', label: 'registration settings', result: settingsResult },
        { key: 'codes', label: 'registration codes', result: codesResult },
      ].filter(
        (entry): entry is { key: string; label: string; result: PromiseRejectedResult } =>
          entry.result.status === 'rejected'
      );

      if (failures.length > 0) {
        const failedLabels = failures.map((entry) => entry.label);
        logger.warn('Registration page data fetch failures', {
          component: 'RegistrationCodesPage',
          failures: failedLabels.join(', '),
        });
        setError(`Some data failed to load: ${failedLabels.join(', ')}`);
      }
    } catch (err: unknown) {
      logger.error('Failed to load registration data', {
        component: 'RegistrationCodesPage',
        error: err instanceof Error ? err.message : String(err),
      });
      setError('Failed to load registration data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const copyToClipboard = async (value: string, label: string) => {
    try {
      await navigator.clipboard.writeText(value);
      success('Copied to clipboard', `${label} copied.`);
    } catch (err: unknown) {
      logger.error('Failed to copy to clipboard', {
        component: 'RegistrationCodesPage',
        error: err instanceof Error ? err.message : String(err),
      });
      showError('Copy failed', 'Please copy manually or check browser permissions.');
    }
  };

  const handleRegistrationSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setRegistrationError('');
    setCreatingRegistration(true);
    try {
      const created = await api.createRegistrationCode(registrationForm);
      setLatestRegistrationCode(created as RegistrationCode);
      success('Registration code created', `Role: ${registrationForm.role_to_grant}`);
      setShowRegistrationDialog(false);
      setRegistrationForm({ max_uses: 1, expiry_days: 7, role_to_grant: 'user' });
      await loadData();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to create registration code';
      setRegistrationError(message);
      showError('Registration code failed', message);
    } finally {
      setCreatingRegistration(false);
    }
  };

  const handleRegistrationDelete = async (code: RegistrationCode) => {
    const codeId = String(code.id);
    if (deletingRegistrationId === codeId) return;
    const confirmed = await confirm({
      title: 'Delete registration code',
      message: `Delete code ${code.code.slice(0, 6)}…? Users will no longer be able to register with it.`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'key',
    });
    if (!confirmed) return;

    try {
      setDeletingRegistrationId(codeId);
      await api.deleteRegistrationCode(code.id);
      success('Registration code deleted', `Code ${code.code.slice(0, 6)}… removed.`);
      await loadData();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete registration code';
      showError('Delete failed', message);
    } finally {
      setDeletingRegistrationId((prev) => (prev === codeId ? null : prev));
    }
  };

  const handleRegistrationSettingsUpdate = async (
    updates: Partial<RegistrationSettings>,
    toastMessage: string
  ) => {
    if (!registrationSettings || savingRegistrationSettings) return;
    const previous = registrationSettings;
    setRegistrationSettings({ ...registrationSettings, ...updates });
    setRegistrationSettingsError('');
    setSavingRegistrationSettings(true);
    try {
      const updated = await api.updateRegistrationSettings(updates as Record<string, unknown>);
      setRegistrationSettings(updated as RegistrationSettings);
      success('Registration settings updated', toastMessage);
    } catch (err: unknown) {
      setRegistrationSettings(previous);
      const message = err instanceof Error ? err.message : 'Failed to update registration settings';
      setRegistrationSettingsError(message);
      showError('Registration settings failed', message);
    } finally {
      setSavingRegistrationSettings(false);
    }
  };

  const handleRegistrationDialogOpenChange = (open: boolean) => {
    setShowRegistrationDialog(open);
    if (!open) setRegistrationError('');
  };

  const registrationEnabled = registrationSettings?.enable_registration ?? false;
  const registrationRequiresCode = registrationSettings?.require_registration_code ?? false;
  const registrationBlocked =
    registrationSettings?.self_registration_allowed === false && registrationEnabled;
  const activeRegistrationCount = registrationCodes.filter(isRegistrationCodeActive).length;

  return (
    <PermissionGuard variant="route" requireAuth permission="read:users">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">Registration Codes</h1>
              <p className="text-sm text-muted-foreground">
                Issue and manage registration codes for user onboarding.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Link href="/users">
                <Button variant="outline" size="sm">Back to Users</Button>
              </Link>
              <CreateRegistrationCodeDialog
                open={showRegistrationDialog}
                onOpenChange={handleRegistrationDialogOpenChange}
                error={registrationError}
                form={registrationForm}
                setForm={setRegistrationForm}
                creating={creatingRegistration}
                onSubmit={handleRegistrationSubmit}
              />
            </div>
          </div>

          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Registration Settings */}
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <KeyRound className="h-5 w-5" />
                Registration Settings
              </CardTitle>
              <CardDescription>
                Control how new users can sign up for the platform.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {registrationSettingsError && (
                <Alert variant="destructive" className="mb-4">
                  <AlertDescription>{registrationSettingsError}</AlertDescription>
                </Alert>
              )}
              <div className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium">Self-registration</p>
                    <p className="text-xs text-muted-foreground">Allow new users to sign up.</p>
                  </div>
                  <Checkbox
                    id="registration-enabled"
                    aria-label="Toggle self-registration"
                    checked={registrationEnabled}
                    disabled={!registrationSettings || savingRegistrationSettings}
                    onCheckedChange={(checked) => {
                      const enabled = Boolean(checked);
                      handleRegistrationSettingsUpdate(
                        { enable_registration: enabled },
                        enabled ? 'Self-registration enabled.' : 'Self-registration disabled.'
                      );
                    }}
                  />
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium">Require registration code</p>
                    <p className="text-xs text-muted-foreground">Limit signups to issued codes.</p>
                  </div>
                  <Checkbox
                    id="registration-requires-code"
                    aria-label="Toggle registration code requirement"
                    checked={registrationRequiresCode}
                    disabled={!registrationSettings || savingRegistrationSettings}
                    onCheckedChange={(checked) => {
                      const required = Boolean(checked);
                      handleRegistrationSettingsUpdate(
                        { require_registration_code: required },
                        required ? 'Registration codes required.' : 'Registration codes optional.'
                      );
                    }}
                  />
                </div>
                {registrationBlocked && (
                  <p className="text-xs text-muted-foreground">
                    Self-registration is blocked by profile{' '}
                    {registrationSettings?.profile ?? 'local-single-user'}.
                  </p>
                )}
                {registrationSettings && (
                  <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                    <Badge
                      className={
                        registrationEnabled
                          ? 'bg-green-500'
                          : 'bg-muted text-muted-foreground'
                      }
                    >
                      {registrationEnabled ? 'Registration enabled' : 'Registration disabled'}
                    </Badge>
                    <Badge variant="secondary">
                      {registrationRequiresCode ? 'Codes required' : 'Codes optional'}
                    </Badge>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Latest Created Code */}
          {latestRegistrationCode && (
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="text-base">Latest Created Code</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-sm break-all">
                    {latestRegistrationCode.code}
                  </span>
                  <AccessibleIconButton
                    icon={Clipboard}
                    label="Copy registration code"
                    variant="ghost"
                    onClick={() =>
                      copyToClipboard(latestRegistrationCode.code, 'Registration code')
                    }
                  />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Registration Codes List */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <KeyRound className="h-5 w-5" />
                Registration Codes
                <Badge variant="secondary" className="ml-auto">
                  {activeRegistrationCount} active
                </Badge>
              </CardTitle>
              <CardDescription>
                All registration codes issued for user onboarding.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="h-16 animate-pulse rounded-lg border bg-muted/40" />
                  ))}
                </div>
              ) : registrationCodes.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No registration codes yet. Create one using the button above.
                </p>
              ) : (
                <div className="space-y-3">
                  {registrationCodes.map((code) => {
                    const active = isRegistrationCodeActive(code);
                    const isDeleting = deletingRegistrationId === String(code.id);
                    return (
                      <div
                        key={code.id}
                        className="flex items-start justify-between gap-2 rounded-lg border p-3"
                      >
                        <div className="min-w-0 space-y-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="font-mono text-xs break-all">{code.code}</span>
                            <Badge
                              className={
                                active
                                  ? 'bg-green-500'
                                  : 'bg-muted text-muted-foreground'
                              }
                            >
                              {active ? 'Active' : 'Expired'}
                            </Badge>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            Role {code.role_to_grant} &bull; {code.times_used}/{code.max_uses}{' '}
                            used &bull; Expires {formatShortDate(String(code.expires_at))}
                          </p>
                        </div>
                        <div className="flex items-center gap-1">
                          <AccessibleIconButton
                            icon={Clipboard}
                            label="Copy registration code"
                            variant="ghost"
                            onClick={() => copyToClipboard(code.code, 'Registration code')}
                          />
                          <AccessibleIconButton
                            icon={Trash2}
                            label={
                              isDeleting
                                ? 'Deleting registration code'
                                : 'Delete registration code'
                            }
                            variant="ghost"
                            onClick={() => handleRegistrationDelete(code)}
                            disabled={isDeleting}
                            loading={isDeleting}
                            className="text-destructive hover:text-destructive"
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
