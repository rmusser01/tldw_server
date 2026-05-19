import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Bell, Mail, Webhook, Send, X, AlertCircle, MessageSquare, ToggleLeft, ToggleRight } from 'lucide-react';
import { FormSkeleton } from '@/components/ui/skeleton';
import type { NotificationSettings, NotificationChannel, RecentNotification } from '../types';
import { CardSkeleton } from '@/components/ui/skeleton';

type NotificationsPanelProps = {
  settings: NotificationSettings | null;
  recentNotifications: RecentNotification[];
  loading: boolean;
  saving: boolean;
  canSave: boolean;
  onSave: (settings: NotificationSettings) => Promise<boolean> | boolean;
  onTest: (payload?: { message?: string; severity?: string }) => Promise<void> | void;
};

const CHANNEL_TYPES = [
  { value: 'email', label: 'Email', icon: Mail },
  { value: 'webhook', label: 'Webhook', icon: Webhook },
  { value: 'slack', label: 'Slack', icon: MessageSquare },
  { value: 'discord', label: 'Discord', icon: MessageSquare },
] as const;

const ALERT_THRESHOLDS = [
  { value: 'info', label: 'Info (All)' },
  { value: 'warning', label: 'Warning+' },
  { value: 'error', label: 'Error+' },
  { value: 'critical', label: 'Critical Only' },
] as const;

const DIGEST_FREQUENCIES = [
  { value: 'hourly', label: 'Hourly' },
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
] as const;

const DELIVERY_CHANNELS = ['email', 'slack', 'webhook', 'discord'] as const;
const SENT_STATUSES = new Set<RecentNotification['status']>(['sent']);
const FAILED_STATUSES = new Set<RecentNotification['status']>(['failed']);

type NotificationsPanelShellProps = {
  children: ReactNode;
  enabledChannels: number;
  loading: boolean;
  saving: boolean;
  onTest: (payload?: { message?: string; severity?: string }) => Promise<void> | void;
};

const NotificationsPanelShell = ({
  children,
  enabledChannels,
  loading,
  saving,
  onTest,
}: NotificationsPanelShellProps) => (
  <Card>
    <CardHeader>
      <div className="flex items-center justify-between">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Notifications
          </CardTitle>
          <CardDescription>
            Configure alert notification channels
          </CardDescription>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            void onTest();
          }}
          disabled={loading || saving || enabledChannels === 0}
        >
          <Send className="mr-2 h-4 w-4" />
          Test
        </Button>
      </div>
    </CardHeader>
    <CardContent className="space-y-6">
      {children}
    </CardContent>
  </Card>
);

const formatTimestamp = (timestamp?: string) => {
  if (!timestamp) return '-';
  return new Date(timestamp).toLocaleString();
};

const getChannelIcon = (type: string) => {
  const channel = CHANNEL_TYPES.find((c) => c.value === type);
  if (channel) {
    const Icon = channel.icon;
    return <Icon className="h-4 w-4" />;
  }
  return <Bell className="h-4 w-4" />;
};

const createClientId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `channel-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const withChannelClientIds = (settings: NotificationSettings): NotificationSettings => ({
  ...settings,
  channels: settings.channels.map((channel) =>
    channel.clientId ? channel : { ...channel, clientId: createClientId() }
  ),
});

const stripChannelClientIds = (settings: NotificationSettings): NotificationSettings => ({
  ...settings,
  channels: settings.channels.map((channel) => {
    const { clientId, ...rest } = channel;
    void clientId;
    return rest;
  }),
});

type NotificationsPanelFormProps = Omit<NotificationsPanelProps, 'settings'> & {
  settings: NotificationSettings;
};

const NotificationsPanelForm = ({
  settings,
  recentNotifications,
  loading,
  saving,
  canSave,
  onSave,
  onTest,
}: NotificationsPanelFormProps) => {
  const baseSettings = useMemo(() => withChannelClientIds(settings), [settings]);
  const [editSettings, setEditSettings] = useState<NotificationSettings>(() => baseSettings);
  const [isDirty, setIsDirty] = useState(false);
  const [showAddChannel, setShowAddChannel] = useState(false);
  const [newChannelType, setNewChannelType] = useState<NotificationChannel['type']>('email');
  const [newChannelConfig, setNewChannelConfig] = useState('');
  const [retryingNotificationId, setRetryingNotificationId] = useState<string | null>(null);
  const effectiveSettings = isDirty ? editSettings : baseSettings;

  const deliveryStats = useMemo(() => {
    const total = recentNotifications.length;
    const sent = recentNotifications.filter((notification) => SENT_STATUSES.has(notification.status)).length;
    const failed = recentNotifications.filter((notification) => FAILED_STATUSES.has(notification.status)).length;
    const byChannel = DELIVERY_CHANNELS.map((channel) => ({
      channel,
      count: recentNotifications.filter(
        (notification) => notification.channel.toLowerCase() === channel,
      ).length,
    }));
    return {
      total,
      sent,
      failed,
      deliveryRate: total > 0 ? (sent / total) * 100 : 0,
      failureRate: total > 0 ? (failed / total) * 100 : 0,
      byChannel,
    };
  }, [recentNotifications]);

  const updateEditSettings = (
    next: NotificationSettings | ((prev: NotificationSettings) => NotificationSettings),
  ) => {
    setIsDirty(true);
    setEditSettings((prev) => {
      const current = isDirty ? prev : baseSettings;
      return typeof next === 'function' ? next(current) : next;
    });
  };

  const handleAddChannel = () => {
    if (!newChannelConfig.trim()) return;

    const newChannel: NotificationChannel = {
      type: newChannelType,
      enabled: true,
      config: {
        [newChannelType === 'email' ? 'address' : 'url']: newChannelConfig.trim(),
      },
      clientId: createClientId(),
    };

    updateEditSettings((prev) => ({
      ...prev,
      channels: [...prev.channels, newChannel],
    }));
    setShowAddChannel(false);
    setNewChannelConfig('');
  };

  const handleRemoveChannel = (index: number) => {
    updateEditSettings((prev) => ({
      ...prev,
      channels: prev.channels.filter((_, i) => i !== index),
    }));
  };

  const handleToggleChannel = (index: number) => {
    updateEditSettings((prev) => {
      const channels = [...prev.channels];
      channels[index] = { ...channels[index], enabled: !channels[index].enabled };
      return { ...prev, channels };
    });
  };

  const handleSave = async () => {
    const saved = await onSave(stripChannelClientIds(effectiveSettings));
    if (saved !== false) {
      setIsDirty(false);
    }
  };

  const enabledChannels = effectiveSettings.channels.filter((c) => c.enabled).length;

  const handleRetryNotification = async (notification: RecentNotification) => {
    setRetryingNotificationId(notification.id);
    try {
      await onTest({
        message: notification.message,
        severity: notification.severity || 'error',
      });
    } finally {
      setRetryingNotificationId((current) => (current === notification.id ? null : current));
    }
  };

  return (
    <NotificationsPanelShell
      enabledChannels={enabledChannels}
      loading={loading}
      saving={saving}
      onTest={onTest}
    >
      <>
            {/* Channels List */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Notification Channels</Label>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowAddChannel(true)}
                >
                  Add Channel
                </Button>
              </div>

              {showAddChannel && (
                <div className="p-3 border rounded-lg space-y-3">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-1">
                      <Label htmlFor="channel-type">Type</Label>
                      <Select
                        id="channel-type"
                        value={newChannelType}
                        onChange={(e) => setNewChannelType(e.target.value as NotificationChannel['type'])}
                      >
                        {CHANNEL_TYPES.map((type) => (
                          <option key={type.value} value={type.value}>
                            {type.label}
                          </option>
                        ))}
                      </Select>
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="channel-config">
                        {newChannelType === 'email' ? 'Email Address' : 'Webhook URL'}
                      </Label>
                      <Input
                        id="channel-config"
                        placeholder={newChannelType === 'email' ? 'alerts@example.com' : 'https://...'}
                        value={newChannelConfig}
                        onChange={(e) => setNewChannelConfig(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button size="sm" onClick={handleAddChannel} disabled={!newChannelConfig.trim()}>
                      Add
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => setShowAddChannel(false)}>
                      Cancel
                    </Button>
                  </div>
                </div>
              )}

              {effectiveSettings.channels.length === 0 ? (
                <div className="text-sm text-muted-foreground text-center py-4 border rounded-lg">
                  No notification channels configured
                </div>
              ) : (
                <div className="space-y-2">
                  {effectiveSettings.channels.map((channel, index) => (
                    <div
                      key={channel.clientId ?? `channel-${index}`}
                      className={`flex items-center justify-between p-3 rounded-lg border ${
                        channel.enabled ? 'bg-background' : 'bg-muted/30 opacity-60'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        {getChannelIcon(channel.type)}
                        <div>
                          <div className="font-medium capitalize">{channel.type}</div>
                          <div className="text-xs text-muted-foreground font-mono">
                            {Object.values(channel.config)[0] || 'No config'}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={channel.enabled ? 'default' : 'secondary'}>
                          {channel.enabled ? 'Enabled' : 'Disabled'}
                        </Badge>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleToggleChannel(index)}
                          aria-label={channel.enabled ? 'Disable channel' : 'Enable channel'}
                          title={channel.enabled ? 'Disable' : 'Enable'}
                        >
                          {channel.enabled ? <ToggleRight className="h-4 w-4" /> : <ToggleLeft className="h-4 w-4" />}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRemoveChannel(index)}
                          aria-label="Remove channel"
                          title="Remove"
                        >
                          <X className="h-4 w-4 text-red-500" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-3" data-testid="notification-delivery-dashboard">
              <Label>Delivery Dashboard</Label>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg border p-3">
                  <p className="text-xs uppercase text-muted-foreground">Total Sent</p>
                  <p className="text-xl font-semibold" data-testid="notification-delivery-total">
                    {deliveryStats.total}
                  </p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs uppercase text-muted-foreground">Delivery Rate</p>
                  <p className="text-xl font-semibold" data-testid="notification-delivery-rate">
                    {deliveryStats.deliveryRate.toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs uppercase text-muted-foreground">Failure Rate</p>
                  <p className="text-xl font-semibold" data-testid="notification-failure-rate">
                    {deliveryStats.failureRate.toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-lg border p-3">
                  <p className="text-xs uppercase text-muted-foreground">Failed</p>
                  <p className="text-xl font-semibold" data-testid="notification-delivery-failed">
                    {deliveryStats.failed}
                  </p>
                </div>
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {deliveryStats.byChannel.map((channelStats) => (
                  <div
                    key={channelStats.channel}
                    className="flex items-center justify-between rounded border px-3 py-2 text-sm"
                    data-testid={`notification-channel-${channelStats.channel}`}
                  >
                    <span className="capitalize">{channelStats.channel}</span>
                    <Badge variant="outline">{channelStats.count}</Badge>
                  </div>
                ))}
              </div>
            </div>

            {/* Settings */}
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-1">
                <Label htmlFor="alert-threshold">Alert Threshold</Label>
                <Select
                  id="alert-threshold"
                  value={effectiveSettings.alert_threshold}
                  onChange={(e) =>
                    updateEditSettings((prev) => ({
                      ...prev,
                      alert_threshold: e.target.value as NotificationSettings['alert_threshold'],
                    }))
                  }
                >
                  {ALERT_THRESHOLDS.map((threshold) => (
                    <option key={threshold.value} value={threshold.value}>
                      {threshold.label}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="space-y-1">
                <Label htmlFor="digest-frequency">Digest Frequency</Label>
                <Select
                  id="digest-frequency"
                  value={effectiveSettings.digest_frequency}
                  onChange={(e) =>
                    updateEditSettings((prev) => ({
                      ...prev,
                      digest_frequency: e.target.value as NotificationSettings['digest_frequency'],
                    }))
                  }
                  disabled={!effectiveSettings.digest_enabled}
                >
                  {DIGEST_FREQUENCIES.map((freq) => (
                    <option key={freq.value} value={freq.value}>
                      {freq.label}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="flex items-end">
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="digest-enabled"
                    checked={effectiveSettings.digest_enabled}
                    onCheckedChange={(checked) =>
                      updateEditSettings((prev) => ({ ...prev, digest_enabled: checked }))
                    }
                  />
                  <Label htmlFor="digest-enabled">Enable Digest</Label>
                </div>
              </div>
            </div>

            <Button onClick={handleSave} disabled={saving || !isDirty || !canSave} loading={saving} loadingText="Saving...">
              Save Settings
            </Button>

            {/* Recent Notifications */}
            {recentNotifications.length > 0 && (
              <div className="space-y-2">
                <Label>Recent Notifications</Label>
                <div className="space-y-1 max-h-64 overflow-y-auto" data-testid="recent-notifications-list">
                  {recentNotifications.slice(0, 5).map((notification) => (
                    <div
                      key={notification.id}
                      className={`p-2 rounded text-sm ${
                        notification.status === 'sent'
                          ? 'bg-green-50 dark:bg-green-900/20'
                          : notification.status === 'failed'
                            ? 'bg-red-50 dark:bg-red-900/20'
                            : 'bg-muted/30'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          {getChannelIcon(notification.channel)}
                          <span className="truncate">{notification.message}</span>
                        </div>
                        {notification.status === 'failed' && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => { void handleRetryNotification(notification); }}
                            loading={retryingNotificationId === notification.id}
                            loadingText="Retrying..."
                          >
                            Retry
                          </Button>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={
                            notification.status === 'sent'
                              ? 'default'
                              : notification.status === 'failed'
                                ? 'destructive'
                                : 'secondary'
                          }
                        >
                          {notification.status}
                        </Badge>
                        <span className="text-xs text-muted-foreground">
                          {formatTimestamp(notification.timestamp)}
                        </span>
                      </div>
                      {notification.error && (
                        <p className="mt-1 text-xs text-destructive" data-testid={`notification-error-${notification.id}`}>
                          {notification.error}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
      </>
    </NotificationsPanelShell>
  );
};

export default function NotificationsPanel({
  settings,
  recentNotifications,
  loading,
  saving,
  canSave,
  onSave,
  onTest,
}: NotificationsPanelProps) {
  const settingsId = settings?.id ?? 'notification-settings';

  if (loading) {
    return (
      <NotificationsPanelShell enabledChannels={0} loading={loading} saving={saving} onTest={onTest}>
        <FormSkeleton />
      </NotificationsPanelShell>
    );
  }

  if (!settings) {
    return (
      <NotificationsPanelShell enabledChannels={0} loading={loading} saving={saving} onTest={onTest}>
        <div className="text-center text-muted-foreground py-8">
          <AlertCircle className="h-12 w-12 mx-auto mb-2" />
          <p>Failed to load notification settings</p>
        </div>
      </NotificationsPanelShell>
    );
  }

  return (
    <NotificationsPanelForm
      key={settingsId}
      settings={settings}
      recentNotifications={recentNotifications}
      loading={loading}
      saving={saving}
      canSave={canSave}
      onSave={onSave}
      onTest={onTest}
    />
  );
}
