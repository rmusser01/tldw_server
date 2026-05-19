'use client';

import { useCallback, useEffect, useState, use } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { useRouter } from 'next/navigation';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Form, FormCheckbox, FormInput, FormSelect, FormTextarea } from '@/components/ui/form';
import { ArrowLeft, Save, Trash2, Mic, MicOff, BarChart3, ShieldCheck, CheckCircle2, XCircle } from 'lucide-react';
import { api } from '@/lib/api-client';
import {
  parseVoiceCommandInputs,
  parseVoiceCommandPhrases,
  testVoiceCommandPhraseMatch,
  type VoiceCommandMatchResult,
} from '@/lib/voice-commands';
import type { VoiceCommand, VoiceActionType, VoiceCommandUsage, VoiceCommandValidationResponse } from '@/types';
import { Skeleton } from '@/components/ui/skeleton';
import { logger } from '@/lib/logger';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/toast';
import Link from 'next/link';

const ACTION_TYPE_OPTIONS: { value: VoiceActionType; label: string }[] = [
  { value: 'mcp_tool', label: 'MCP Tool' },
  { value: 'workflow', label: 'Workflow' },
  { value: 'custom', label: 'Custom Handler' },
  { value: 'llm_chat', label: 'LLM Chat' },
];

const editCommandSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  phrases: z.string().min(1, 'At least one phrase is required'),
  action_type: z.enum(['mcp_tool', 'workflow', 'custom', 'llm_chat']),
  action_config: z.string().optional(),
  description: z.string().optional(),
  priority: z.coerce.number().int().min(0).default(0),
  requires_confirmation: z.boolean().default(false),
});

type EditCommandFormInput = z.input<typeof editCommandSchema>;
type EditCommandFormData = z.output<typeof editCommandSchema>;

export default function VoiceCommandDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: commandId } = use(params);
  const router = useRouter();
  const confirm = useConfirm();
  const { success, error: showError } = useToast();
  const [command, setCommand] = useState<VoiceCommand | null>(null);
  const [usage, setUsage] = useState<VoiceCommandUsage | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [isToggling, setIsToggling] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState('');
  const [saveError, setSaveError] = useState('');
  const [testInput, setTestInput] = useState('');
  const [testResult, setTestResult] = useState<VoiceCommandMatchResult | null>(null);
  const [validating, setValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<VoiceCommandValidationResponse | null>(null);
  const [validationError, setValidationError] = useState('');

  const form = useForm<EditCommandFormInput, unknown, EditCommandFormData>({
    resolver: zodResolver(editCommandSchema),
    defaultValues: {
      name: '',
      phrases: '',
      action_type: 'llm_chat',
      action_config: '{}',
      description: '',
      priority: 0,
      requires_confirmation: false,
    },
  });

  const loadCommand = useCallback(async (signal?: AbortSignal) => {
    try {
      setLoading(true);
      setError('');
      const data = await api.getVoiceCommand(commandId, signal);
      if (signal?.aborted) return;
      setCommand(data);

      // Populate form with existing data
      form.reset({
        name: data.name,
        phrases: data.phrases.join('\n'),
        action_type: data.action_type,
        action_config: JSON.stringify(data.action_config, null, 2),
        description: data.description || '',
        priority: data.priority,
        requires_confirmation: data.requires_confirmation,
      });
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      logger.error('Failed to load voice command', { component: 'VoiceCommandDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error ? err.message : 'Failed to load voice command');
    } finally {
      if (signal?.aborted) return;
      setLoading(false);
    }
  }, [commandId, form]);

  const loadUsage = useCallback(async (signal?: AbortSignal) => {
    try {
      const data = await api.getVoiceCommandUsage(commandId, { days: 30 }, signal);
      if (signal?.aborted) return;
      setUsage(data);
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      logger.warn('Failed to load usage data', { component: 'VoiceCommandDetailPage', error: err instanceof Error ? err.message : String(err) });
    }
  }, [commandId]);

  useEffect(() => {
    const controller = new AbortController();
    void loadCommand(controller.signal);
    void loadUsage(controller.signal);
    return () => controller.abort();
  }, [loadCommand, loadUsage]);

  const handleSave = form.handleSubmit(async (data) => {
    setSaveError('');
    try {
      setSaving(true);

      const parsedInputs = parseVoiceCommandInputs(data.phrases, data.action_config);
      if (!parsedInputs.ok) {
        setSaveError(parsedInputs.error);
        return;
      }

      await api.updateVoiceCommand(commandId, {
        name: data.name,
        phrases: parsedInputs.phrases,
        action_type: data.action_type,
        action_config: parsedInputs.actionConfig,
        description: data.description || undefined,
        priority: data.priority,
        requires_confirmation: data.requires_confirmation,
      });

      success('Command updated', `"${data.name}" has been saved.`);
      void loadCommand();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to save command';
      setSaveError(message);
      showError('Save failed', message);
    } finally {
      setSaving(false);
    }
  });

  const handleToggleEnabled = async () => {
    if (!command || isToggling) return;
    try {
      setIsToggling(true);
      await api.toggleVoiceCommand(commandId, !command.enabled);
      success(
        command.enabled ? 'Command disabled' : 'Command enabled',
        `"${command.name}" has been ${command.enabled ? 'disabled' : 'enabled'}.`
      );
      void loadCommand();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to toggle command';
      showError('Toggle failed', message);
    } finally {
      setIsToggling(false);
    }
  };

  const handleDelete = async () => {
    if (!command || isDeleting) return;
    const confirmed = await confirm({
      title: 'Delete Voice Command',
      message: `Delete "${command.name}"? This cannot be undone.`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;

    try {
      setIsDeleting(true);
      await api.deleteVoiceCommand(commandId);
      success('Command deleted', `"${command.name}" has been removed.`);
      router.push('/voice-commands');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete command';
      showError('Delete failed', message);
    } finally {
      setIsDeleting(false);
    }
  };

  const handleTestCommand = () => {
    const phraseInput = form.getValues('phrases');
    const triggerPhrases = parseVoiceCommandPhrases(phraseInput);
    setTestResult(testVoiceCommandPhraseMatch(testInput, triggerPhrases));
  };

  const handleValidate = async () => {
    if (validating) return;
    try {
      setValidating(true);
      setValidationError('');
      setValidationResult(null);
      const result = await api.validateVoiceCommand(commandId);
      setValidationResult(result);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Validation request failed';
      setValidationError(message);
      showError('Validation failed', message);
    } finally {
      setValidating(false);
    }
  };

  if (loading) {
    return (
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <Skeleton className="h-8 w-48 mb-2" />
            <Skeleton className="h-4 w-64 mb-8" />
            <Card>
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-24 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              </CardContent>
            </Card>
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    );
  }

  if (error || !command) {
    return (
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <Link href="/voice-commands" className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Voice Commands
            </Link>
            <Alert variant="destructive">
              <AlertDescription>{error || 'Voice command not found'}</AlertDescription>
            </Alert>
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    );
  }

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <Link href="/voice-commands" className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Voice Commands
          </Link>

          <div className="mb-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-3xl font-bold">{command.name}</h1>
                <Badge variant={command.enabled ? 'default' : 'secondary'}>
                  {command.enabled ? 'Enabled' : 'Disabled'}
                </Badge>
              </div>
              <p className="text-muted-foreground">
                {command.description || 'Voice command configuration'}
              </p>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={handleToggleEnabled}
                disabled={isToggling}
                loading={isToggling}
                loadingText={command.enabled ? 'Disabling...' : 'Enabling...'}
              >
                {command.enabled ? (
                  <>
                    <MicOff className="mr-2 h-4 w-4" />
                    Disable
                  </>
                ) : (
                  <>
                    <Mic className="mr-2 h-4 w-4" />
                    Enable
                  </>
                )}
              </Button>
              <Button variant="destructive" onClick={handleDelete} disabled={isDeleting} loading={isDeleting} loadingText="Deleting...">
                <Trash2 className="mr-2 h-4 w-4" />
                Delete
              </Button>
            </div>
          </div>

          {/* Usage Stats */}
          {usage && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-muted-foreground" />
                    <span className="text-2xl font-bold">{usage.total_invocations ?? 0}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">Total Invocations</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold text-green-600">{usage.success_count ?? 0}</div>
                  <p className="text-xs text-muted-foreground">Successful</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold text-red-600">{usage.error_count ?? 0}</div>
                  <p className="text-xs text-muted-foreground">Errors</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-2xl font-bold">
                    {usage.avg_response_time_ms != null ? `${usage.avg_response_time_ms.toFixed(0)}ms` : 'N/A'}
                  </div>
                  <p className="text-xs text-muted-foreground">Avg Response Time</p>
                </CardContent>
              </Card>
            </div>
          )}

          {saveError && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>{saveError}</AlertDescription>
            </Alert>
          )}

          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Test Command</CardTitle>
              <CardDescription>
                Enter sample text to check whether this command&apos;s trigger phrases would match.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-col gap-2 sm:flex-row">
                <Input
                  value={testInput}
                  onChange={(event) => {
                    setTestInput(event.target.value);
                    setTestResult(null);
                  }}
                  placeholder="Try: search for engineering notes"
                  aria-label="Sample command text"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleTestCommand}
                  disabled={testInput.trim().length === 0}
                >
                  Test Command
                </Button>
              </div>
              {testResult ? (
                <Alert variant={testResult.matched ? 'success' : 'default'}>
                  <AlertDescription>
                    <p className="font-medium">
                      {testResult.matched ? 'Matched' : 'No strong match'} ({Math.round(testResult.confidence * 100)}% confidence)
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {testResult.bestPhrase
                        ? `Best phrase: "${testResult.bestPhrase}" (threshold ${Math.round(testResult.threshold * 100)}%).`
                        : 'No trigger phrase candidates found.'}
                    </p>
                  </AlertDescription>
                </Alert>
              ) : null}
            </CardContent>
          </Card>

          {/* Dry-Run Validation */}
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Validate Configuration</CardTitle>
              <CardDescription>
                Dry-run check: verify the action config is valid and the referenced target
                (MCP tool, workflow template, handler) exists -- without executing anything.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button
                type="button"
                variant="outline"
                onClick={handleValidate}
                disabled={validating}
                loading={validating}
                loadingText="Validating..."
                data-testid="validate-button"
              >
                <ShieldCheck className="mr-2 h-4 w-4" />
                Validate
              </Button>

              {validationError && (
                <Alert variant="destructive">
                  <AlertDescription>{validationError}</AlertDescription>
                </Alert>
              )}

              {validationResult && (
                <div className="space-y-2" data-testid="validation-report">
                  <div className="flex items-center gap-2 mb-3">
                    {validationResult.valid ? (
                      <Badge variant="default" className="bg-green-600">All checks passed</Badge>
                    ) : (
                      <Badge variant="destructive">Validation failed</Badge>
                    )}
                    <span className="text-sm text-muted-foreground">
                      Action type: {validationResult.action_type}
                    </span>
                  </div>

                  {validationResult.steps.map((step) => (
                    <div
                      key={step.name}
                      className="flex items-start gap-3 rounded-md border p-3"
                      data-testid={`validation-step-${step.name}`}
                    >
                      {step.passed ? (
                        <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-green-600" />
                      ) : (
                        <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-600" />
                      )}
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm">{step.name}</p>
                        <p className="text-sm text-muted-foreground">{step.message}</p>
                        {step.details && (
                          <pre className="mt-1 text-xs bg-muted p-2 rounded overflow-auto max-h-32">
                            {JSON.stringify(step.details, null, 2)}
                          </pre>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Edit Command</CardTitle>
              <CardDescription>Modify the voice command configuration</CardDescription>
            </CardHeader>
            <CardContent>
              <FormProvider {...form}>
                <Form onSubmit={handleSave}>
                  <div className="space-y-6">
                    <FormInput<EditCommandFormData>
                      name="name"
                      label="Name"
                      placeholder="search_media"
                      description="Unique identifier for this command"
                      required
                    />

                    <FormTextarea<EditCommandFormData>
                      name="phrases"
                      label="Trigger Phrases"
                      placeholder="search for&#10;find&#10;look up"
                      description="One phrase per line or comma separated. These phrases trigger this command."
                      required
                    />

                    <FormSelect<EditCommandFormData>
                      name="action_type"
                      label="Action Type"
                      options={ACTION_TYPE_OPTIONS}
                      description="What type of action to execute when this command is triggered"
                    />

                    <FormTextarea<EditCommandFormData>
                      name="action_config"
                      label="Action Configuration (JSON)"
                      placeholder='{"tool_name": "media.search"}'
                      description="JSON configuration passed to the action handler"
                      className="font-mono text-sm"
                    />

                    <FormInput<EditCommandFormData>
                      name="description"
                      label="Description"
                      placeholder="Searches the media library for content"
                      description="Human-readable description of what this command does"
                    />

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <FormInput<EditCommandFormData>
                        name="priority"
                        label="Priority"
                        type="number"
                        description="Higher priority commands are matched first (0-100)"
                      />

                      <div className="space-y-2">
                        <Label>Options</Label>
                        <div className="pt-2">
                          <FormCheckbox<EditCommandFormData>
                            name="requires_confirmation"
                            label="Requires user confirmation before executing"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="flex justify-end gap-2 pt-4">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => router.push('/voice-commands')}
                        disabled={saving}
                      >
                        Cancel
                      </Button>
                      <Button type="submit" disabled={saving} loading={saving} loadingText="Saving...">
                        <Save className="mr-2 h-4 w-4" />
                        Save Changes
                      </Button>
                    </div>
                  </div>
                </Form>
              </FormProvider>
            </CardContent>
          </Card>

          {/* Raw Data Card */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Raw Data</CardTitle>
              <CardDescription>Complete command data for debugging</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted p-4 rounded-md overflow-auto text-sm">
                {JSON.stringify(command, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
