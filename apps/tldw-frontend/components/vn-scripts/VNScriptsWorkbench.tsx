import React, { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';
import JsonEditor from '@web/components/ui/JsonEditor';
import { apiClient } from '@web/lib/api';
import {
  applyVNScriptSnippet,
  createVNScriptFromTemplate,
  createVNScript,
  evaluateVNScriptVersionPolicy,
  getVNScriptAuthoringCatalog,
  getVNScriptDiagnostics,
  getVNScriptDraft,
  getVNScriptDraftGraph,
  getVNScriptManifestSnapshot,
  getVNScriptVersionGraph,
  listVNScriptTemplates,
  listVNScripts,
  listVNScriptVersions,
  previewVNScriptDraftGraph,
  previewVNScriptSnippet,
  publishVNScript,
  putVNScriptDraft,
  validateVNScriptDraft,
} from '@web/lib/api/vnScripts';
import type {
  VNScriptAuthoringCatalogResponse,
  VNScriptAuthoringGraphResponse,
  VNScriptAuthoringSnippet,
  VNScriptContentRating,
  VNScriptDiagnosticsResponse,
  VNScriptDraftResponse,
  VNScriptSnippetAnchor,
  VNScriptSnippetPreviewResponse,
  VNScriptResponse,
  VNScriptTemplateSummary,
  VNScriptValidationResponse,
  VNScriptVersionResponse,
} from '@web/types/vn-scripts';

const contentRatings: VNScriptContentRating[] = ['general', 'teen', 'suggestive', 'mature'];
const defaultSnippetAnchor: VNScriptSnippetAnchor = { label: 'start', mode: 'append', op_index: null };

interface VNCapabilities {
  features?: Record<string, boolean>;
}

interface ParameterField {
  enum?: unknown[];
  title?: string;
  type?: string | string[];
}

function formatJson(value: unknown): string {
  return JSON.stringify(value ?? {}, null, 2);
}

function errorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    const message = error.message.trim();
    if (/^[a-z0-9_.:-]{1,120}$/i.test(message)) return message;
  }
  return fallback;
}

function randomIdSuffix(): string {
  const randomSource = globalThis.crypto;
  if (randomSource && 'randomUUID' in randomSource) {
    return randomSource.randomUUID();
  }
  if (randomSource && 'getRandomValues' in randomSource) {
    const values = new Uint32Array(2);
    randomSource.getRandomValues(values);
    return Array.from(values, (value) => value.toString(36)).join('-');
  }
  const highResolutionTime = globalThis.performance?.now?.() ?? Date.now();
  return `${Date.now()}-${highResolutionTime.toString(36).replace('.', '')}`;
}

function createPublishIdempotencyKey(scriptId: number, draftRevision: number): string {
  return `vn-script-publish-${scriptId}-${draftRevision}-${randomIdSuffix()}`;
}

function validationBadge(validation: Record<string, unknown>): string {
  if (typeof validation.valid === 'boolean') {
    return validation.valid ? 'Valid' : 'Invalid';
  }
  if (typeof validation.status === 'string') return validation.status;
  return 'Validation saved';
}

function statusVariant(status: string): 'neutral' | 'success' | 'warning' | 'danger' | 'info' {
  if (status === 'ready' || status === 'published') return 'success';
  if (status === 'archived') return 'neutral';
  if (status === 'draft') return 'warning';
  return 'info';
}

function isSensitiveKey(key: string): boolean {
  return /(raw|debug|secret|token|credential|internal|prompt)/i.test(key);
}

function summarizeValue(value: unknown, depth = 0): unknown {
  if (depth > 3) return '[truncated]';
  if (Array.isArray(value)) {
    return value.slice(0, 8).map((item) => summarizeValue(item, depth + 1));
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, nestedValue]) => [
        key,
        isSensitiveKey(key) ? '[redacted]' : summarizeValue(nestedValue, depth + 1),
      ])
    );
  }
  return value;
}

function summaryLines(value: unknown): string[] {
  const summarized = summarizeValue(value);
  return (JSON.stringify(summarized ?? null, null, 2) ?? 'null').split('\n');
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function readString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function readErrorCode(record: Record<string, unknown> | null): string | undefined {
  if (!record) return undefined;
  return (
    readString(record.reason) ??
    readString(record.error_code) ??
    readErrorCode(asRecord(record.detail)) ??
    readErrorCode(asRecord(record.details)) ??
    readString(record.message) ??
    readString(record.code)
  );
}

function titleCase(value: string): string {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function getParameterProperties(snippet: VNScriptAuthoringSnippet): Record<string, ParameterField> {
  const properties = snippet.parameters_schema?.properties;
  if (!properties || typeof properties !== 'object' || Array.isArray(properties)) return {};
  return properties as Record<string, ParameterField>;
}

function parameterType(field: ParameterField): string | null {
  if (Array.isArray(field.type)) {
    return field.type.find((type) => type !== 'null') ?? null;
  }
  return field.type ?? null;
}

function enumOptionValue(index: number): string {
  return `enum-${index}`;
}

function enumValueFromOption(field: ParameterField, optionValue: unknown): unknown {
  if (!field.enum?.length || typeof optionValue !== 'string') return optionValue;
  const match = /^enum-(\d+)$/.exec(optionValue);
  if (!match) return optionValue;
  const index = Number(match[1]);
  return Number.isInteger(index) && index >= 0 && index < field.enum.length
    ? field.enum[index]
    : optionValue;
}

function selectedEnumOptionValue(field: ParameterField, value: unknown): string {
  const index = field.enum?.findIndex((option) => Object.is(option, value)) ?? -1;
  return index >= 0 ? enumOptionValue(index) : '';
}

function coerceParameterValue(field: ParameterField, value: unknown): unknown {
  const type = parameterType(field);
  if (field.enum?.length) return enumValueFromOption(field, value);
  if (type === 'number' || type === 'integer') {
    if (value === '') return '';
    const nextNumber = Number(value);
    return Number.isFinite(nextNumber) ? nextNumber : value;
  }
  if (type === 'boolean') return Boolean(value);
  if (type === 'string') return String(value ?? '');
  if (typeof value === 'string') {
    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }
  return value;
}

function snippetCategory(
  snippet: VNScriptAuthoringSnippet,
  catalog: VNScriptAuthoringCatalogResponse
): string {
  const firstOperation = snippet.operation_sequence[0];
  const operationCategory = catalog.operations.find((operation) => operation.op === firstOperation)?.category;
  if (operationCategory) return operationCategory;
  const categoryMatch = Object.entries(catalog.operation_categories).find(([, operations]) =>
    operations.includes(firstOperation)
  );
  return categoryMatch?.[0] ?? 'other';
}

function diagnosticCodes(items: Array<{ code?: unknown; message?: unknown }> | undefined): string {
  if (!items?.length) return 'none';
  return items
    .map((item) => readString(item.code) ?? readString(item.message) ?? 'diagnostic')
    .join(', ');
}

function GraphSummary({
  graph,
  onSourcePathSelect,
}: {
  graph: VNScriptAuthoringGraphResponse;
  onSourcePathSelect?: (sourcePath: string) => void;
}) {
  const validationErrors = graph.validation_diagnostics.errors;
  const validationWarnings = graph.validation_diagnostics.warnings;
  const limits = Object.entries(graph.limits ?? {});

  return (
    <div className="space-y-3 text-sm">
      {graph.truncated && (
        <p className="rounded-md border border-warn/30 bg-warn/10 p-2 text-warn">
          Graph output is truncated. Use diagnostics to see which limit was reached.
        </p>
      )}
      <dl className="grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-md border border-border bg-bg p-2">
          <dt className="text-text-muted">Source</dt>
          <dd className="break-words font-medium">{graph.source}</dd>
        </div>
        <div className="rounded-md border border-border bg-bg p-2">
          <dt className="text-text-muted">Content hash</dt>
          <dd className="break-words font-medium">{graph.content_hash}</dd>
        </div>
        <div className="rounded-md border border-border bg-bg p-2">
          <dt className="text-text-muted">Schema</dt>
          <dd className="break-words font-medium">{graph.schema_version}</dd>
        </div>
        <div className="rounded-md border border-border bg-bg p-2">
          <dt className="text-text-muted">Program schema</dt>
          <dd className="break-words font-medium">{graph.program_schema_version}</dd>
        </div>
        <div className="rounded-md border border-border bg-bg p-2">
          <dt className="text-text-muted">Base revision</dt>
          <dd>{graph.base_revision ?? 'n/a'}</dd>
        </div>
        <div className="rounded-md border border-border bg-bg p-2">
          <dt className="text-text-muted">Validation context</dt>
          <dd className="break-words">{graph.validation_context_source}</dd>
        </div>
        <div className="rounded-md border border-border bg-bg p-2">
          <dt className="text-text-muted">Graph semantics</dt>
          <dd className="break-words">{graph.graph_semantics_version}</dd>
        </div>
        <div className="rounded-md border border-border bg-bg p-2">
          <dt className="text-text-muted">Nodes / edges</dt>
          <dd>{graph.graph.nodes.length} / {graph.graph.edges.length}</dd>
        </div>
      </dl>
      {limits.length > 0 && (
        <p className="text-xs text-text-muted">
          Limits: {limits.map(([key, value]) => `${key} ${value}`).join(', ')}
        </p>
      )}
      <div>
        <h3 className="mb-1 text-xs font-semibold uppercase text-text-muted">Outline</h3>
        {graph.outline.labels.length === 0 ? (
          <p className="text-text-muted">No outline labels returned.</p>
        ) : (
          <ul className="space-y-2">
            {graph.outline.labels.map((label) => (
              <li key={label.id} className="rounded-md border border-border bg-bg p-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-medium">{label.label}</span>
                  <span className="text-xs text-text-muted">{label.id}</span>
                </div>
                <p className="mt-1 text-xs">{label.summary}</p>
                <p className="mt-1 text-xs text-text-muted">
                  {label.op_count} ops, {label.incoming_edge_count} in, {label.outgoing_edge_count} out,
                  {' '}{label.reachable ? 'reachable' : 'unreachable'}, {label.terminal}
                </p>
                {label.source_path ? (
                  <button
                    type="button"
                    aria-label={`Select source path for ${label.label}`}
                    className="mt-1 block break-all rounded-sm font-mono text-[11px] text-primary underline-offset-2 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                    onClick={() => onSourcePathSelect?.(label.source_path)}
                  >
                    {label.source_path}
                  </button>
                ) : (
                  <p className="mt-1 break-words font-mono text-[11px] text-text-muted">No source path.</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="grid gap-2 lg:grid-cols-2">
        <div className="rounded-md border border-border bg-bg p-2">
          <h3 className="text-xs font-semibold uppercase text-text-muted">Graph diagnostics</h3>
          <p className="mt-1 text-xs text-danger">Errors: {diagnosticCodes(graph.diagnostics.errors)}</p>
          <p className="mt-1 text-xs text-warn">Warnings: {diagnosticCodes(graph.diagnostics.warnings)}</p>
        </div>
        <div className="rounded-md border border-border bg-bg p-2">
          <h3 className="text-xs font-semibold uppercase text-text-muted">Validation diagnostics</h3>
          <p className="mt-1 text-xs text-danger">Errors: {diagnosticCodes(validationErrors)}</p>
          <p className="mt-1 text-xs text-warn">Warnings: {diagnosticCodes(validationWarnings)}</p>
        </div>
      </div>
    </div>
  );
}

export default function VNScriptsWorkbench() {
  const [scripts, setScripts] = useState<VNScriptResponse[]>([]);
  const [templates, setTemplates] = useState<VNScriptTemplateSummary[]>([]);
  const [selectedScript, setSelectedScript] = useState<VNScriptResponse | null>(null);
  const [draft, setDraft] = useState<VNScriptDraftResponse | null>(null);
  const [draftText, setDraftText] = useState('{}');
  const [versions, setVersions] = useState<VNScriptVersionResponse[]>([]);
  const [validation, setValidation] = useState<VNScriptValidationResponse | null>(null);
  const [diagnostics, setDiagnostics] = useState<VNScriptDiagnosticsResponse | null>(null);
  const [manifestSnapshots, setManifestSnapshots] = useState<Record<number, unknown>>({});
  const [policySummaries, setPolicySummaries] = useState<Record<number, unknown>>({});
  const [vnCapabilities, setVnCapabilities] = useState<VNCapabilities | null>(null);
  const [authoringCatalog, setAuthoringCatalog] = useState<VNScriptAuthoringCatalogResponse | null>(null);
  const [isLoadingScripts, setIsLoadingScripts] = useState(true);
  const [isLoadingTemplates, setIsLoadingTemplates] = useState(true);
  const [isLoadingCatalog, setIsLoadingCatalog] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isSavingDraft, setIsSavingDraft] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isLoadingDiagnostics, setIsLoadingDiagnostics] = useState(false);
  const [isPublishing, setIsPublishing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editorError, setEditorError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const [title, setTitle] = useState('Untitled VN script');
  const [primaryAssetPackId, setPrimaryAssetPackId] = useState('1');
  const [policyProfileId, setPolicyProfileId] = useState('');
  const [generationProfileId, setGenerationProfileId] = useState('');
  const [contentRating, setContentRating] = useState<VNScriptContentRating>('teen');
  const [selectedTemplateId, setSelectedTemplateId] = useState('blank');
  const [publishLabel, setPublishLabel] = useState('');
  const [catalogStatus, setCatalogStatus] = useState<string | null>(null);
  const [selectedSnippetId, setSelectedSnippetId] = useState('');
  const [snippetParameters, setSnippetParameters] = useState<Record<string, unknown>>({});
  const [snippetAnchor, setSnippetAnchor] = useState<VNScriptSnippetAnchor>(defaultSnippetAnchor);
  const [snippetPreview, setSnippetPreview] = useState<VNScriptSnippetPreviewResponse | null>(null);
  const [snippetPreviewContextKey, setSnippetPreviewContextKey] = useState<string | null>(null);
  const [isPreviewingSnippet, setIsPreviewingSnippet] = useState(false);
  const [isApplyingSnippet, setIsApplyingSnippet] = useState(false);
  const [snippetError, setSnippetError] = useState<string | null>(null);
  const [scriptGraph, setScriptGraph] = useState<VNScriptAuthoringGraphResponse | null>(null);
  const [versionGraphs, setVersionGraphs] = useState<Record<number, VNScriptAuthoringGraphResponse>>({});
  const [loadingGraphAction, setLoadingGraphAction] = useState<'saved' | 'preview' | null>(null);
  const [loadingVersionGraphId, setLoadingVersionGraphId] = useState<number | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [selectedGraphSourcePath, setSelectedGraphSourcePath] = useState<string | null>(null);
  const selectedScriptIdRef = useRef<number | null>(null);
  const draftHydrationRef = useRef<VNScriptDraftResponse | null>(null);
  const draftEditorRegionRef = useRef<HTMLDivElement | null>(null);
  const graphRequestIdRef = useRef(0);
  const versionGraphRequestIdRef = useRef(0);
  const publishKeyRef = useRef<Record<string, string>>({});
  const snippetPreviewContextKeyRef = useRef<string | null>(null);
  const previewRequestIdRef = useRef(0);

  function clearSnippetPreviewState() {
    setSnippetPreview(null);
    setSnippetPreviewContextKey(null);
    snippetPreviewContextKeyRef.current = null;
  }

  function clearGraphState() {
    graphRequestIdRef.current += 1;
    versionGraphRequestIdRef.current += 1;
    setScriptGraph(null);
    setVersionGraphs({});
    setGraphError(null);
    setLoadingGraphAction(null);
    setLoadingVersionGraphId(null);
    setSelectedGraphSourcePath(null);
  }

  useEffect(() => {
    selectedScriptIdRef.current = selectedScript?.id ?? null;
  }, [selectedScript?.id]);

  const selectedMeta = useMemo(() => {
    if (!selectedScript) return null;
    return [
      ['Asset pack', selectedScript.primary_asset_pack_id],
      ['Policy', selectedScript.policy_profile_id],
      ['Generation', selectedScript.generation_profile_id],
      ['Rating', selectedScript.content_rating],
    ];
  }, [selectedScript]);

  const selectedTemplate = useMemo(
    () => templates.find((template) => template.id === selectedTemplateId) ?? null,
    [selectedTemplateId, templates]
  );

  const catalogCapabilityTokens = useMemo(() => {
    const tokens = new Set<string>();
    if (vnCapabilities?.features) {
      Object.entries(vnCapabilities.features).forEach(([feature, enabled]) => {
        if (enabled) tokens.add(feature);
      });
    }
    authoringCatalog?.capability_tokens.forEach((token) => tokens.add(token));
    authoringCatalog?.generation_output_schemas.forEach((schema) => tokens.add(schema));
    return tokens;
  }, [authoringCatalog, vnCapabilities]);

  const visibleSnippets = useMemo(() => {
    if (!authoringCatalog) return [];
    return authoringCatalog.snippets.filter((snippet) => {
      if (!vnCapabilities) return true;
      return snippet.required_capability_tokens.every((token) => catalogCapabilityTokens.has(token));
    });
  }, [authoringCatalog, catalogCapabilityTokens, vnCapabilities]);

  const groupedSnippets = useMemo(() => {
    if (!authoringCatalog) return [];
    const groups = new Map<string, VNScriptAuthoringSnippet[]>();
    visibleSnippets.forEach((snippet) => {
      const category = snippetCategory(snippet, authoringCatalog);
      groups.set(category, [...(groups.get(category) ?? []), snippet]);
    });
    return Array.from(groups.entries()).sort(([left], [right]) => left.localeCompare(right));
  }, [authoringCatalog, visibleSnippets]);

  const selectedSnippet = useMemo(
    () => visibleSnippets.find((snippet) => snippet.id === selectedSnippetId) ?? null,
    [selectedSnippetId, visibleSnippets]
  );

  const savedDraftText = useMemo(() => formatJson(draft?.draft), [draft?.draft]);
  const hasUnsavedDraftText = draftText !== savedDraftText;
  const graphCapabilityEnabled = vnCapabilities?.features?.script_authoring_graph === true;

  const currentSnippetPreviewContextKey = useMemo(() => {
    if (!selectedScript || !selectedSnippet || !draft) return null;
    return JSON.stringify({
      script_id: selectedScript.id,
      draft_script_id: draft.script_id,
      snippet_id: selectedSnippet.id,
      anchor: snippetAnchor,
      parameters: snippetParameters,
      draft_revision: draft.revision,
      draft_text: draftText,
    });
  }, [draft, draftText, selectedScript, selectedSnippet, snippetAnchor, snippetParameters]);

  const hasCurrentSnippetPreview = Boolean(
    snippetPreview &&
    snippetPreviewContextKey &&
    currentSnippetPreviewContextKey &&
    snippetPreviewContextKey === currentSnippetPreviewContextKey &&
    !hasUnsavedDraftText
  );

  const handleTemplateChange = useCallback((templateId: string) => {
    setSelectedTemplateId(templateId);
    const nextTemplate = templates.find((template) => template.id === templateId);
    if (!nextTemplate) return;
    setTitle(nextTemplate.default_title);
    setContentRating(nextTemplate.recommended_content_rating);
  }, [templates]);

  const refreshVersions = useCallback(async (scriptId: number) => {
    const nextVersions = await listVNScriptVersions(scriptId);
    if (selectedScriptIdRef.current === scriptId) {
      setVersions(nextVersions.items ?? []);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadAuthoringCatalog() {
      setIsLoadingCatalog(true);
      setCatalogStatus(null);
      try {
        const capabilities = await apiClient.get('/vn/vn-capabilities') as VNCapabilities;
        if (cancelled) return;
        setVnCapabilities(capabilities);
        if (capabilities.features?.script_authoring_catalog !== true) {
          setAuthoringCatalog(null);
          return;
        }
        const catalog = await getVNScriptAuthoringCatalog();
        if (cancelled) return;
        setAuthoringCatalog(catalog);
      } catch {
        if (!cancelled) {
          setAuthoringCatalog(null);
          setCatalogStatus('Guided insert catalog unavailable. Raw JSON editing remains available.');
        }
      } finally {
        if (!cancelled) setIsLoadingCatalog(false);
      }
    }

    void loadAuthoringCatalog();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadTemplates() {
      setIsLoadingTemplates(true);
      try {
        const response = await listVNScriptTemplates();
        if (cancelled) return;
        setTemplates(response.items ?? []);
      } catch (loadError) {
        if (!cancelled) setError(errorMessage(loadError, 'Failed to load VN script templates'));
      } finally {
        if (!cancelled) setIsLoadingTemplates(false);
      }
    }

    void loadTemplates();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadScripts() {
      setIsLoadingScripts(true);
      setError(null);
      try {
        const response = await listVNScripts({ limit: 25, offset: 0 });
        if (cancelled) return;
        const nextScripts = response.items ?? [];
        setScripts(nextScripts);
        setSelectedScript(nextScripts[0] ?? null);
      } catch (loadError) {
        if (!cancelled) setError(errorMessage(loadError, 'Failed to load VN scripts'));
      } finally {
        if (!cancelled) setIsLoadingScripts(false);
      }
    }

    void loadScripts();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedScript) {
      setDraft(null);
      setDraftText('{}');
      setVersions([]);
      setValidation(null);
      setDiagnostics(null);
      setManifestSnapshots({});
      setPolicySummaries({});
      clearSnippetPreviewState();
      clearGraphState();
      return;
    }

    let cancelled = false;
    async function loadScriptDetails() {
      const hydratedDraft = draftHydrationRef.current;
      if (hydratedDraft?.script_id === selectedScript.id) {
        draftHydrationRef.current = null;
        setDraft(hydratedDraft);
        setDraftText(formatJson(hydratedDraft.draft));
        setVersions([]);
        setError(null);
        setEditorError(null);
        setStatusMessage(null);
        setValidation(null);
        setDiagnostics(null);
        setManifestSnapshots({});
        setPolicySummaries({});
        clearSnippetPreviewState();
        clearGraphState();
        try {
          const nextVersions = await listVNScriptVersions(selectedScript.id);
          if (!cancelled) setVersions(nextVersions.items ?? []);
        } catch {
          if (!cancelled) setVersions([]);
        }
        return;
      }
      setDraft(null);
      setDraftText('{}');
      setVersions([]);
      setError(null);
      setEditorError(null);
      setStatusMessage(null);
      setValidation(null);
      setDiagnostics(null);
      setManifestSnapshots({});
      setPolicySummaries({});
      clearSnippetPreviewState();
      clearGraphState();
      try {
        const [nextDraft, nextVersions] = await Promise.all([
          getVNScriptDraft(selectedScript.id),
          listVNScriptVersions(selectedScript.id),
        ]);
        if (cancelled) return;
        setDraft(nextDraft);
        setDraftText(formatJson(nextDraft.draft));
        setVersions(nextVersions.items ?? []);
      } catch (loadError) {
        if (!cancelled) {
          setDraft(null);
          setDraftText('{}');
          setVersions([]);
          setError(errorMessage(loadError, 'Failed to load script draft or versions'));
        }
      }
    }

    void loadScriptDetails();
    return () => {
      cancelled = true;
    };
  }, [selectedScript]);

  const parseDraftText = useCallback((): Record<string, unknown> | null => {
    try {
      const parsed = JSON.parse(draftText);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        setEditorError('Draft JSON must be an object.');
        return null;
      }
      setEditorError(null);
      return parsed as Record<string, unknown>;
    } catch (parseError) {
      setEditorError(
        `Draft JSON is invalid: ${parseError instanceof Error ? parseError.message : 'Parse failed'}`
      );
      return null;
    }
  }, [draftText]);

  const handleSnippetSelect = useCallback((snippet: VNScriptAuthoringSnippet) => {
    setSelectedSnippetId(snippet.id);
    setSnippetParameters({ ...snippet.default_parameters });
    clearSnippetPreviewState();
    setSnippetError(null);
  }, []);

  const handleSnippetParameterChange = useCallback((
    name: string,
    field: ParameterField,
    value: unknown
  ) => {
    setSnippetParameters((previous) => ({
      ...previous,
      [name]: coerceParameterValue(field, value),
    }));
    clearSnippetPreviewState();
  }, []);

  const handleSnippetAnchorChange = useCallback((nextAnchor: VNScriptSnippetAnchor) => {
    setSnippetAnchor(nextAnchor);
    clearSnippetPreviewState();
  }, []);

  const handlePreviewSnippet = useCallback(async () => {
    if (!selectedScript || !selectedSnippet) return;
    const previewScriptId = selectedScript.id;
    const previewContextKey = currentSnippetPreviewContextKey;
    const requestDraft = parseDraftText();
    if (!requestDraft) return;
    if (!previewContextKey) return;

    setIsPreviewingSnippet(true);
    setSnippetError(null);
    setSnippetPreview(null);
    setSnippetPreviewContextKey(null);
    snippetPreviewContextKeyRef.current = previewContextKey;
    const previewRequestId = previewRequestIdRef.current + 1;
    previewRequestIdRef.current = previewRequestId;
    try {
      const preview = await previewVNScriptSnippet(previewScriptId, {
        snippet_id: selectedSnippet.id,
        anchor: snippetAnchor,
        parameters: snippetParameters,
        draft: requestDraft,
        draft_revision: draft?.revision ?? null,
      });
      if (
        selectedScriptIdRef.current !== previewScriptId ||
        snippetPreviewContextKeyRef.current !== previewContextKey
      ) {
        return;
      }
      setSnippetPreview(preview);
      setSnippetPreviewContextKey(previewContextKey);
    } catch (previewError) {
      if (
        selectedScriptIdRef.current !== previewScriptId ||
        snippetPreviewContextKeyRef.current !== previewContextKey
      ) {
        return;
      }
      setSnippetError(errorMessage(previewError, 'Failed to preview snippet'));
    } finally {
      if (previewRequestIdRef.current === previewRequestId) {
        setIsPreviewingSnippet(false);
      }
    }
  }, [
    currentSnippetPreviewContextKey,
    draft?.revision,
    parseDraftText,
    selectedScript,
    selectedSnippet,
    snippetAnchor,
    snippetParameters,
  ]);

  const handleApplySnippet = useCallback(async () => {
    if (!selectedScript || !selectedSnippet || !draft || !hasCurrentSnippetPreview) return;
    const scriptId = selectedScript.id;
    const applyContextKey = snippetPreviewContextKeyRef.current;
    if (!applyContextKey) return;
    const isApplyContextCurrent = () =>
      selectedScriptIdRef.current === scriptId && snippetPreviewContextKeyRef.current === applyContextKey;

    setIsApplyingSnippet(true);
    setSnippetError(null);
    setStatusMessage(null);
    try {
      const applied = await applyVNScriptSnippet(scriptId, {
        if_revision: draft.revision,
        snippet_id: selectedSnippet.id,
        anchor: snippetAnchor,
        parameters: snippetParameters,
      });
      if (!isApplyContextCurrent()) return;
      const nextDraft = {
        script_id: applied.script_id,
        revision: applied.revision,
        draft: applied.draft,
        diagnostics: applied.diagnostics,
      };
      setDraft(nextDraft);
      setDraftText(formatJson(applied.draft));
      setDiagnostics({
        script_id: applied.script_id,
        revision: applied.revision,
        diagnostics: applied.diagnostics,
      });
      setValidation(
        typeof applied.diagnostics.valid === 'boolean'
          ? {
              valid: applied.diagnostics.valid,
              errors: Array.isArray(applied.diagnostics.errors) ? applied.diagnostics.errors : [],
              warnings: Array.isArray(applied.diagnostics.warnings) ? applied.diagnostics.warnings : [],
            }
          : null
      );
      setSnippetPreview({
        script_id: applied.script_id,
        base_revision: draft.revision,
        snippet_id: applied.snippet_id,
        draft: applied.draft,
        diagnostics: applied.diagnostics,
        patch_summary: applied.patch_summary,
        warnings: [],
      });
      setSnippetPreviewContextKey(null);
      snippetPreviewContextKeyRef.current = null;
      setStatusMessage(`Applied snippet at revision ${applied.revision}.`);
    } catch (applyError) {
      if (!isApplyContextCurrent()) return;
      const message = errorMessage(applyError, 'Failed to apply snippet');
      const errorCode = readErrorCode(asRecord(applyError));
      if (message === 'draft_revision_conflict' || errorCode === 'draft_revision_conflict') {
        try {
          const latestDraft = await getVNScriptDraft(scriptId);
          if (!isApplyContextCurrent()) return;
          setDraft(latestDraft);
          setDraftText(formatJson(latestDraft.draft));
          setDiagnostics(null);
          setValidation(null);
          clearSnippetPreviewState();
          setStatusMessage('Draft changed on the server. Reloaded the latest draft; review before applying again.');
        } catch {
          if (!isApplyContextCurrent()) return;
          setSnippetError('Draft changed on the server. Refresh the draft before applying again.');
        }
      } else {
        setSnippetError(message);
      }
    } finally {
      if (selectedScriptIdRef.current === scriptId) {
        setIsApplyingSnippet(false);
      }
    }
  }, [draft, hasCurrentSnippetPreview, selectedScript, selectedSnippet, snippetAnchor, snippetParameters]);

  const handleCreateScript = useCallback(async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const parsedAssetPackId = Number(primaryAssetPackId);
    if (!title.trim() || !Number.isInteger(parsedAssetPackId) || parsedAssetPackId <= 0) {
      setError('Enter a title and a positive primary asset pack ID.');
      return;
    }

    setIsCreating(true);
    setError(null);
    setStatusMessage(null);
    try {
      const request = {
        title: title.trim(),
        ...(selectedTemplate?.default_description
          ? { description: selectedTemplate.default_description }
          : {}),
        primary_asset_pack_id: parsedAssetPackId,
        ...(policyProfileId.trim() ? { policy_profile_id: policyProfileId.trim() } : {}),
        ...(generationProfileId.trim() ? { generation_profile_id: generationProfileId.trim() } : {}),
        content_rating: contentRating,
      };
      const created =
        selectedTemplateId === 'blank'
          ? await createVNScript(request)
          : await createVNScriptFromTemplate(selectedTemplateId, request);
      const createdScript = 'script' in created ? created.script : created;
      if ('draft' in created) {
        draftHydrationRef.current = created.draft;
        setDraft(created.draft);
        setDraftText(formatJson(created.draft.draft));
        setVersions([]);
        setValidation(null);
        setDiagnostics(null);
        setManifestSnapshots({});
        setPolicySummaries({});
      }
      setScripts((previous) => [
        createdScript,
        ...previous.filter((script) => script.id !== createdScript.id),
      ]);
      setSelectedScript(createdScript);
      setStatusMessage(`Created script #${createdScript.id}.`);
    } catch (createError) {
      setError(errorMessage(createError, 'Failed to create VN script'));
    } finally {
      setIsCreating(false);
    }
  }, [
    contentRating,
    generationProfileId,
    policyProfileId,
    primaryAssetPackId,
    selectedTemplate,
    selectedTemplateId,
    title,
  ]);

  const handleSaveDraft = useCallback(async () => {
    if (!selectedScript || !draft) return;
    const parsed = parseDraftText();
    if (!parsed) return;

    setIsSavingDraft(true);
    setError(null);
    setStatusMessage(null);
    try {
      const updated = await putVNScriptDraft(selectedScript.id, {
        if_revision: draft.revision,
        draft: parsed,
      });
      setDraft(updated);
      setDraftText(formatJson(updated.draft));
      clearGraphState();
      setStatusMessage(`Draft saved at revision ${updated.revision}.`);
    } catch (saveError) {
      setError(errorMessage(saveError, 'Failed to save draft'));
    } finally {
      setIsSavingDraft(false);
    }
  }, [draft, parseDraftText, selectedScript]);

  const handleValidate = useCallback(async () => {
    if (!selectedScript) return;
    const requestDraft = parseDraftText();
    if (!requestDraft) return;

    setIsValidating(true);
    setError(null);
    setStatusMessage(null);
    try {
      const result = await validateVNScriptDraft(selectedScript.id, {
        draft: requestDraft,
      });
      setValidation(result);
    } catch (validateError) {
      setError(errorMessage(validateError, 'Failed to validate draft'));
    } finally {
      setIsValidating(false);
    }
  }, [parseDraftText, selectedScript]);

  const handleDiagnostics = useCallback(async () => {
    if (!selectedScript) return;
    setIsLoadingDiagnostics(true);
    setError(null);
    try {
      const result = await getVNScriptDiagnostics(selectedScript.id);
      setDiagnostics(result);
    } catch (diagnosticsError) {
      setError(errorMessage(diagnosticsError, 'Failed to load diagnostics'));
    } finally {
      setIsLoadingDiagnostics(false);
    }
  }, [selectedScript]);

  const handleLoadDraftGraph = useCallback(async () => {
    if (!selectedScript) return;
    const scriptId = selectedScript.id;
    const requestId = graphRequestIdRef.current + 1;
    graphRequestIdRef.current = requestId;
    setLoadingGraphAction('saved');
    setGraphError(null);
    try {
      const graph = await getVNScriptDraftGraph(scriptId);
      if (selectedScriptIdRef.current !== scriptId || graphRequestIdRef.current !== requestId) return;
      setScriptGraph(graph);
    } catch (graphLoadError) {
      if (selectedScriptIdRef.current === scriptId && graphRequestIdRef.current === requestId) {
        setGraphError(errorMessage(graphLoadError, 'Failed to load script graph'));
      }
    } finally {
      if (graphRequestIdRef.current === requestId) setLoadingGraphAction(null);
    }
  }, [selectedScript]);

  const handlePreviewDraftGraph = useCallback(async () => {
    if (!selectedScript) return;
    const requestDraft = parseDraftText();
    if (!requestDraft) return;
    const scriptId = selectedScript.id;
    const requestId = graphRequestIdRef.current + 1;
    graphRequestIdRef.current = requestId;
    setLoadingGraphAction('preview');
    setGraphError(null);
    try {
      const graph = await previewVNScriptDraftGraph(scriptId, {
        draft: requestDraft,
        draft_revision: draft?.revision ?? null,
      });
      if (selectedScriptIdRef.current !== scriptId || graphRequestIdRef.current !== requestId) return;
      setScriptGraph(graph);
    } catch (graphPreviewError) {
      if (selectedScriptIdRef.current === scriptId && graphRequestIdRef.current === requestId) {
        setGraphError(errorMessage(graphPreviewError, 'Failed to preview script graph'));
      }
    } finally {
      if (graphRequestIdRef.current === requestId) setLoadingGraphAction(null);
    }
  }, [draft?.revision, parseDraftText, selectedScript]);

  const handleVersionGraph = useCallback(async (version: VNScriptVersionResponse) => {
    if (!selectedScript) return;
    const scriptId = selectedScript.id;
    const requestId = versionGraphRequestIdRef.current + 1;
    versionGraphRequestIdRef.current = requestId;
    setLoadingVersionGraphId(version.id);
    setGraphError(null);
    try {
      const graph = await getVNScriptVersionGraph(scriptId, version.id);
      if (selectedScriptIdRef.current !== scriptId || versionGraphRequestIdRef.current !== requestId) return;
      setVersionGraphs((previous) => ({ ...previous, [version.id]: graph }));
    } catch (versionGraphError) {
      if (selectedScriptIdRef.current === scriptId && versionGraphRequestIdRef.current === requestId) {
        setGraphError(errorMessage(versionGraphError, 'Failed to load version graph'));
      }
    } finally {
      if (versionGraphRequestIdRef.current === requestId) setLoadingVersionGraphId(null);
    }
  }, [selectedScript]);

  const handleGraphSourcePathSelect = useCallback((sourcePath: string) => {
    setSelectedGraphSourcePath(sourcePath);
    draftEditorRegionRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, []);

  const handlePublish = useCallback(async () => {
    if (!selectedScript || !draft) return;
    const scriptId = selectedScript.id;
    const publishScope = `${scriptId}:${draft.revision}`;
    const idempotencyKey =
      publishKeyRef.current[publishScope] ?? createPublishIdempotencyKey(scriptId, draft.revision);
    publishKeyRef.current[publishScope] = idempotencyKey;

    setIsPublishing(true);
    setError(null);
    setStatusMessage(null);
    let response: Awaited<ReturnType<typeof publishVNScript>>;
    try {
      response = await publishVNScript(scriptId, {
        draft_revision: draft.revision,
        label: publishLabel.trim() || null,
        idempotency_key: idempotencyKey,
        acknowledgements: [],
      });
    } catch (publishError) {
      setError(errorMessage(publishError, 'Failed to publish script'));
      setIsPublishing(false);
      return;
    }

    delete publishKeyRef.current[publishScope];
    setVersions((previous) => {
      if (selectedScriptIdRef.current !== scriptId) return previous;
      const optimisticVersion: VNScriptVersionResponse = {
        id: response.version_id,
        script_id: response.script_id,
        version_number: response.version_number,
        label: publishLabel.trim() || null,
        draft_revision: draft.revision,
        program: {},
        asset_pack_id: response.asset_pack_id,
        manifest_snapshot_id: response.manifest_snapshot_id,
        policy_snapshot_id: response.policy_snapshot_id,
        generation_profile_snapshot_id: response.generation_profile_snapshot_id,
        generation_profile_snapshots: response.generation_profile_snapshots,
        script_defaults: {},
        validation: response.validation,
        created_at: response.created_at,
      };
      return [optimisticVersion, ...previous.filter((version) => version.id !== optimisticVersion.id)];
    });

    try {
      setStatusMessage(`Published version ${response.version_number}.`);
      await refreshVersions(scriptId);
    } catch (refreshError) {
      setError(errorMessage(refreshError, 'Published, but failed to refresh versions'));
    } finally {
      setIsPublishing(false);
    }
  }, [draft, publishLabel, refreshVersions, selectedScript]);

  const handleManifestSnapshot = useCallback(async (version: VNScriptVersionResponse) => {
    if (!selectedScript) return;
    setError(null);
    try {
      const snapshot = await getVNScriptManifestSnapshot(selectedScript.id, version.id);
      setManifestSnapshots((previous) => ({
        ...previous,
        [version.id]: {
          id: snapshot.id,
          asset_pack_id: snapshot.asset_pack_id,
          manifest_hash: snapshot.manifest_hash,
          manifest_summary: summarizeValue(snapshot.manifest),
          created_at: snapshot.created_at,
        },
      }));
    } catch (manifestError) {
      setError(errorMessage(manifestError, 'Failed to load manifest snapshot'));
    }
  }, [selectedScript]);

  const handlePolicyEvaluate = useCallback(async (version: VNScriptVersionResponse) => {
    if (!selectedScript) return;
    setError(null);
    try {
      const summary = await evaluateVNScriptVersionPolicy(selectedScript.id, version.id);
      setPolicySummaries((previous) => ({
        ...previous,
        [version.id]: {
          decision: summary.decision,
          profile_id: summary.profile_id,
          blocked: summary.blocked,
          requires_acknowledgement: summary.requires_acknowledgement,
          reasons: summarizeValue(summary.reasons),
          remediation: summary.remediation,
        },
      }));
    } catch (policyError) {
      setError(errorMessage(policyError, 'Failed to evaluate version policy'));
    }
  }, [selectedScript]);

  return (
    <main className="min-h-screen bg-bg text-text">
      <div className="flex min-h-screen flex-col gap-3 p-4">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
          <div>
            <h1 className="text-xl font-semibold">VN Scripts</h1>
            <p className="text-sm text-text-muted">Author JSON drafts, inspect backend validation, and publish immutable versions.</p>
          </div>
          <div className="flex items-center gap-2 text-xs text-text-muted">
            <Badge variant="info">Backend-owned validation</Badge>
            <span>{scripts.length} loaded</span>
          </div>
        </header>

        {(error || editorError || statusMessage) && (
          <section className="grid gap-2 md:grid-cols-3">
            {error && <div className="rounded-md border border-danger/30 bg-danger/10 p-2 text-sm text-danger">{error}</div>}
            {editorError && <div className="rounded-md border border-warn/30 bg-warn/10 p-2 text-sm text-warn">{editorError}</div>}
            {statusMessage && <div className="rounded-md border border-success/30 bg-success/10 p-2 text-sm text-success">{statusMessage}</div>}
          </section>
        )}

        <div className="grid min-h-0 flex-1 gap-3 xl:grid-cols-[320px_minmax(0,1fr)_420px]">
          <aside className="flex min-h-0 flex-col gap-3 border-r border-border pr-3">
            <section className="rounded-md border border-border bg-surface p-3">
              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-sm font-semibold">Scripts</h2>
                {isLoadingScripts && <span className="text-xs text-text-muted">Loading...</span>}
              </div>
              <div className="max-h-80 space-y-2 overflow-auto">
                {scripts.length === 0 && !isLoadingScripts && (
                  <p className="text-sm text-text-muted">No VN scripts yet.</p>
                )}
                {scripts.map((script) => (
                  <button
                    key={script.id}
                    type="button"
                    onClick={() => setSelectedScript(script)}
                    className={`w-full rounded-md border p-2 text-left text-sm transition ${
                      selectedScript?.id === script.id
                        ? 'border-primary bg-primary/10'
                        : 'border-border bg-bg hover:bg-surface2'
                    }`}
                  >
                    <span className="block font-medium">{script.title}</span>
                    <span className="mt-1 flex flex-wrap items-center gap-2 text-xs text-text-muted">
                      <Badge variant={statusVariant(script.status)}>{script.status}</Badge>
                      <span>#{script.id}</span>
                      <span>Pack {script.primary_asset_pack_id}</span>
                    </span>
                  </button>
                ))}
              </div>
            </section>

            <form onSubmit={handleCreateScript} className="rounded-md border border-border bg-surface p-3">
              <h2 className="mb-3 text-sm font-semibold">Create script</h2>
              <div className="space-y-3">
                <div>
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <label className="block text-sm font-medium text-text" htmlFor="vn-script-template">
                      Starter template
                    </label>
                    {isLoadingTemplates && <span className="text-xs text-text-muted">Loading...</span>}
                  </div>
                  <select
                    id="vn-script-template"
                    className="block w-full rounded-md border-border bg-bg shadow-sm focus:border-primary focus:ring-primary"
                    value={selectedTemplateId}
                    onChange={(event) => handleTemplateChange(event.target.value)}
                  >
                    <option value="blank">Blank/custom JSON</option>
                    {templates.map((template) => (
                      <option key={template.id} value={template.id}>{template.label}</option>
                    ))}
                  </select>
                  {selectedTemplate && (
                    <p className="mt-1 text-xs text-text-muted">{selectedTemplate.description}</p>
                  )}
                </div>
                <Input label="Title" value={title} onChange={(event) => setTitle(event.target.value)} />
                <Input
                  label="Primary asset pack ID"
                  inputMode="numeric"
                  value={primaryAssetPackId}
                  onChange={(event) => setPrimaryAssetPackId(event.target.value)}
                />
                <Input
                  label="Policy profile ID"
                  value={policyProfileId}
                  onChange={(event) => setPolicyProfileId(event.target.value)}
                />
                <Input
                  label="Generation profile ID"
                  value={generationProfileId}
                  onChange={(event) => setGenerationProfileId(event.target.value)}
                />
                <label className="block text-sm font-medium text-text" htmlFor="vn-script-content-rating">
                  Content rating
                </label>
                <select
                  id="vn-script-content-rating"
                  className="block w-full rounded-md border-border bg-bg shadow-sm focus:border-primary focus:ring-primary"
                  value={contentRating}
                  onChange={(event) => setContentRating(event.target.value as VNScriptContentRating)}
                >
                  {contentRatings.map((rating) => (
                    <option key={rating} value={rating}>{rating}</option>
                  ))}
                </select>
                <Button type="submit" size="sm" loading={isCreating} className="w-full">Create script</Button>
              </div>
            </form>
          </aside>

          <section className="flex min-h-0 flex-col gap-3">
            <div className="rounded-md border border-border bg-surface p-3">
              {selectedScript ? (
                <>
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h2 className="text-lg font-semibold">{selectedScript.title}</h2>
                      <p className="text-sm text-text-muted">Script #{selectedScript.id}</p>
                    </div>
                    <Badge variant={statusVariant(selectedScript.status)}>{selectedScript.status}</Badge>
                  </div>
                  <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
                    {selectedMeta?.map(([label, value]) => (
                      <div key={label} className="rounded-md border border-border bg-bg p-2">
                        <dt className="text-xs uppercase text-text-muted">{label}</dt>
                        <dd className="mt-1 break-words font-medium">{value}</dd>
                      </div>
                    ))}
                  </dl>
                </>
              ) : (
                <p className="text-sm text-text-muted">Select or create a script to edit its draft.</p>
              )}
            </div>

            <div className="min-h-0 flex-1 rounded-md border border-border bg-surface p-3">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-sm font-semibold">Draft JSON</h2>
                  <p className="text-xs text-text-muted">
                    Revision {draft?.revision ?? 'n/a'}; JSON parsing only, opcode semantics stay on the API server.
                  </p>
                  {selectedGraphSourcePath && (
                    <div className="mt-2 rounded-md border border-primary/30 bg-primary/10 p-2 text-xs text-primary">
                      <span className="font-semibold">Selected graph path</span>
                      <code className="ml-2 break-all font-mono">{selectedGraphSourcePath}</code>
                    </div>
                  )}
                </div>
                <div className="flex gap-2">
                  <Button type="button" size="sm" variant="secondary" loading={isValidating} disabled={!selectedScript} onClick={handleValidate}>
                    Validate
                  </Button>
                  <Button type="button" size="sm" loading={isSavingDraft} disabled={!selectedScript || !draft} onClick={handleSaveDraft}>
                    Save draft
                  </Button>
                </div>
              </div>
              {(isLoadingCatalog || authoringCatalog || catalogStatus) && (
                <section className="mb-3 rounded-md border border-border bg-bg p-3">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                    <h2 className="text-sm font-semibold">Guided insert</h2>
                    {isLoadingCatalog && <span className="text-xs text-text-muted">Loading catalog...</span>}
                  </div>
                  {catalogStatus && (
                    <p className="rounded-md border border-warn/30 bg-warn/10 p-2 text-sm text-warn">
                      {catalogStatus}
                    </p>
                  )}
                  {authoringCatalog && (
                    <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(240px,0.9fr)]">
                      <div className="space-y-3">
                        {groupedSnippets.length === 0 ? (
                          <p className="text-sm text-text-muted">No snippets are available for the current backend capabilities.</p>
                        ) : (
                          groupedSnippets.map(([category, snippets]) => (
                            <div key={category}>
                              <h3 className="mb-1 text-xs font-semibold uppercase text-text-muted">{titleCase(category)}</h3>
                              <div className="flex flex-wrap gap-2">
                                {snippets.map((snippet) => (
                                  <Button
                                    key={snippet.id}
                                    type="button"
                                    size="xs"
                                    variant={selectedSnippetId === snippet.id ? 'primary' : 'secondary'}
                                    onClick={() => handleSnippetSelect(snippet)}
                                  >
                                    {snippet.label}
                                  </Button>
                                ))}
                              </div>
                            </div>
                          ))
                        )}
                      </div>

                      <div className="space-y-3">
                        {selectedSnippet ? (
                          <>
                            <div className="grid gap-2 sm:grid-cols-2">
                              {Object.entries(getParameterProperties(selectedSnippet)).map(([name, field]) => {
                                const label = field.title ?? titleCase(name);
                                const value = snippetParameters[name] ?? '';
                                const type = parameterType(field);
                                if (field.enum?.length) {
                                  return (
                                    <label key={name} className="block text-sm font-medium text-text">
                                      <span className="mb-1 block">{label}</span>
                                      <select
                                        aria-label={label}
                                        className="block w-full rounded-md border-border bg-bg shadow-sm focus:border-primary focus:ring-primary"
                                        value={selectedEnumOptionValue(field, value)}
                                        onChange={(event) => handleSnippetParameterChange(name, field, event.target.value)}
                                      >
                                        {field.enum.map((option, index) => (
                                          <option key={`${index}:${String(option)}`} value={enumOptionValue(index)}>{String(option)}</option>
                                        ))}
                                      </select>
                                    </label>
                                  );
                                }
                                if (type === 'boolean') {
                                  return (
                                    <label key={name} className="flex items-center gap-2 text-sm font-medium text-text">
                                      <input
                                        aria-label={label}
                                        type="checkbox"
                                        checked={Boolean(value)}
                                        onChange={(event) => handleSnippetParameterChange(name, field, event.target.checked)}
                                      />
                                      <span>{label}</span>
                                    </label>
                                  );
                                }
                                if (type === 'string' || type === 'number' || type === 'integer') {
                                  return (
                                    <Input
                                      key={name}
                                      label={label}
                                      type={type === 'string' ? 'text' : 'number'}
                                      value={type === 'string' ? String(value) : (value as number | '')}
                                      onChange={(event) => handleSnippetParameterChange(name, field, event.target.value)}
                                    />
                                  );
                                }
                                return (
                                  <label key={name} className="block text-sm font-medium text-text sm:col-span-2">
                                    <span className="mb-1 block">{label}</span>
                                    <textarea
                                      aria-label={label}
                                      className="min-h-24 w-full rounded-md border border-border bg-bg p-2 font-mono text-xs"
                                      value={typeof value === 'string' ? value : formatJson(value)}
                                      onChange={(event) => handleSnippetParameterChange(name, field, event.target.value)}
                                    />
                                  </label>
                                );
                              })}
                            </div>
                            <div
                              className={
                                snippetAnchor.mode === 'append'
                                  ? 'grid gap-2 sm:grid-cols-[minmax(0,1fr)_130px]'
                                  : 'grid gap-2 sm:grid-cols-[minmax(0,1fr)_130px_120px]'
                              }
                            >
                              <Input
                                label="Anchor label"
                                value={snippetAnchor.label}
                                onChange={(event) => handleSnippetAnchorChange({
                                  ...snippetAnchor,
                                  label: event.target.value,
                                })}
                              />
                              <label className="block text-sm font-medium text-text">
                                <span className="mb-1 block">Anchor mode</span>
                                <select
                                  className="block w-full rounded-md border-border bg-bg shadow-sm focus:border-primary focus:ring-primary"
                                  value={snippetAnchor.mode ?? 'append'}
                                  onChange={(event) => handleSnippetAnchorChange({
                                    ...snippetAnchor,
                                    mode: event.target.value as 'append' | 'before' | 'after',
                                    op_index: event.target.value === 'append' ? null : snippetAnchor.op_index ?? 0,
                                  })}
                                >
                                  <option value="append">append</option>
                                  <option value="before">before</option>
                                  <option value="after">after</option>
                                </select>
                              </label>
                              {snippetAnchor.mode !== 'append' && (
                                <Input
                                  label="Op index"
                                  type="number"
                                  min={0}
                                  value={String(snippetAnchor.op_index ?? 0)}
                                  onChange={(event) => {
                                    const nextIndex = Number(event.target.value);
                                    handleSnippetAnchorChange({
                                      ...snippetAnchor,
                                      op_index: Number.isInteger(nextIndex) && nextIndex >= 0 ? nextIndex : 0,
                                    });
                                  }}
                                />
                              )}
                            </div>
                            {snippetError && (
                              <p className="rounded-md border border-danger/30 bg-danger/10 p-2 text-sm text-danger">
                                {snippetError}
                              </p>
                            )}
                            <div className="flex flex-wrap gap-2">
                              <Button
                                type="button"
                                size="sm"
                                variant="secondary"
                                aria-busy={isPreviewingSnippet}
                                disabled={!selectedScript || !draft}
                                onClick={handlePreviewSnippet}
                              >
                                Preview snippet
                              </Button>
                              <Button
                                type="button"
                                size="sm"
                                loading={isApplyingSnippet}
                                disabled={!selectedScript || !draft || !hasCurrentSnippetPreview || hasUnsavedDraftText}
                                onClick={handleApplySnippet}
                              >
                                Apply snippet
                              </Button>
                            </div>
                          </>
                        ) : (
                          <p className="text-sm text-text-muted">Select a snippet to configure parameters.</p>
                        )}
                      </div>
                    </div>
                  )}
                  {snippetPreview && (
                    <div className="mt-3 grid gap-3 text-sm lg:grid-cols-2">
                      <div className="rounded-md border border-border bg-surface p-2">
                        <h3 className="text-xs font-semibold uppercase text-text-muted">Patch summary</h3>
                        <p className="mt-1">
                          Preview inserted {snippetPreview.patch_summary.inserted_ops} operations.
                        </p>
                        <p className="mt-1 break-words text-xs text-text-muted">
                          Changed paths: {snippetPreview.patch_summary.changed_paths.join(', ') || 'none'}
                        </p>
                      </div>
                      <div className="rounded-md border border-border bg-surface p-2">
                        <h3 className="text-xs font-semibold uppercase text-text-muted">Preview diagnostics</h3>
                        <pre className="mt-1 max-h-32 overflow-auto text-xs">
                          {summaryLines(snippetPreview.diagnostics).join('\n')}
                        </pre>
                      </div>
                    </div>
                  )}
                </section>
              )}
              <div ref={draftEditorRegionRef} className="min-h-[420px] flex-1">
                <JsonEditor
                  value={draftText}
                  onChange={(nextValue) => {
                    setDraftText(nextValue);
                    clearSnippetPreviewState();
                    clearGraphState();
                  }}
                  height="100%"
                  readOnly={!selectedScript}
                />
              </div>
            </div>
          </section>

          <aside className="flex min-h-0 flex-col gap-3 overflow-auto">
            <section className="rounded-md border border-border bg-surface p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold">Validation</h2>
                {validation && (
                  <Badge variant={validation.valid ? 'success' : 'danger'}>
                    {validation.valid ? 'Valid' : 'Invalid'}
                  </Badge>
                )}
              </div>
              {validation ? (
                <div className="space-y-3 text-sm">
                  <div>
                    <h3 className="mb-1 text-xs font-semibold uppercase text-text-muted">Errors</h3>
                    {validation.errors.length === 0 ? (
                      <p className="text-text-muted">No errors.</p>
                    ) : (
                      <ul className="space-y-1">
                        {validation.errors.map((item, index) => (
                          <li key={index} className="rounded-md bg-danger/10 p-2 text-danger">{formatJson(item)}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                  <div>
                    <h3 className="mb-1 text-xs font-semibold uppercase text-text-muted">Warnings</h3>
                    {validation.warnings.length === 0 ? (
                      <p className="text-text-muted">No warnings.</p>
                    ) : (
                      <ul className="space-y-1">
                        {validation.warnings.map((item, index) => (
                          <li key={index} className="rounded-md bg-warn/10 p-2 text-warn">{formatJson(item)}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              ) : (
                <p className="text-sm text-text-muted">Run backend validation to see script readiness.</p>
              )}
            </section>

            <section className="rounded-md border border-border bg-surface p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold">Diagnostics</h2>
                <Button type="button" size="xs" variant="secondary" loading={isLoadingDiagnostics} disabled={!selectedScript} onClick={handleDiagnostics}>
                  Diagnostics
                </Button>
              </div>
              {diagnostics ? (
                <pre className="max-h-56 overflow-auto rounded-md bg-bg p-2 text-xs">
                  {summaryLines(diagnostics.diagnostics).join('\n')}
                </pre>
              ) : (
                <p className="text-sm text-text-muted">Diagnostics are loaded on demand from the draft endpoint.</p>
              )}
            </section>

            {graphCapabilityEnabled && (
              <section className="rounded-md border border-border bg-surface p-3">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h2 className="text-sm font-semibold">Script graph</h2>
                    <p className="text-xs text-text-muted">Read-only outline from the backend graph API.</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="button"
                      size="xs"
                      variant="secondary"
                      loading={loadingGraphAction === 'saved'}
                      disabled={!selectedScript || loadingGraphAction !== null}
                      onClick={handleLoadDraftGraph}
                    >
                      Load saved graph
                    </Button>
                    <Button
                      type="button"
                      size="xs"
                      variant="secondary"
                      loading={loadingGraphAction === 'preview'}
                      disabled={!selectedScript || loadingGraphAction !== null}
                      onClick={handlePreviewDraftGraph}
                    >
                      Preview current JSON graph
                    </Button>
                  </div>
                </div>
                {graphError && (
                  <p className="mb-3 rounded-md border border-danger/30 bg-danger/10 p-2 text-sm text-danger">
                    {graphError}
                  </p>
                )}
                {scriptGraph ? (
                  <GraphSummary graph={scriptGraph} onSourcePathSelect={handleGraphSourcePathSelect} />
                ) : (
                  <p className="text-sm text-text-muted">
                    Load the saved draft graph or preview the current editor JSON without saving.
                  </p>
                )}
              </section>
            )}

            <section className="rounded-md border border-border bg-surface p-3">
              <h2 className="mb-3 text-sm font-semibold">Publish</h2>
              <div className="space-y-3">
                <Input label="Publish label" value={publishLabel} onChange={(event) => setPublishLabel(event.target.value)} />
                <p className="text-xs text-text-muted">
                  Publish acknowledgements are only sent when the backend exposes publish-safe codes.
                </p>
                <Button type="button" size="sm" loading={isPublishing} disabled={!selectedScript || !draft} onClick={handlePublish}>
                  Publish
                </Button>
              </div>
            </section>

            <section className="rounded-md border border-border bg-surface p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold">Published versions</h2>
                <Badge variant="neutral">{versions.length}</Badge>
              </div>
              <div className="space-y-3">
                {versions.length === 0 && <p className="text-sm text-text-muted">No published versions yet.</p>}
                {versions.map((version) => (
                  <article
                    key={version.id}
                    data-testid={`version-${version.id}`}
                    className="rounded-md border border-border bg-bg p-3"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <h3 className="font-medium">Version {version.version_number}</h3>
                        {version.label && <p className="text-sm text-text-muted">{version.label}</p>}
                      </div>
                      <Badge variant={validationBadge(version.validation) === 'Valid' ? 'success' : 'warning'}>
                        {validationBadge(version.validation)}
                      </Badge>
                    </div>
                    <dl className="mt-3 grid grid-cols-2 gap-2 text-xs">
                      <div><dt className="text-text-muted">Manifest</dt><dd>Manifest {version.manifest_snapshot_id}</dd></div>
                      <div><dt className="text-text-muted">Policy</dt><dd>Policy {version.policy_snapshot_id}</dd></div>
                      <div><dt className="text-text-muted">Generation</dt><dd>Generation {version.generation_profile_snapshot_id}</dd></div>
                      <div><dt className="text-text-muted">Created</dt><dd>{version.created_at}</dd></div>
                    </dl>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Button
                        type="button"
                        size="xs"
                        variant="secondary"
                        aria-label={`Load manifest for version ${version.version_number}`}
                        onClick={() => void handleManifestSnapshot(version)}
                      >
                        Manifest
                      </Button>
                      <Button
                        type="button"
                        size="xs"
                        variant="secondary"
                        aria-label={`Evaluate policy for version ${version.version_number}`}
                        onClick={() => void handlePolicyEvaluate(version)}
                      >
                        Policy
                      </Button>
                      {graphCapabilityEnabled && (
                        <Button
                          type="button"
                          size="xs"
                          variant="secondary"
                          loading={loadingVersionGraphId === version.id}
                          disabled={loadingVersionGraphId !== null}
                          aria-label={`Graph for version ${version.version_number}`}
                          onClick={() => void handleVersionGraph(version)}
                        >
                          Graph
                        </Button>
                      )}
                    </div>
                    {manifestSnapshots[version.id] && (
                      <div className="mt-3">
                        <h4 className="mb-1 text-xs font-semibold uppercase text-text-muted">Manifest summary</h4>
                        <pre className="max-h-48 overflow-auto rounded-md bg-surface2 p-2 text-xs">
                          {summaryLines(manifestSnapshots[version.id]).join('\n')}
                        </pre>
                      </div>
                    )}
                    {policySummaries[version.id] && (
                      <div className="mt-3">
                        <h4 className="mb-1 text-xs font-semibold uppercase text-text-muted">Policy summary</h4>
                        <pre className="max-h-48 overflow-auto rounded-md bg-surface2 p-2 text-xs">
                          {summaryLines(policySummaries[version.id]).join('\n')}
                        </pre>
                      </div>
                    )}
                    {versionGraphs[version.id] && (
                      <div className="mt-3">
                        <h4 className="mb-1 text-xs font-semibold uppercase text-text-muted">Version graph</h4>
                        <GraphSummary graph={versionGraphs[version.id]} onSourcePathSelect={handleGraphSourcePathSelect} />
                      </div>
                    )}
                  </article>
                ))}
              </div>
            </section>
          </aside>
        </div>
      </div>
    </main>
  );
}
