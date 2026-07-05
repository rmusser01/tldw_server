import React from "react";
import { Boxes, ExternalLink } from "lucide-react";
import { Link } from "react-router-dom";

import type {
  McpToolsApplyRequest,
  McpToolsApplyResponse,
  McpToolsCatalogResponse,
  McpToolsValidateRequest,
  McpToolsValidateResponse,
} from "@/types/setup-onboarding";

export type McpToolsStepProps = {
  catalog: McpToolsCatalogResponse | null;
  initialStepData?: Record<string, unknown> | null;
  loadCatalog: () => Promise<McpToolsCatalogResponse>;
  applyMcpTools: (
    payload: McpToolsApplyRequest,
  ) => Promise<McpToolsApplyResponse>;
  validateMcpTools: (
    payload: McpToolsValidateRequest,
  ) => Promise<McpToolsValidateResponse>;
  skipPending?: boolean;
  onContinue: () => void;
  onBack: () => void;
  onSkip: () => void;
};

const hubHref = (profileId?: number | null) => {
  const params = new URLSearchParams({ source: "first-run" });
  if (profileId) params.set("profile_id", String(profileId));
  return `/mcp-hub?${params.toString()}`;
};

const toggle = (items: string[], id: string) =>
  items.includes(id) ? items.filter((item) => item !== id) : [...items, id];

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string");

const asNumberOrNull = (value: unknown) =>
  typeof value === "number" ? value : null;

const CONFLICT_REASON_LABELS: Record<string, string> = {
  profile_manually_changed: "Generated profile was changed in MCP Hub.",
  profile_conflict: "MCP Hub profile needs review.",
};

const VALIDATION_STATE_LABELS: Record<string, string> = {
  not_run: "Not run",
  built_in_passed: "Built-in check passed",
  external_discovered: "External tools discovered",
  external_tool_passed: "Sample external tool passed",
  no_safe_external_tool: "No safe sample tool available",
  external_discovery_incomplete: "External discovery incomplete",
  failed: "Failed",
  skipped: "Skipped",
};

const isValidationState = (
  value: unknown,
): value is McpToolsApplyResponse["validation_state"] =>
  typeof value === "string" && value in VALIDATION_STATE_LABELS;

const EXTERNAL_STATUS_LABELS: Record<string, string> = {
  not_configured: "Not configured",
  ready: "Ready",
  unavailable: "Unavailable",
  incomplete: "Incomplete",
};

const humanizeToken = (value: string) =>
  value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const labelFromMap = (value: string | null | undefined, labels: Record<string, string>) =>
  value ? labels[value] ?? humanizeToken(value) : null;

const labelsForIds = (
  ids: string[],
  catalog: McpToolsCatalogResponse | null,
): string[] =>
  ids.map(
    (id) =>
      catalog?.packs.find((pack) => pack.pack_id === id)?.label ??
      catalog?.add_ons.find((addon) => addon.addon_id === id)?.label ??
      id,
  );

export function McpToolsStep({
  catalog,
  initialStepData = null,
  loadCatalog,
  applyMcpTools,
  validateMcpTools,
  skipPending = false,
  onContinue,
  onBack,
  onSkip,
}: McpToolsStepProps) {
  const [activeCatalog, setActiveCatalog] =
    React.useState<McpToolsCatalogResponse | null>(catalog);
  const [selectedPacks, setSelectedPacks] = React.useState<string[]>([]);
  const [selectedAddOns, setSelectedAddOns] = React.useState<string[]>([]);
  const [confirmedAddOns, setConfirmedAddOns] = React.useState<string[]>([]);
  const [applyResult, setApplyResult] =
    React.useState<McpToolsApplyResponse | null>(null);
  const [validationResult, setValidationResult] =
    React.useState<McpToolsValidateResponse | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [validating, setValidating] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const initializedRef = React.useRef(false);

  const clearSavedResults = () => {
    setApplyResult(null);
    setValidationResult(null);
  };

  React.useEffect(() => {
    if (catalog) setActiveCatalog(catalog);
  }, [catalog]);

  React.useEffect(() => {
    if (activeCatalog || catalog) return;
    void loadCatalog()
      .then(setActiveCatalog)
      .catch((err) => {
        console.error("MCP tools catalog could not be loaded", err);
        setError("MCP tools catalog could not be loaded.");
      });
  }, [activeCatalog, catalog, loadCatalog]);

  React.useEffect(() => {
    if (!activeCatalog || initializedRef.current) return;
    initializedRef.current = true;
    const savedPackIds = initialStepData?.selected_pack_ids;
    const savedAddonIds = initialStepData?.selected_addon_ids;
    const selectedPackIds = isStringArray(savedPackIds)
      ? savedPackIds
      : activeCatalog.packs
          .filter((pack) => pack.default_selected)
          .map((pack) => pack.pack_id);
    const selectedAddonIds = isStringArray(savedAddonIds)
      ? savedAddonIds
      : activeCatalog.add_ons
          .filter((addon) => addon.default_selected)
          .map((addon) => addon.addon_id);
    const confirmedAddonIds = isStringArray(initialStepData?.confirmed_addon_ids)
      ? initialStepData.confirmed_addon_ids
      : [];
    setSelectedPacks(selectedPackIds);
    setSelectedAddOns(selectedAddonIds);
    setConfirmedAddOns(confirmedAddonIds);

    if (
      initialStepData?.acknowledged === true &&
      initialStepData.validation_state !== "skipped" &&
      isStringArray(savedPackIds)
    ) {
      const effectiveTools = isStringArray(initialStepData.effective_tools)
        ? initialStepData.effective_tools
        : activeCatalog.packs
            .filter((pack) => selectedPackIds.includes(pack.pack_id))
            .flatMap((pack) => pack.available_tools.map((tool) => tool.tool_name));
      setApplyResult({
        status: "applied",
        profile_id: asNumberOrNull(initialStepData.profile_id),
        assignment_id: asNumberOrNull(initialStepData.assignment_id),
        catalog_version:
          typeof initialStepData.catalog_version === "string"
            ? initialStepData.catalog_version
            : activeCatalog.catalog_version,
        selected_pack_ids: selectedPackIds,
        selected_addon_ids: selectedAddonIds,
        effective_tool_count:
          typeof initialStepData.effective_tool_count === "number"
            ? initialStepData.effective_tool_count
            : effectiveTools.length,
        effective_tools: effectiveTools,
        disabled_addons: isStringArray(initialStepData.disabled_addons)
          ? initialStepData.disabled_addons
          : [],
        validation_state: isValidationState(initialStepData.validation_state)
          ? initialStepData.validation_state
          : "not_run",
        conflict: null,
      });
    }
  }, [activeCatalog, initialStepData]);

  const selectedStrongAddOns =
    activeCatalog?.add_ons.filter(
      (addon) =>
        addon.strong_confirmation &&
        selectedAddOns.includes(addon.addon_id) &&
        !confirmedAddOns.includes(addon.addon_id),
    ) ?? [];
  const needsConfirmation = selectedStrongAddOns.length > 0;
  const saved =
    Boolean(applyResult) && applyResult?.status !== "conflict" && !saving;
  const conflict = applyResult?.status === "conflict" ? applyResult : null;

  const buildPayload = (
    extra: Partial<McpToolsApplyRequest> = {},
  ): McpToolsApplyRequest => ({
    selected_pack_ids: selectedPacks,
    selected_addon_ids: selectedAddOns,
    confirmed_addon_ids: confirmedAddOns.filter((id) =>
      selectedAddOns.includes(id),
    ),
    confirmation_version: activeCatalog?.confirmation_version ?? null,
    ...extra,
  });

  const handleApply = async (extra: Partial<McpToolsApplyRequest> = {}) => {
    setSaving(true);
    setError(null);
    setValidationResult(null);
    try {
      const result = await applyMcpTools(buildPayload(extra));
      setApplyResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "MCP tools were not saved.");
    } finally {
      setSaving(false);
    }
  };

  const handleValidate = async () => {
    setValidating(true);
    setError(null);
    try {
      setValidationResult(await validateMcpTools({}));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sample tool failed.");
    } finally {
      setValidating(false);
    }
  };

  const profileId =
    validationResult?.profile_id ??
    applyResult?.profile_id ??
    applyResult?.conflict?.profile_id ??
    null;
  const selectedPackLabels = labelsForIds(
    applyResult?.selected_pack_ids ?? selectedPacks,
    activeCatalog,
  );
  const effectiveToolCount =
    validationResult?.effective_tool_count ?? applyResult?.effective_tool_count;
  const validationLabel = labelFromMap(
    validationResult?.validation_state ?? applyResult?.validation_state,
    VALIDATION_STATE_LABELS,
  );
  const externalStatusLabel = labelFromMap(
    validationResult?.external_status,
    EXTERNAL_STATUS_LABELS,
  );

  return (
    <section aria-labelledby="mcp-tools-title" className="space-y-5">
      <div className="flex items-start gap-3">
        <span className="inline-flex size-10 items-center justify-center rounded-md bg-surface2 text-primary">
          <Boxes className="size-5" aria-hidden="true" />
        </span>
        <div>
          <h2 id="mcp-tools-title" className="text-lg font-semibold text-text">
            MCP tools
          </h2>
          <p className="mt-1 text-sm text-text-muted">
            Choose tool packs for chat and research workflows.
          </p>
        </div>
      </div>

      {error ? (
        <div
          role="alert"
          className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          {error}
        </div>
      ) : null}

      <fieldset className="space-y-2">
        <legend className="text-sm font-medium text-text">Packs</legend>
        <div className="grid gap-2 md:grid-cols-2">
          {(activeCatalog?.packs ?? []).map((pack) => (
            <label
              key={pack.pack_id}
              className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
            >
              <input
                type="checkbox"
                checked={selectedPacks.includes(pack.pack_id)}
                onChange={() => {
                  clearSavedResults();
                  setSelectedPacks(toggle(selectedPacks, pack.pack_id));
                }}
                disabled={!pack.available || saving}
                className="mr-2"
              />
              <span className="font-medium">{pack.label}</span>
              <span className="ml-2 text-text-muted">{pack.purpose}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <details
        data-testid="mcp-tools-addons"
        className="rounded-md border border-border bg-surface px-3 py-2"
      >
        <summary className="cursor-pointer text-sm font-medium text-text">
          Add-ons
        </summary>
        <div className="mt-3 space-y-2">
          {(activeCatalog?.add_ons ?? []).map((addon) => {
            const selected = selectedAddOns.includes(addon.addon_id);
            return (
              <div key={addon.addon_id} className="space-y-2">
                <label className="block text-sm text-text">
                  <input
                    type="checkbox"
                    checked={selected}
                    onChange={() => {
                      clearSavedResults();
                      setSelectedAddOns(toggle(selectedAddOns, addon.addon_id));
                      setConfirmedAddOns(
                        confirmedAddOns.filter((id) => id !== addon.addon_id),
                      );
                    }}
                    disabled={saving}
                    className="mr-2"
                  />
                  <span className="font-medium">{addon.label}</span>
                  <span className="ml-2 text-text-muted">
                    {addon.requirement}
                  </span>
                </label>
                {selected && addon.strong_confirmation ? (
                  <label className="ml-6 block text-sm text-text">
                    <input
                      type="checkbox"
                      aria-label={`Confirm ${addon.label}`}
                      checked={confirmedAddOns.includes(addon.addon_id)}
                      onChange={() => {
                        clearSavedResults();
                        setConfirmedAddOns(toggle(confirmedAddOns, addon.addon_id));
                      }}
                      disabled={saving}
                      className="mr-2"
                    />
                    Confirm {addon.label}
                  </label>
                ) : null}
              </div>
            );
          })}
        </div>
      </details>

      {needsConfirmation ? (
        <div className="rounded-md border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-text">
          Confirm selected strong add-ons before saving.
        </div>
      ) : null}

      {conflict ? (
        <div
          role="alert"
          className="rounded-md border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-text"
        >
          <p className="font-medium">MCP Hub profile changed</p>
          <p className="mt-1 text-text-muted">
            {labelFromMap(
              conflict.conflict?.reason ?? "profile_conflict",
              CONFLICT_REASON_LABELS,
            )}
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() =>
                void handleApply({
                  conflict_resolution: "keep_existing",
                  profile_id: conflict.conflict?.profile_id ?? conflict.profile_id,
                })
              }
              disabled={saving}
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
            >
              Keep existing
            </button>
            <button
              type="button"
              onClick={() =>
                void handleApply({
                  conflict_resolution: "replace_existing",
                  profile_id: conflict.conflict?.profile_id ?? conflict.profile_id,
                })
              }
              disabled={saving}
              className="rounded-md bg-primary px-3 py-1.5 text-sm font-semibold text-primary-foreground disabled:opacity-50"
            >
              Replace generated profile
            </button>
            <Link
              to={hubHref(conflict.conflict?.profile_id ?? conflict.profile_id)}
              className="inline-flex items-center gap-1 rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text hover:bg-surface2"
            >
              Open MCP Hub
              <ExternalLink className="size-3.5" aria-hidden="true" />
            </Link>
          </div>
        </div>
      ) : null}

      {applyResult && applyResult.status !== "conflict" ? (
        <div className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-text">
          <div className="font-medium">Enabled packs</div>
          <div className="mt-1 text-text-muted">
            {selectedPackLabels.length ? selectedPackLabels.join(", ") : "None"}
          </div>
          <div className="mt-3 font-medium">Available tools</div>
          <div className="mt-1 text-text-muted">
            {typeof effectiveToolCount === "number"
              ? `${effectiveToolCount} tools`
              : "Not checked"}
          </div>
          {applyResult.effective_tools.length ? (
            <div className="mt-1 text-text-muted">
              {applyResult.effective_tools.join(", ")}
            </div>
          ) : null}
          {applyResult.disabled_addons.length ? (
            <div className="mt-3 text-text-muted">
              Disabled add-ons:{" "}
              {labelsForIds(applyResult.disabled_addons, activeCatalog).join(", ")}
            </div>
          ) : null}
          {validationLabel ? (
            <div className="mt-3 text-text-muted">
              Validation: {validationLabel}
            </div>
          ) : null}
          {validationResult?.validation_message ? (
            <div className="mt-1 text-text-muted">
              {validationResult.validation_message}
            </div>
          ) : null}
          {externalStatusLabel ? (
            <div className="mt-1 text-text-muted">
              External status: {externalStatusLabel}
            </div>
          ) : null}
          <Link
            to={hubHref(profileId)}
            className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-primary"
          >
            Open MCP Hub
            <ExternalLink className="size-3.5" aria-hidden="true" />
          </Link>
        </div>
      ) : null}

      <div className="flex flex-wrap justify-between gap-2">
        <button
          type="button"
          onClick={onBack}
          disabled={saving || validating}
          className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
        >
          Back
        </button>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={onSkip}
            disabled={saving || validating || skipPending}
            className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
          >
            {skipPending ? "Skipping MCP tools..." : "Skip MCP tools"}
          </button>
          <button
            type="button"
            onClick={() => void handleApply()}
            disabled={!activeCatalog || saving || validating || needsConfirmation}
            className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save packs"}
          </button>
          <button
            type="button"
            onClick={handleValidate}
            disabled={!saved || saving || validating}
            className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {validating ? "Running..." : "Run sample tool"}
          </button>
          <button
            type="button"
            onClick={onContinue}
            disabled={!saved || saving || validating}
            className="rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            Continue
          </button>
        </div>
      </div>
    </section>
  );
}
