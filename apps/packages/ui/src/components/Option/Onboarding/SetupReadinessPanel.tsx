import React from "react";

import type {
  SetupReadinessLane,
  SetupReadinessStatusResponse,
} from "@/services/tldw/setup-readiness";

export type SetupReadinessPanelProps = {
  status: SetupReadinessStatusResponse | null;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
};

const REQUIRED_LANES = new Set(["chat", "embeddings_rag", "speech"]);
const OPTIONAL_LANES = new Set(["embeddings_rag", "speech"]);
const OPTIONAL_BLOCKING_OVERLAYS = new Set([
  "downloads_disabled",
  "package_installs_disabled",
  "network_unavailable",
  "remote_setup_blocked",
]);

const overlayLabels: Record<string, string> = {
  restart_required: "Restart required",
  admin_required: "Admin required",
  requires_admin: "Admin required",
  remote_setup_blocked: "Remote setup blocked",
  downloads_disabled: "Downloads disabled",
  package_installs_disabled: "Package installs disabled",
  network_unavailable: "Network unavailable",
};

const humanize = (value: string) =>
  value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

const statusCopy = (status: string | undefined) =>
  (status || "not_configured").replace(/_/g, " ");

const statusTone = (lane: SetupReadinessLane) => {
  if (lane.status === "ready" || lane.status === "ready_with_warnings") {
    return "border-success/30 bg-success/10 text-success";
  }
  if (lane.status === "failed" || lane.status === "blocked") {
    return lane.lane_id === "chat"
      ? "border-danger/30 bg-danger/10 text-danger"
      : "border-warn/30 bg-warn/10 text-warn";
  }
  if (lane.status === "provisioning" || lane.status === "previewed") {
    return "border-primary/30 bg-primary/10 text-primary";
  }
  return "border-border bg-surface2 text-text-muted";
};

const laneHasDetails = (lane: SetupReadinessLane) =>
  Boolean(
    lane.warnings?.length || lane.blockers?.length || lane.consequences?.length,
  );

const isOptionalDeferrable = (
  lane: SetupReadinessLane,
  overlays: string[],
) => {
  if (!OPTIONAL_LANES.has(lane.lane_id)) return false;
  if (lane.status === "not_configured" || lane.status === "skipped") {
    return true;
  }
  return Boolean(
    lane.status === "blocked" &&
      overlays.some((overlay) => OPTIONAL_BLOCKING_OVERLAYS.has(overlay)),
  );
};

const readinessLanes = (status: SetupReadinessStatusResponse | null) =>
  (status?.lanes ?? []).filter((lane) => REQUIRED_LANES.has(lane.lane_id));

const mergedOverlays = (status: SetupReadinessStatusResponse | null) =>
  Array.from(
    new Set([...(status?.active_overlays ?? []), ...(status?.overlays ?? [])]),
  );

const DetailList = ({
  title,
  items,
}: {
  title: string;
  items?: string[];
}) => {
  if (!items?.length) return null;
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-normal text-text-muted">
        {title}
      </dt>
      <dd className="mt-1 space-y-1">
        {items.map((item, index) => (
          <p key={`${item}-${index}`} className="text-xs text-text-muted">
            {item}
          </p>
        ))}
      </dd>
    </div>
  );
};

export function SetupReadinessPanel({
  status,
  loading = false,
  error = null,
  onRetry,
}: SetupReadinessPanelProps) {
  const lanes = readinessLanes(status);
  const overlays = mergedOverlays(status);
  const shouldRender = Boolean(status || loading || error || onRetry);

  if (!shouldRender) return null;

  return (
    <section
      aria-label="Setup readiness"
      data-testid="setup-readiness-panel"
      className="mb-4 rounded-md border border-border bg-surface px-3 py-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-text">Setup readiness</h2>
          <p className="text-xs text-text-muted">
            Chat must be ready. RAG and speech can be configured later.
          </p>
        </div>
        {onRetry ? (
          <button
            type="button"
            onClick={onRetry}
            className="rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs font-medium text-text hover:bg-surface2"
          >
            Retry
          </button>
        ) : null}
      </div>

      {loading ? (
        <p className="mt-3 text-xs text-text-muted">
          Checking setup readiness...
        </p>
      ) : null}

      {error ? (
        <div
          role="alert"
          className="mt-3 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger"
        >
          {error}
        </div>
      ) : null}

      {lanes.length > 0 ? (
        <div className="mt-3 grid gap-2 md:grid-cols-3">
          {lanes.map((lane) => {
            const deferrable = isOptionalDeferrable(lane, overlays);
            const blocksFirstChat =
              lane.lane_id === "chat" &&
              lane.status !== "ready" &&
              lane.status !== "ready_with_warnings";
            return (
              <div
                key={lane.lane_id}
                data-testid={`setup-readiness-lane-${lane.lane_id}`}
                className="rounded-md border border-border bg-bg px-3 py-2"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-sm font-medium text-text">
                      {lane.label || humanize(lane.lane_id)}
                    </p>
                    <p className="mt-0.5 text-xs text-text-muted">
                      {blocksFirstChat
                        ? "Blocks first chat"
                        : deferrable
                          ? "Deferrable"
                          : lane.lane_id === "chat"
                            ? "First-chat lane"
                            : "Optional lane"}
                    </p>
                  </div>
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${statusTone(
                      lane,
                    )}`}
                  >
                    {statusCopy(lane.status)}
                  </span>
                </div>

                {laneHasDetails(lane) ? (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-xs font-medium text-text-muted hover:text-text">
                      {lane.label || humanize(lane.lane_id)} details
                    </summary>
                    <dl className="mt-2 space-y-2">
                      <DetailList title="Warnings" items={lane.warnings} />
                      <DetailList title="Blockers" items={lane.blockers} />
                      <DetailList
                        title="Effects"
                        items={lane.consequences}
                      />
                    </dl>
                  </details>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : status ? (
        <p className="mt-3 text-xs text-text-muted">
          Lane readiness is not available yet.
        </p>
      ) : null}

      {overlays.length > 0 ? (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {overlays.map((overlay) => (
            <span
              key={overlay}
              className="rounded-full border border-warn/30 bg-warn/10 px-2 py-0.5 text-[10px] font-medium text-warn"
            >
              {overlayLabels[overlay] || humanize(overlay)}
            </span>
          ))}
        </div>
      ) : null}
    </section>
  );
}

export default SetupReadinessPanel;
