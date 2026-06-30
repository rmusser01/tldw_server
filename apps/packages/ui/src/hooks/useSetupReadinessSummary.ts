import React from "react";

import {
  getSetupReadinessProfiles,
  getSetupReadinessStatus,
  type SetupReadinessProfilesResponse,
  type SetupReadinessStatusResponse,
} from "@/services/tldw/setup-readiness";

const SETUP_READINESS_ERROR = "Setup readiness could not be loaded.";

const hasItems = <T,>(items?: T[]) => Array.isArray(items) && items.length > 0;

const mergeProfileFallbacks = (
  status: SetupReadinessStatusResponse,
  profiles: SetupReadinessProfilesResponse,
): SetupReadinessStatusResponse => {
  const merged: SetupReadinessStatusResponse = { ...status };
  if (!hasItems(merged.lane_ids) && hasItems(profiles.lane_ids)) {
    merged.lane_ids = profiles.lane_ids;
  }
  if (!hasItems(merged.lanes) && hasItems(profiles.lanes)) {
    merged.lanes = profiles.lanes;
  }
  if (
    !hasItems(merged.active_overlays) &&
    hasItems(profiles.active_overlays)
  ) {
    merged.active_overlays = profiles.active_overlays;
  }
  if (!hasItems(merged.overlays) && hasItems(profiles.overlays)) {
    merged.overlays = profiles.overlays;
  }
  if (!hasItems(merged.profiles) && hasItems(profiles.profiles)) {
    merged.profiles = profiles.profiles;
  }
  if (
    !hasItems(merged.supported_statuses) &&
    hasItems(profiles.supported_statuses)
  ) {
    merged.supported_statuses = profiles.supported_statuses;
  }
  if (
    !hasItems(merged.supported_overlays) &&
    hasItems(profiles.supported_overlays)
  ) {
    merged.supported_overlays = profiles.supported_overlays;
  }
  return merged;
};

export const useSetupReadinessSummary = () => {
  const [status, setStatus] =
    React.useState<SetupReadinessStatusResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const latestRequestId = React.useRef(0);
  const mounted = React.useRef(true);

  React.useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refresh = React.useCallback(async () => {
    const requestId = latestRequestId.current + 1;
    latestRequestId.current = requestId;
    setLoading(true);
    try {
      const [statusResult, profilesResult] = await Promise.allSettled([
        getSetupReadinessStatus({ mode: "first-run" }),
        getSetupReadinessProfiles({ mode: "first-run" }),
      ]);
      if (statusResult.status === "rejected") {
        throw statusResult.reason;
      }
      const nextSummary =
        profilesResult.status === "fulfilled"
          ? mergeProfileFallbacks(statusResult.value, profilesResult.value)
          : statusResult.value;
      if (mounted.current && latestRequestId.current === requestId) {
        setStatus(nextSummary);
        setError(null);
      }
      return nextSummary;
    } catch {
      if (mounted.current && latestRequestId.current === requestId) {
        setError(SETUP_READINESS_ERROR);
      }
      return null;
    } finally {
      if (mounted.current && latestRequestId.current === requestId) {
        setLoading(false);
      }
    }
  }, []);

  React.useEffect(() => {
    void refresh();
  }, [refresh]);

  return {
    status,
    loading,
    error,
    refresh,
  };
};

export type UseSetupReadinessSummaryResult = ReturnType<
  typeof useSetupReadinessSummary
>;
