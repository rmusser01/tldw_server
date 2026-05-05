import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button, Input, message } from "antd";
import { Check } from "lucide-react";
import {
  useConnectionState,
  useConnectionUxState,
  useConnectionActions,
} from "@/hooks/useConnectionState";
import { ConnectionPhase } from "@/types/connection";
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient";
import {
  RecoveryCallout,
  type RecoveryState,
  type StateAction,
} from "@/components/ui/state";

type ConnectionBannerProps = {
  className?: string;
};

/**
 * Connection status banner displayed below the header when not connected.
 * Shows contextual messages and actions based on connection state.
 *
 * States:
 * - Connecting: Shows spinner with "Connecting..." message
 * - Auth error: Shows key icon with "API key needs attention" message
 * - Unreachable: Shows wifi-off icon with "Can't reach server" message
 * - Unconfigured: Shows settings icon with "Set up connection" message
 */
export const ConnectionBanner: React.FC<ConnectionBannerProps> = ({
  className,
}) => {
  const { t } = useTranslation(["sidepanel", "settings", "common"]);
  const { phase, isConnected, serverUrl } = useConnectionState();
  const { uxState, isChecking, hasCompletedFirstRun } = useConnectionUxState();
  const { checkOnce, setConfigPartial } = useConnectionActions();

  // Inline API key form state
  const [showApiKeyForm, setShowApiKeyForm] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [authMode, setAuthMode] =
    useState<TldwConfig["authMode"]>("single-user");

  useEffect(() => {
    let cancelled = false;

    void tldwClient
      .getConfig()
      .then((config) => {
        if (!cancelled && config?.authMode) {
          setAuthMode(config.authMode);
        }
      })
      .catch(() => undefined);

    return () => {
      cancelled = true;
    };
  }, []);

  // Don't show banner if connected
  const isConnectionReady = isConnected && phase === ConnectionPhase.CONNECTED;
  if (isConnectionReady) {
    return null;
  }

  const canInlineRepairApiKey =
    uxState === "error_auth" && authMode === "single-user";

  const openSettings = () => {
    try {
      if (typeof chrome !== "undefined" && chrome.runtime?.openOptionsPage) {
        chrome.runtime.openOptionsPage();
        return;
      }
    } catch {}
    window.open("/options.html#/settings/tldw", "_blank");
  };

  const handleRetry = () => {
    void checkOnce();
  };

  const handleSaveApiKey = async () => {
    if (!apiKeyInput.trim()) {
      message.error(
        t(
          "sidepanel:connectionBanner.apiKeyRequired",
          "Please enter an API key",
        ),
      );
      return;
    }

    setIsSaving(true);
    try {
      await setConfigPartial({
        apiKey: apiKeyInput.trim(),
      });
      message.success(
        t("sidepanel:connectionBanner.apiKeySaved", "API key saved"),
      );
      setShowApiKeyForm(false);
      setApiKeyInput("");
      void checkOnce();
    } catch {
      message.error(
        t(
          "sidepanel:connectionBanner.apiKeySaveError",
          "Failed to save API key",
        ),
      );
    } finally {
      setIsSaving(false);
    }
  };

  // Determine banner content based on state
  const getBannerConfig = () => {
    if (isChecking || uxState === "testing") {
      return {
        state: "retrying" as RecoveryState,
        message: t(
          "sidepanel:connectionBanner.connecting",
          "Connecting to your tldw server...",
        ),
        description: null,
        showRetry: false,
        showSettings: false,
      };
    }

    if (uxState === "error_auth") {
      return {
        state: "auth_required" as RecoveryState,
        message: t(
          "sidepanel:connectionBanner.authErrorTitle",
          "API key needs attention",
        ),
        description: t(
          "sidepanel:connectionBanner.authErrorBody",
          "Your server is reachable but the API key is wrong or missing.",
        ),
        showRetry: true,
        showSettings: true,
      };
    }

    if (uxState === "error_unreachable") {
      return {
        state: "unavailable" as RecoveryState,
        message: t(
          "sidepanel:connectionBanner.unreachableTitle",
          "Can't reach your tldw server",
        ),
        description: serverUrl
          ? t(
              "sidepanel:connectionBanner.unreachableBody",
              "Check that your server is running and accessible.",
            )
          : t(
              "sidepanel:connectionBanner.noUrlBody",
              "Add your server URL in Settings to get started.",
            ),
        showRetry: !!serverUrl,
        showSettings: true,
      };
    }

    // Default: unconfigured or unknown state
    return {
      state: "setup_required" as RecoveryState,
      message: hasCompletedFirstRun
        ? t(
            "sidepanel:connectionBanner.disconnectedTitle",
            "Not connected to tldw server",
          )
        : t(
            "sidepanel:connectionBanner.setupTitle",
            "Finish setup to start chatting",
          ),
      description: hasCompletedFirstRun
        ? t(
            "sidepanel:connectionBanner.disconnectedBody",
            "Open Settings to configure your server connection.",
          )
        : t(
            "sidepanel:connectionBanner.setupBody",
            "Complete the setup wizard in Settings to connect.",
          ),
      showRetry: false,
      showSettings: true,
    };
  };

  const config = getBannerConfig();
  const showInlineApiKeyForm = canInlineRepairApiKey && showApiKeyForm;

  const primaryAction: StateAction = (() => {
    if (canInlineRepairApiKey && !showApiKeyForm) {
      return {
        label: t("sidepanel:connectionBanner.enterApiKey", "Enter API Key"),
        onClick: () => setShowApiKeyForm(true),
      };
    }

    if (config.showRetry) {
      return {
        label: t("common:retry", "Retry"),
        onClick: handleRetry,
        loading: isChecking,
      };
    }

    if (config.showSettings) {
      return {
        label: t("sidepanel:connectionBanner.openSettings", "Open Settings"),
        onClick: openSettings,
      };
    }

    return {
      label: t("sidepanel:connectionBanner.connectingCta", "Connecting"),
      disabled: true,
    };
  })();

  const secondaryActionCandidates: Array<StateAction | null> = [
    canInlineRepairApiKey && !showApiKeyForm && config.showRetry
      ? {
          label: t("common:retry", "Retry"),
          onClick: handleRetry,
          loading: isChecking,
        }
      : null,
    config.showSettings && (config.showRetry || canInlineRepairApiKey)
      ? {
          label: t("sidepanel:connectionBanner.openSettings", "Open Settings"),
          onClick: openSettings,
        }
      : null,
  ];
  const secondaryActions = secondaryActionCandidates.filter(
    (action): action is StateAction => Boolean(action),
  );

  return (
    <div className={`px-3 py-2 ${className || ""}`}>
      <RecoveryCallout
        state={config.state}
        title={config.message}
        message={config.description ?? undefined}
        primaryAction={primaryAction}
        secondaryActions={secondaryActions}
        data-testid="sidepanel-connection-banner"
      >
        {showInlineApiKeyForm ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Input.Password
                size="small"
                placeholder={t(
                  "sidepanel:connectionBanner.apiKeyPlaceholder",
                  "Enter your API key",
                )}
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                onPressEnter={handleSaveApiKey}
                visibilityToggle={{
                  visible: showApiKey,
                  onVisibleChange: setShowApiKey,
                }}
                className="flex-1"
                autoFocus
              />
              <Button
                size="small"
                type="primary"
                icon={<Check className="size-3" />}
                onClick={handleSaveApiKey}
                loading={isSaving}
                title={t("common:save", "Save")}
              >
                {t("common:save", "Save")}
              </Button>
            </div>
            <button
              type="button"
              onClick={() => {
                setShowApiKeyForm(false);
                setApiKeyInput("");
              }}
              className="text-xs text-text-subtle hover:text-text underline"
              title={t("common:cancel", "Cancel")}
            >
              {t("common:cancel", "Cancel")}
            </button>
          </div>
        ) : null}
      </RecoveryCallout>
    </div>
  );
};

export default ConnectionBanner;
