import React from "react";
import { useTranslation } from "react-i18next";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { BetaTag } from "../Common/Beta";
import { XIcon } from "lucide-react";
import { getSettingsNavGroups, type SettingsNavItem } from "./settings-nav";
import {
  isSettingsNavItemActive,
  resolveCurrentSettingsNavItem,
} from "./settings-active-route";
import { isChromeTarget } from "@/config/platform";
import { isSidepanelSupported, openSidepanel } from "@/utils/sidepanel";
import { setSetting } from "@/services/settings/registry";
import { UI_MODE_SETTING } from "@/services/settings/ui-settings";
import { getSettingsReturnTo } from "@/utils/settings-return";
import {
  ACTION_ICON_CLICK_SETTING,
  CONTEXT_MENU_CLICK_SETTING,
} from "@/services/action";
import { useServerCapabilities } from "@/hooks/useServerCapabilities";

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(" ");
}

const shouldHideForBrowser = (item: SettingsNavItem) =>
  // Hide Chrome-specific settings on non-Chrome targets
  !isChromeTarget && item.to === "/settings/chrome";

const SETTINGS_HIDE_BETA_BADGES_STORAGE_KEY = "tldw:settings:hide-beta-badges";

const readHideBetaBadgesPreference = (): boolean => {
  try {
    return localStorage.getItem(SETTINGS_HIDE_BETA_BADGES_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
};

const writeHideBetaBadgesPreference = (hidden: boolean) => {
  try {
    if (hidden) {
      localStorage.setItem(SETTINGS_HIDE_BETA_BADGES_STORAGE_KEY, "1");
    } else {
      localStorage.removeItem(SETTINGS_HIDE_BETA_BADGES_STORAGE_KEY);
    }
  } catch {
    // localStorage may be unavailable
  }
};

export const SettingsLayout = ({ children }: { children: React.ReactNode }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { t } = useTranslation(["settings", "common"]);
  const [settingsFilterQuery, setSettingsFilterQuery] = React.useState("");
  const [hideBetaBadges, setHideBetaBadges] = React.useState<boolean>(
    readHideBetaBadgesPreference,
  );
  const { capabilities, loading: capabilitiesLoading } =
    useServerCapabilities();
  const sidepanelSupported = isSidepanelSupported();
  const settingsNavGroups = React.useMemo(
    () => getSettingsNavGroups(capabilitiesLoading ? undefined : capabilities),
    [capabilities, capabilitiesLoading],
  );
  const settingsNavItemCount = React.useMemo(
    () =>
      settingsNavGroups.reduce((count, group) => {
        return (
          count +
          group.items.filter((item) => !shouldHideForBrowser(item)).length
        );
      }, 0),
    [settingsNavGroups],
  );
  const showSettingsFilter = settingsNavItemCount > 12;
  const normalizedFilterQuery = settingsFilterQuery.trim().toLowerCase();
  const visibleSettingsNavGroups = React.useMemo(() => {
    if (!showSettingsFilter || !normalizedFilterQuery) {
      return settingsNavGroups;
    }
    return settingsNavGroups
      .map((group) => ({
        ...group,
        items: group.items.filter((item) => {
          const label = t(item.labelToken).toLowerCase();
          return label.includes(normalizedFilterQuery);
        }),
      }))
      .filter((group) => group.items.length > 0);
  }, [normalizedFilterQuery, settingsNavGroups, showSettingsFilter, t]);
  const currentNavItem = React.useMemo(() => {
    return resolveCurrentSettingsNavItem(location.pathname, settingsNavGroups);
  }, [location.pathname, settingsNavGroups]);

  const currentBreadcrumbLabel = currentNavItem
    ? t(currentNavItem.labelToken)
    : null;
  const routeHeadingLabel =
    location.pathname === "/settings/model"
      ? t("settings:modelSettings.heading", "Model settings")
      : location.pathname === "/settings"
        ? t("settings:heading", "Settings")
        : currentBreadcrumbLabel || t("settings:heading", "Settings");
  const hasVisibleBetaItems = React.useMemo(
    () =>
      settingsNavGroups.some((group) =>
        group.items.some((item) => !shouldHideForBrowser(item) && item.beta),
      ),
    [settingsNavGroups],
  );
  const toggleBetaBadgesVisibility = React.useCallback(() => {
    setHideBetaBadges((prev) => {
      const next = !prev;
      writeHideBetaBadgesPreference(next);
      return next;
    });
  }, []);

  return (
    <div className="flex min-h-screen w-full min-w-0 flex-col">
      <main className="relative w-full min-w-0 flex-1">
        <div className="mx-auto w-full h-full custom-scrollbar overflow-y-auto">
          <div className="flex min-w-0 flex-col lg:flex-row lg:gap-x-16 lg:px-24">
            <aside className="lg:sticky lg:mt-0 mt-14 lg:top-0 z-20 min-w-0 bg-surface border-b border-border lg:w-64 lg:shrink-0 lg:border-0 lg:bg-transparent">
              <nav
                className="w-full overflow-x-auto px-4 py-4 sm:px-6 lg:px-0 lg:py-0 lg:mt-20"
                aria-label={t(
                  "settings:navigation.ariaLabel",
                  "Settings navigation",
                )}
                data-testid="settings-navigation"
              >
                <div className="flex items-center justify-between mb-3">
                  <button
                    className="text-xs border rounded px-2 py-1 text-text  disabled:opacity-50 disabled:cursor-not-allowed"
                    disabled={!sidepanelSupported}
                    onClick={async () => {
                      await setSetting(UI_MODE_SETTING, "sidePanel");
                      await setSetting(ACTION_ICON_CLICK_SETTING, "sidePanel");
                      await setSetting(CONTEXT_MENU_CLICK_SETTING, "sidePanel");
                      try {
                        await openSidepanel();
                      } catch {}
                    }}
                    title={t("settings:switchToSidebar", "Switch to Sidebar")}
                  >
                    {t("settings:switchToSidebar", "Switch to Sidebar")}
                  </button>
                </div>
                {showSettingsFilter ? (
                  <div className="mb-3">
                    <label htmlFor="settings-nav-filter" className="sr-only">
                      {t(
                        "settings:navigation.filterLabel",
                        "Filter settings",
                      )}
                    </label>
                    <input
                      id="settings-nav-filter"
                      type="text"
                      value={settingsFilterQuery}
                      onChange={(event) =>
                        setSettingsFilterQuery(event.target.value)
                      }
                      placeholder={t(
                        "settings:navigation.filterPlaceholder",
                        "Find a setting",
                      )}
                      className="w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm text-text placeholder:text-text-subtle"
                      data-testid="settings-nav-filter"
                    />
                  </div>
                ) : null}
                <div className="flex flex-col gap-6">
                  {visibleSettingsNavGroups.map((group) => {
                    const items = group.items.filter(
                      (item) => !shouldHideForBrowser(item),
                    );
                    if (items.length === 0) {
                      return null;
                    }
                    return (
                      <div
                        key={group.key}
                        className="min-w-0"
                        data-testid={`settings-nav-group-${group.key}`}
                      >
                        <div className="mb-2 flex flex-col gap-1">
                          <span className="text-xs font-semibold uppercase tracking-wide text-text-muted ">
                            {t(group.titleToken)}
                          </span>
                        </div>
                        <ul
                          role="list"
                          className="flex flex-row flex-wrap gap-2 lg:flex-col"
                        >
                          {items.map((item) => {
                            const isActive = isSettingsNavItemActive(
                              location.pathname,
                              item.to,
                            );
                            return (
                              <li
                                key={item.to}
                                className="inline-flex items-center"
                              >
                                <Link
                                  to={item.to}
                                  className={classNames(
                                    isActive
                                      ? "border border-border bg-surface2 text-text"
                                      : "border border-transparent text-text-muted hover:text-text hover:bg-surface2",
                                    "group flex items-center gap-x-3 rounded-md py-2 pl-2 pr-3 text-sm font-semibold",
                                  )}
                                  aria-current={isActive ? "page" : undefined}
                                  data-testid={`settings-nav-link-${item.to.replace(
                                    /[^a-z0-9]+/gi,
                                    "-",
                                  )}`}
                                >
                                  <item.icon
                                    className={classNames(
                                      isActive
                                        ? "text-text"
                                        : "text-text-subtle group-hover:text-text",
                                      "h-6 w-6 shrink-0",
                                    )}
                                  />
                                  <span className="truncate">
                                    {t(item.labelToken)}
                                  </span>
                                  {isActive ? (
                                    <span
                                      className="h-1.5 w-1.5 rounded-full bg-primary"
                                      aria-hidden="true"
                                    />
                                  ) : null}
                                </Link>
                                {item.beta && !hideBetaBadges ? <BetaTag /> : null}
                              </li>
                            );
                          })}
                        </ul>
                      </div>
                    );
                  })}
                  {showSettingsFilter &&
                  normalizedFilterQuery &&
                  visibleSettingsNavGroups.length === 0 ? (
                    <p
                      className="rounded-md border border-border bg-surface2 px-3 py-2 text-xs text-text-muted"
                      data-testid="settings-nav-filter-empty"
                    >
                      {t(
                        "settings:navigation.filterEmpty",
                        "No settings match this search.",
                      )}
                    </p>
                  ) : null}
                </div>
                {hasVisibleBetaItems ? (
                  <div className="mt-4">
                    <button
                      type="button"
                      className="text-xs border rounded px-2 py-1 text-text-muted hover:text-text hover:bg-surface2"
                      onClick={toggleBetaBadgesVisibility}
                      data-testid="settings-beta-badges-toggle"
                    >
                      {hideBetaBadges
                        ? t("settings:showBetaBadges", "Show beta badges")
                        : t("settings:hideBetaBadges", "Hide beta badges")}
                    </button>
                  </div>
                ) : null}
              </nav>
            </aside>
            <main className="relative min-w-0 flex-1 px-4 py-8 sm:px-6 lg:px-0 lg:py-20">
              {/* Close button over right of content area */}
              <div className="absolute right-4 top-4 lg:right-0 lg:top-6 lg:translate-x-[-1rem]">
                <button
                  className="inline-flex items-center gap-1 text-xs border rounded px-2 py-1 text-text  hover:bg-surface2 "
                  title={t("common:close", "Close")}
                  onClick={(e) => {
                    e.preventDefault();
                    const returnTo = getSettingsReturnTo();
                    if (returnTo) {
                      navigate(returnTo);
                      return;
                    }
                    navigate("/");
                  }}
                >
                  <XIcon className="h-4 w-4" />
                  <span>{t("common:close", "Close")}</span>
                </button>
              </div>
              <div className="mx-auto w-full min-w-0 max-w-4xl space-y-8 sm:space-y-10">
                <h1 className="text-2xl font-semibold text-text">
                  {routeHeadingLabel}
                </h1>
                {currentBreadcrumbLabel ? (
                  <div
                    className="rounded-md border border-border bg-surface2 px-3 py-2"
                    data-testid="settings-current-section"
                  >
                    <p
                      className="text-xs text-text-muted"
                      aria-label={t(
                        "settings:breadcrumb.ariaLabel",
                        "Current settings location",
                      )}
                    >
                      <span className="font-semibold text-text">
                        {t("settings:currentSectionLabel", "Current section")}
                      </span>
                      <span className="mx-1 text-text-muted">:</span>
                      <span>{currentBreadcrumbLabel}</span>
                    </p>
                  </div>
                ) : null}
                {children}
              </div>
            </main>
          </div>
        </div>
      </main>
    </div>
  );
};
