import OptionLayout from "~/components/Layouts/Layout"
import MediaReviewPage from "@/components/Review/MediaReviewPage"
import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { useConnectionUxState } from "@/hooks/useConnectionState"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useDemoMode } from "@/context/demo-mode"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"

const MediaMultiInner = () => {
  const { t } = useTranslation(["review", "common", "settings"])
  const navigate = useNavigate()
  const isOnline = useServerOnline()
  const { demoEnabled } = useDemoMode()
  const { uxState, hasCompletedFirstRun } = useConnectionUxState()
  const { capabilities, loading: capsLoading } = useServerCapabilities()

  const demoMediaItems = React.useMemo(
    () => [
      {
        id: "demo-media-1",
        title: t("review:mediaEmpty.demoSample1Title", {
          defaultValue: "Demo media: Team call recording"
        }),
        meta: t("review:mediaEmpty.demoSample1Meta", {
          defaultValue: "Video · 25 min · Keywords: standup, planning"
        })
      },
      {
        id: "demo-media-2",
        title: t("review:mediaEmpty.demoSample2Title", {
          defaultValue: "Demo media: Product walkthrough"
        }),
        meta: t("review:mediaEmpty.demoSample2Meta", {
          defaultValue: "Screen recording · 12 min · Keywords: onboarding"
        })
      },
      {
        id: "demo-media-3",
        title: t("review:mediaEmpty.demoSample3Title", {
          defaultValue: "Demo media: Research article PDF"
        }),
        meta: t("review:mediaEmpty.demoSample3Meta", {
          defaultValue: "PDF · 6 pages · Keywords: summarization"
        })
      }
    ],
    [t]
  )

  const demoConnectionWarning = (() => {
    if (uxState === "error_auth" || uxState === "configuring_auth") {
      return "Demo stays available, but your Media credentials need attention."
    }
    if (uxState === "unconfigured" || uxState === "configuring_url") {
      return "Demo stays available while you finish Media setup."
    }
    if (uxState === "error_unreachable") {
      return "Demo stays available, but your tldw server is unreachable."
    }
    return null
  })()

  if (!isOnline) {
    return demoEnabled ? (
      <div className="space-y-4">
        {demoConnectionWarning ? (
          <div
            data-testid="media-demo-connection-warning"
            className="rounded-2xl border border-warn/40 bg-warn/10 px-4 py-3 text-sm text-text"
          >
            <div className="font-medium">{demoConnectionWarning}</div>
          </div>
        ) : null}
        <FeatureEmptyState
          title={t("review:mediaEmpty.demoTitle", {
            defaultValue: "Explore Media in demo mode"
          })}
          description={t("review:mediaEmpty.demoDescription", {
            defaultValue:
              "This demo shows how tldw can display and inspect processed media. Connect your own server later to browse your own recordings and documents."
          })}
          examples={[
            t("review:mediaEmpty.demoExample1", {
              defaultValue: "See how processed media items appear in the Media viewer."
            }),
            t("review:mediaEmpty.demoExample2", {
              defaultValue:
                "When you connect, you’ll be able to browse and inspect media ingested from your own recordings and files."
            }),
            t("review:mediaEmpty.demoExample3", {
              defaultValue:
                "Use Media together with Review to summarize or analyze recordings."
            })
          ]}
          primaryActionLabel={t("settings:tldw.setupLink", "Set up server")}
          onPrimaryAction={() => navigate("/settings/tldw")}
          secondaryActionLabel={t(
            "settings:healthSummary.diagnostics",
            "Health & diagnostics"
          )}
          onSecondaryAction={() => navigate("/settings/health")}
        />
        <div className="rounded-lg border border-dashed border-border bg-surface p-3 text-xs text-text">
          <div className="mb-2 font-semibold">
            {t("review:mediaEmpty.demoPreviewHeading", {
              defaultValue: "Example media items (preview only)"
            })}
          </div>
          <div className="divide-y divide-border">
            {demoMediaItems.map((item) => (
              <div key={item.id} className="py-2">
                <div className="text-sm font-medium text-text">
                  {item.title}
                </div>
                <div className="mt-1 text-[11px] text-text-muted">
                  {item.meta}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    ) : (
      (() => {
        if (uxState === "error_auth" || uxState === "configuring_auth") {
          return (
            <FeatureEmptyState
              title="Add your credentials to use Media"
              description="Media Review needs a reachable tldw server plus valid credentials before your recordings and documents can load."
              primaryActionLabel="Open Settings"
              onPrimaryAction={() => navigate("/settings/tldw")}
            />
          )
        }
        if (uxState === "unconfigured" || uxState === "configuring_url") {
          return (
            <FeatureEmptyState
              title="Finish setup to use Media"
              description={
                hasCompletedFirstRun
                  ? "Media Review still needs a configured tldw server before recordings and documents can load."
                  : "Finish connecting your tldw server before Media Review can load your recordings and documents."
              }
              primaryActionLabel={hasCompletedFirstRun ? "Open Settings" : "Finish Setup"}
              onPrimaryAction={() =>
                navigate(hasCompletedFirstRun ? "/settings/tldw" : "/")
              }
            />
          )
        }
        if (uxState === "error_unreachable") {
          return (
            <FeatureEmptyState
              title="Can't reach your tldw server right now"
              description="Media Review depends on a reachable tldw server. Review your server status and URL before trying again."
              primaryActionLabel={t(
                "settings:healthSummary.diagnostics",
                "Health & diagnostics"
              )}
              onPrimaryAction={() => navigate("/settings/health")}
              secondaryActionLabel="Open Settings"
              onSecondaryAction={() => navigate("/settings/tldw")}
            />
          )
        }
        return (
          <FeatureEmptyState
            title={t("review:mediaEmpty.connectTitle", {
              defaultValue: "Connect to use Media"
            })}
            description={t("review:mediaEmpty.connectDescription", {
              defaultValue:
                "To view processed media, first connect to your tldw server so recordings and documents can be listed here."
            })}
            examples={[
              t("review:mediaEmpty.connectExample1", {
                defaultValue: "Open Settings → tldw server to add your server URL."
              }),
              t("review:mediaEmpty.connectExample2", {
                defaultValue:
                  "Once connected, use Quick ingest in the header to add media from your own recordings and files."
              })
            ]}
            primaryActionLabel={t("settings:tldw.setupLink", "Set up server")}
            onPrimaryAction={() => navigate("/settings/tldw")}
          />
        )
      })()
    )
  }

  const mediaUnsupported = !capsLoading && capabilities && !capabilities.hasMedia

  if (isOnline && mediaUnsupported) {
    return (
      <FeatureEmptyState
        title={t("review:mediaEmpty.offlineTitle", {
          defaultValue: "Media Review isn’t available on this server yet"
        })}
        description={t("review:mediaEmpty.offlineDescription", {
          defaultValue:
            "This workspace depends on Media Review support in your tldw server. You can continue using chat, notes, and other tools while you upgrade to a version that includes Media."
        })}
        examples={[
          t("review:mediaEmpty.offlineExample1", {
            defaultValue:
              "Open Diagnostics to confirm your server version and available APIs."
          }),
          t("review:mediaEmpty.offlineExample2", {
            defaultValue:
              "After upgrading, reload the extension and return to Media."
          }),
          t("review:mediaEmpty.offlineTechnicalDetails", {
            defaultValue:
              "Technical details: this tldw server does not advertise the Media endpoints (for example, /api/v1/media and /api/v1/media/search)."
          })
        ]}
        primaryActionLabel={t("settings:healthSummary.diagnostics", {
          defaultValue: "Open Diagnostics"
        })}
        onPrimaryAction={() => navigate("/settings/health")}
      />
    )
  }

  return <MediaReviewPage />
}

const OptionMediaMulti = () => {
  return (
    <OptionLayout>
      <RouteErrorBoundary routeId="media-multi" routeLabel="Media Review">
        <MediaMultiInner />
      </RouteErrorBoundary>
    </OptionLayout>
  )
}

export default OptionMediaMulti
