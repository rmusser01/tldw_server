import React from "react"
import { ChevronDown, ChevronUp } from "lucide-react"
import type { WorkspaceBanner as WorkspaceBannerModel } from "@/types/workspace"

type WorkspaceBannerProps = {
  banner: WorkspaceBannerModel
  workspaceName: string
  isMobile: boolean
}

const hasBannerContent = (banner: WorkspaceBannerModel): boolean =>
  banner.title.trim().length > 0 ||
  banner.subtitle.trim().length > 0 ||
  Boolean(banner.image?.dataUrl)

export const WorkspaceBanner: React.FC<WorkspaceBannerProps> = ({
  banner,
  workspaceName,
  isMobile
}) => {
  const hasImage = Boolean(banner.image?.dataUrl)
  const [collapsed, setCollapsed] = React.useState(!hasImage)

  if (!hasBannerContent(banner)) {
    return null
  }

  const title = banner.title.trim() || workspaceName || "Research Workspace"
  const subtitle = banner.subtitle.trim()
  const backgroundImage = banner.image?.dataUrl
    ? `linear-gradient(120deg, rgba(8, 12, 18, 0.72) 0%, rgba(8, 12, 18, 0.24) 100%), url(${banner.image.dataUrl})`
    : "linear-gradient(125deg, color-mix(in oklab, var(--primary) 24%, transparent) 0%, color-mix(in oklab, var(--surface-2) 76%, transparent) 100%)"

  const CollapseIcon = collapsed ? ChevronDown : ChevronUp

  if (collapsed) {
    return (
      <section
        data-testid="workspace-banner"
        className="mx-2 mt-2 flex h-12 cursor-pointer items-center justify-between overflow-hidden rounded-2xl border border-border/70 px-4 shadow-card"
        style={{
          backgroundImage,
          backgroundPosition: "center",
          backgroundSize: "cover"
        }}
        onClick={() => setCollapsed(false)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            setCollapsed(false)
          }
        }}
      >
        <h2
          data-testid="workspace-banner-title"
          className="truncate text-sm font-semibold text-white"
        >
          {title}
        </h2>
        <CollapseIcon className="h-4 w-4 shrink-0 text-white/70" />
      </section>
    )
  }

  return (
    <section
      data-testid="workspace-banner"
      className={`group relative mx-2 mt-2 overflow-hidden rounded-2xl border border-border/70 shadow-card ${
        isMobile ? "min-h-[108px]" : "min-h-[132px]"
      }`}
      style={{
        backgroundImage,
        backgroundPosition: "center",
        backgroundSize: "cover"
      }}
    >
      <button
        type="button"
        data-testid="workspace-banner-collapse"
        onClick={() => setCollapsed(true)}
        className="absolute right-2 top-2 rounded-full bg-black/30 p-1 text-white/70 opacity-0 transition hover:bg-black/50 hover:text-white group-hover:opacity-100"
        aria-label="Collapse banner"
      >
        <CollapseIcon className="h-3.5 w-3.5" />
      </button>
      <div
        className={`flex h-full flex-col justify-end px-4 py-3 ${
          isMobile ? "gap-1.5" : "gap-2"
        }`}
      >
        <h2
          data-testid="workspace-banner-title"
          className={`line-clamp-2 font-semibold text-white ${
            isMobile ? "text-lg" : "text-xl"
          }`}
        >
          {title}
        </h2>
        {subtitle.length > 0 && (
          <p
            data-testid="workspace-banner-subtitle"
            className="line-clamp-2 max-w-3xl text-sm text-white/90"
          >
            {subtitle}
          </p>
        )}
      </div>
    </section>
  )
}
