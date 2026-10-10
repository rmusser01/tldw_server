import React, { useEffect, useState } from "react"
import { Typography } from "antd"
import type { TFunction } from "i18next"

type RefreshedAtLabelProps = {
  /** Timestamp of the last refresh; null before the first refresh lands. */
  at: Date | null
  /** Translation function from the host page (stable for the session). */
  t: TFunction
}

/**
 * "Updated Xs ago" staleness label (admin perf C-S1 / F14).
 *
 * The 10s tick interval and its `timeSinceRefresh` state live inside this
 * tiny component on purpose: a tick only re-renders this label, never the
 * host page or its tables. `React.memo` (with the stable `t` reference) keeps
 * the label itself from re-rendering on unrelated host re-renders.
 */
function RefreshedAtLabelBase({ at, t }: RefreshedAtLabelProps) {
  const [timeSinceRefresh, setTimeSinceRefresh] = useState("")

  useEffect(() => {
    const tick = () => {
      if (!at) { setTimeSinceRefresh(""); return }
      const secs = Math.floor((Date.now() - at.getTime()) / 1000)
      if (secs < 10) setTimeSinceRefresh(t("settings:adminMonitoring.justNow", "just now"))
      else if (secs < 60) setTimeSinceRefresh(`${secs}${t("settings:adminMonitoring.secondsAgoSuffix", "s ago")}`)
      else setTimeSinceRefresh(`${Math.floor(secs / 60)}${t("settings:adminMonitoring.minutesAgoSuffix", "m ago")}`)
    }
    tick()
    const id = setInterval(tick, 10_000)
    return () => clearInterval(id)
  }, [at, t])

  if (!timeSinceRefresh) return null

  return (
    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
      {t("settings:adminMonitoring.updated", "Updated")} {timeSinceRefresh}
    </Typography.Text>
  )
}

export const RefreshedAtLabel = React.memo(RefreshedAtLabelBase)

export default RefreshedAtLabel
