import React, { useEffect, useState } from "react"
import { Tabs } from "antd"
import { useTranslation } from "react-i18next"

type TldwSettingsTabKey = "connection" | "timeouts" | "billing"

type TldwSettingsTabsProps = {
  authMode: "single-user" | "multi-user"
  isLoggedIn: boolean
}

const SECTION_IDS: Record<TldwSettingsTabKey, string> = {
  connection: "tldw-settings-connection",
  timeouts: "tldw-settings-timeouts",
  billing: "tldw-settings-billing"
}

const getVisibleSectionKeys = (
  authMode: TldwSettingsTabsProps["authMode"],
  isLoggedIn: boolean
): TldwSettingsTabKey[] =>
  authMode === "multi-user" && isLoggedIn
    ? ["connection", "timeouts", "billing"]
    : ["connection", "timeouts"]

export const TldwSettingsTabs = ({
  authMode,
  isLoggedIn
}: TldwSettingsTabsProps) => {
  const { t } = useTranslation()
  const visibleSectionKeys = getVisibleSectionKeys(authMode, isLoggedIn)
  const [activeKey, setActiveKey] = useState<TldwSettingsTabKey>(
    visibleSectionKeys[0]
  )

  useEffect(() => {
    if (!visibleSectionKeys.includes(activeKey)) {
      setActiveKey(visibleSectionKeys[0])
    }
  }, [activeKey, visibleSectionKeys, authMode, isLoggedIn])

  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return

    const observer = new IntersectionObserver(
      (entries) => {
        const mostVisibleEntry = entries
          .filter((entry) => entry.isIntersecting)
          .map((entry) => {
            const key = visibleSectionKeys.find(
              (candidate) => SECTION_IDS[candidate] === entry.target.id
            )

            if (!key) return null

            return {
              key,
              ratio: entry.intersectionRatio
            }
          })
          .filter((entry): entry is { key: TldwSettingsTabKey; ratio: number } =>
            entry !== null
          )
          .sort((left, right) => right.ratio - left.ratio)[0]

        if (mostVisibleEntry) {
          setActiveKey(mostVisibleEntry.key)
        }
      },
      {
        threshold: [0.2, 0.4, 0.6],
        rootMargin: "-72px 0px -45% 0px"
      }
    )

    visibleSectionKeys.forEach((key) => {
      const element = document.getElementById(SECTION_IDS[key])
      if (element) {
        observer.observe(element)
      }
    })

    return () => {
      observer.disconnect()
    }
  }, [authMode, isLoggedIn, visibleSectionKeys])

  return (
    <Tabs
      activeKey={activeKey}
      size="small"
      className="sticky top-0 z-20 mb-4 border-b border-border bg-surface/95 px-1 py-2 backdrop-blur supports-[backdrop-filter]:bg-surface/85"
      items={[
        {
          key: "connection",
          label: t("settings:tldw.tabs.connection", "Connection")
        },
        {
          key: "timeouts",
          label: t("settings:tldw.tabs.timeouts", "Timeouts")
        },
        ...(authMode === "multi-user" && isLoggedIn
          ? [
              {
                key: "billing",
                label: t("settings:tldw.tabs.billing", "Billing")
              }
            ]
          : [])
      ]}
      onChange={(key) => {
        setActiveKey(key as TldwSettingsTabKey)
      }}
      onTabClick={(key) => {
        const element = document.getElementById(
          SECTION_IDS[key as TldwSettingsTabKey]
        )
        if (element) {
          element.scrollIntoView({
            behavior: "smooth",
            block: "start"
          })
        }
      }}
    />
  )
}
