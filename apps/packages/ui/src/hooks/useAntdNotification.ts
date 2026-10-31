import { useMemo } from "react"
import { App, notification as staticNotification } from "antd"
import type { ArgsProps, NotificationInstance } from "antd/es/notification/interface"
import { normalizeNotificationConfig } from "@/utils/antd-notification-compat"

export const useAntdNotification = (): NotificationInstance => {
  const { notification } = App.useApp()
  const base = typeof notification?.open === "function" ? notification : staticNotification

  return useMemo(() => {
    const normalize = (config: ArgsProps) => normalizeNotificationConfig(config) as ArgsProps
    const method = (type: "success" | "info" | "warning" | "error") =>
      (config: ArgsProps) => {
        if (typeof base[type] === "function") {
          base[type](normalize(config))
        } else {
          base.open({ ...normalize(config), type })
        }
      }
    return {
      open: (config: ArgsProps) => base.open(normalize(config)),
      success: method("success"),
      info: method("info"),
      warning: method("warning"),
      error: method("error"),
      destroy: (key?: React.Key) => base.destroy(key)
    }
  }, [base])
}
