import { getOllamaURL } from "~/services/tldw-server"
import { useTranslation } from "react-i18next"
import { useQuery } from "@tanstack/react-query"
import { Skeleton } from "antd"
import { cleanUrl } from "@/libs/clean-url"
import { Descriptions } from "antd"
import fetcher from "@/libs/fetcher"
import { translateMessage } from "@/i18n/translateMessage"

export const AboutApp = () => {
  const { t } = useTranslation("settings")

  const { data, status } = useQuery({
    queryKey: ["fetchServerVersion"],
    queryFn: async () => {
      const runtime =
        typeof browser !== "undefined" && browser?.runtime?.getManifest
          ? browser.runtime
          : typeof chrome !== "undefined" && chrome?.runtime?.getManifest
            ? chrome.runtime
            : null
      const chromeVersion = runtime?.getManifest()?.version ?? "unknown"
      try {
        const url = await getOllamaURL()
        const req = await fetcher(`${cleanUrl(url)}/openapi.json`)

        if (!req.ok) {
          return {
            serverVersion: "N/A",
            chromeVersion
          }
        }

        const res = (await req.json()) as {
          info?: {
            version?: string
          }
        }
        return {
          serverVersion: res.info?.version || "N/A",
          chromeVersion
        }
      } catch {
        return {
          serverVersion: "N/A",
          chromeVersion
        }
      }
    }
  })

  return (
    <div className="flex flex-col space-y-3">
      {status === "pending" && <Skeleton paragraph={{ rows: 4 }} active />}
      {status === "error" && (
        <div className="text-danger">
          {translateMessage(
            t,
            "settings:about.errorLoading",
            "Failed to load version information."
          )}
        </div>
      )}
      {status === "success" && (
        <div className="flex flex-col space-y-4">
          <div>
            <h2 className="text-base font-semibold leading-7 text-text">
              {translateMessage(t, "settings:about.heading", "About")}
            </h2>
            <div className="border-b border-border mt-3 mb-4"></div>
          </div>
          <Descriptions
            column={1}
            size="middle"
            items={[
              {
                key: 1,
                label: translateMessage(
                  t,
                  "settings:about.chromeVersion",
                  "tldw Assistant Version"
                ),
                children: data.chromeVersion
              },
              {
                key: 2,
                label: translateMessage(
                  t,
                  "settings:about.ollamaVersion",
                  "Server Version"
                ),
                children: data.serverVersion
              },
              {
                key: 3,
                label: "GitHub",
                children: (
                  <a
                    href="https://github.com/rmusser01/tldw_server"
                    target="_blank"
                    rel="noreferrer"
                    className="text-primary">
                    tldw_server on GitHub
                  </a>
                )
              },
              {
                key: 4,
                label: "License",
                children: (
                  <a
                    href="https://github.com/rmusser01/tldw_server/blob/dev/LICENSE"
                    target="_blank"
                    rel="noreferrer"
                    className="text-primary">
                    Source-available frontend terms
                  </a>
                )
              }
            ]}
          />
        </div>
      )}
    </div>
  )
}
