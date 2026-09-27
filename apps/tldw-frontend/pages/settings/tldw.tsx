import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const ServerSettings = dynamic(async () => {
  const { SettingsRoute } = await import("@/routes/settings-route")
  const mod = await import("@/components/Option/Settings/tldw")
  const Component = mod.TldwSettings
  const Page = () => (
    <SettingsRoute>
      <Component />
    </SettingsRoute>
  )
  return { default: Page }
}, { ssr: false })

export default withPageTitle("Server Settings", ServerSettings)
