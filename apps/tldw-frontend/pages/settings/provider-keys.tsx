import dynamic from "next/dynamic"
import { withPageTitle } from "@web/components/navigation/withPageTitle"

const ProviderKeys = dynamic(async () => {
  const { SettingsRoute } = await import("@/routes/settings-route")
  const mod = await import("@/components/Option/Settings/ProviderKeysSettings")
  const Component = mod.ProviderKeysSettings
  const Page = () => (
    <SettingsRoute>
      <Component />
    </SettingsRoute>
  )
  return { default: Page }
}, { ssr: false })

export default withPageTitle("Provider Keys", ProviderKeys)
