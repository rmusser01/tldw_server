import dynamic from "next/dynamic"
import Head from "next/head"

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

export default function ProviderKeysPage() {
  return <><Head><title>Provider Keys | tldw</title></Head><ProviderKeys /></>
}
