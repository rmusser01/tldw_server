import dynamic from "next/dynamic"
import Head from "next/head"

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

export default function ServerSettingsPage() {
  return <><Head><title>Server Settings | tldw</title></Head><ServerSettings /></>
}
