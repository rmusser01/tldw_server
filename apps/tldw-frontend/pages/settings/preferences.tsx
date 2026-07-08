import dynamic from "next/dynamic"
import Head from "next/head"

export default dynamic(async () => {
  const { SettingsRoute } = await import("@/routes/settings-route")
  const mod = await import("@/components/Option/Settings/preferences-settings")
  const Component = mod.PreferencesSettings
  const Page = () => (
    <>
      <Head>
        <title>Preferences | Settings | tldw</title>
      </Head>
      <SettingsRoute>
        <Component />
      </SettingsRoute>
    </>
  )
  return { default: Page }
}, { ssr: false })
