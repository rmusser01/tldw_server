import dynamic from "next/dynamic"
import Head from "next/head"

export default dynamic(async () => {
  const { SettingsRoute } = await import("@/routes/settings-route")
  const mod = await import("@/components/Option/Settings/ui-customization")
  const Component = mod.UiCustomizationSettings
  const Page = () => (
    <>
      <Head>
        <title>UI Customization | Settings | tldw</title>
      </Head>
      <SettingsRoute>
        <Component />
      </SettingsRoute>
    </>
  )
  return { default: Page }
}, { ssr: false })
