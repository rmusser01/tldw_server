import dynamic from "next/dynamic"
import Head from "next/head"

export default dynamic(async () => {
  const { SettingsRoute } = await import("@/routes/settings-route")
  const mod = await import("@/components/Option/Settings/setup-recovery-settings")
  const Component = mod.SetupRecoverySettings
  const Page = () => (
    <>
      <Head>
        <title>Setup &amp; Recovery | Settings | tldw</title>
      </Head>
      <SettingsRoute>
        <Component />
      </SettingsRoute>
    </>
  )
  return { default: Page }
}, { ssr: false })
