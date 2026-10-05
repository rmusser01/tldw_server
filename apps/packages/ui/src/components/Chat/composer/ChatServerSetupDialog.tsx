import React from "react"
import { Button, Modal } from "antd"
import { useTranslation } from "react-i18next"
import { useAntdModal } from "@/hooks/useAntdModal"
import { TldwSettings } from "@/components/Option/Settings/tldw"
import { resolveSettingsNavigationUrl } from "@/utils/settings-return"

export function ChatServerSetupDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { t } = useTranslation(["settings", "playground"])
  const modal = useAntdModal()
  const confirmation = React.useRef<ReturnType<typeof modal.confirm> | null>(null)
  React.useEffect(() => () => confirmation.current?.destroy(), [])
  const [dirty, setDirty] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const close = () => {
    if (saving) return
    if (dirty) {
      if (confirmation.current) return
      const clearConfirmation = () => { if (confirmation.current === instance) confirmation.current = null }
      const instance = modal.confirm({
        title: t("playground:composer.discardSettingsTitle", "Discard connection changes?"),
        content: t("settings:tldw.unsaved.leave", "You have unsaved connection settings. Leave this page and discard these changes?"),
        okText: t("playground:composer.discardSettingsAction", "Discard changes"),
        cancelText: t("playground:composer.keepEditingSettings", "Keep editing"),
        onOk: onClose,
        onCancel: clearConfirmation,
        afterClose: clearConfirmation
      })
      confirmation.current = instance
      return
    }
    onClose()
  }
  const diagnostics = () => {
    const url = resolveSettingsNavigationUrl("/settings/health", window.location.href)
    if (url) window.open(url, "_blank", "noopener")
  }
  return <Modal open={open} title={t("playground:composer.serverSetupTitle", "Server settings")}
    centered width={760} maskClosable={false} closable={!saving} keyboard={!saving} onCancel={close} afterClose={() => {
      setDirty(false)
      if (document.activeElement === document.body) document.getElementById("main-content")?.focus({ preventScroll: true })
    }} destroyOnHidden
    styles={{ body: { maxHeight: "calc(100dvh - 180px)", overflowY: "auto" } }}
    footer={<Button disabled={saving} onClick={close}>{t("playground:composer.returnToChat", "Return to chat")}</Button>}>
    <TldwSettings onDirtyChange={setDirty} onSavingChange={setSaving} onOpenDiagnostics={diagnostics} />
  </Modal>
}
