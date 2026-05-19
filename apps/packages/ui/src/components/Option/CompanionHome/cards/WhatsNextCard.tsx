import { Card, Button } from "antd"
import { ArrowRight } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { useMissionCards } from "../hooks/useMissionCards"

export function WhatsNextCard() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const { whatsNextCard, allComplete } = useMissionCards()

  if (allComplete || !whatsNextCard) return null

  const Icon = whatsNextCard.icon

  return (
    <Card
      data-testid="whats-next-card"
      className="border-primary/30 bg-primary/5"
      size="small"
    >
      <div className="flex items-center gap-3">
        <Icon size={24} className="text-primary flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="font-medium text-sm text-text">
            {t("companionHome.whatsNext.prefix", "What's next:")} {whatsNextCard.title}
          </div>
          <div className="text-xs text-text-subtle mt-0.5">
            {whatsNextCard.description}
          </div>
        </div>
        <Button type="primary" size="small" icon={<ArrowRight size={14} />} onClick={() => navigate(whatsNextCard.href)}>
          {t("common:go", "Go")}
        </Button>
      </div>
    </Card>
  )
}
