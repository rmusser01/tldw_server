import { Card } from "antd"
import { CheckCircle2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Link } from "react-router-dom"
import { useMissionCards } from "./hooks/useMissionCards"

export function GettingStartedSection() {
  const { t } = useTranslation()
  const { gettingStartedCards, completedCount, totalCount, allComplete } = useMissionCards()

  // Auto-hide when all milestones complete
  if (allComplete || gettingStartedCards.length === 0) return null

  return (
    <div data-testid="getting-started-section" className="mb-6">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-medium text-text m-0">{t("companionHome.gettingStarted.title", "Getting Started")}</h3>
        <span className="text-sm text-text-subtle">
          {t("companionHome.gettingStarted.progress", "{{completed}} of {{total}} complete", { completed: completedCount, total: totalCount })}
        </span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {gettingStartedCards.map((card) => {
          const Icon = card.icon
          return (
            <Link key={card.id} to={card.href} className="no-underline">
              <Card
                size="small"
                className={`transition-colors hover:border-primary ${card.isCompleted ? "opacity-60" : ""}`}
                data-testid={`mission-card-${card.id}`}
              >
                <div className="flex items-start gap-3">
                  <div className={`mt-0.5 ${card.isCompleted ? "text-green-500" : "text-primary"}`}>
                    {card.isCompleted ? <CheckCircle2 size={20} /> : <Icon size={20} />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className={`font-medium text-sm ${card.isCompleted ? "line-through text-text-subtle" : "text-text"}`}>
                      {card.title}
                    </div>
                    <div className="text-xs text-text-subtle mt-0.5 line-clamp-2">
                      {card.description}
                    </div>
                  </div>
                </div>
              </Card>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
