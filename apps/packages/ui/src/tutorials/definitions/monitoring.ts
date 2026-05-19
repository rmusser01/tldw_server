/**
 * Monitoring Tutorial Definitions
 */

import { Activity } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const monitoringBasics: TutorialDefinition = {
  id: "monitoring-basics",
  routePattern: "/admin/monitoring",
  labelKey: "tutorials:monitoring.basics.label",
  labelFallback: "Monitoring Basics",
  descriptionKey: "tutorials:monitoring.basics.description",
  descriptionFallback:
    "Learn how to monitor server health and set up alert rules",
  icon: Activity,
  priority: 1,
  steps: [
    {
      target: ".ant-card:first-of-type",
      titleKey: "tutorials:monitoring.basics.overviewTitle",
      titleFallback: "System Overview",
      contentKey: "tutorials:monitoring.basics.overviewContent",
      contentFallback:
        "This section shows your server's current health metrics and security status. Use the refresh button or enable auto-refresh to keep it up to date.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: ".ant-form-inline",
      titleKey: "tutorials:monitoring.basics.createRuleTitle",
      titleFallback: "Create Alert Rules",
      contentKey: "tutorials:monitoring.basics.createRuleContent",
      contentFallback:
        "Set up rules to be notified when metrics cross thresholds. Pick a metric name, set a threshold and duration, then choose a severity level.",
      placement: "bottom"
    },
    {
      target: ".ant-table:nth-of-type(2), .ant-card:nth-of-type(3)",
      titleKey: "tutorials:monitoring.basics.historyTitle",
      titleFallback: "Alert History",
      contentKey: "tutorials:monitoring.basics.historyContent",
      contentFallback:
        "When alerts trigger, they appear here. You can assign alerts to yourself, snooze them, or escalate to critical.",
      placement: "top"
    }
  ]
}

export const monitoringTutorials: TutorialDefinition[] = [monitoringBasics]
