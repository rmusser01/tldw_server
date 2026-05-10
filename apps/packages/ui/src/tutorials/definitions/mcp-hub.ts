/**
 * MCP Hub Tutorial Definitions
 */

import { Plug } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const mcpHubBasics: TutorialDefinition = {
  id: "mcp-hub-basics",
  routePattern: "/mcp-hub",
  labelKey: "tutorials:mcpHub.basics.label",
  labelFallback: "MCP Hub Basics",
  descriptionKey: "tutorials:mcpHub.basics.description",
  descriptionFallback:
    "Learn how to browse tools, connect servers, and manage permissions.",
  icon: Plug,
  priority: 1,
  steps: [
    {
      target: '[data-testid="mcp-hub-shell"]',
      titleKey: "tutorials:mcpHub.basics.welcomeTitle",
      titleFallback: "Welcome to MCP Hub",
      contentKey: "tutorials:mcpHub.basics.welcomeContent",
      contentFallback:
        "MCP Hub manages external AI tools. Start here to see what tools are available and connect new servers.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="mcp-hub-tab-tool-catalogs"]',
      titleKey: "tutorials:mcpHub.basics.catalogTitle",
      titleFallback: "Tool Catalog",
      contentKey: "tutorials:mcpHub.basics.catalogContent",
      contentFallback:
        "Browse all registered tools here. Each tool shows its capabilities and risk level.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="mcp-hub-tab-credentials"]',
      titleKey: "tutorials:mcpHub.basics.credentialsTitle",
      titleFallback: "Servers & Credentials",
      contentKey: "tutorials:mcpHub.basics.credentialsContent",
      contentFallback:
        "Connect external MCP servers here. Each server provides additional tools for your AI assistant.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="mcp-hub-workflow-access"]',
      titleKey: "tutorials:mcpHub.basics.profilesTitle",
      titleFallback: "Access Workflow",
      contentKey: "tutorials:mcpHub.basics.profilesContent",
      contentFallback:
        "Use Access to create profiles and assign which tools each user or persona can reach.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="mcp-hub-workflow-audit"]',
      titleKey: "tutorials:mcpHub.basics.auditTitle",
      titleFallback: "Governance Audit",
      contentKey: "tutorials:mcpHub.basics.auditContent",
      contentFallback:
        "Review policy findings and configuration issues across all your MCP Hub settings.",
      placement: "bottom",
      disableBeacon: true
    }
  ]
}

export const mcpHubTutorials: TutorialDefinition[] = [mcpHubBasics]
