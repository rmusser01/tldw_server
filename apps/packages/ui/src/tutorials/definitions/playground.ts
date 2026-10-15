/**
 * Playground Tutorial Definitions
 * Tutorials for the Playground/Chat page, split into focused bite-sized tutorials
 */

import { MessageSquare, Wrench, Mic } from "lucide-react"
import type { TutorialDefinition } from "../registry"

/**
 * Playground Basics Tutorial
 * Covers: Model selection, sending messages, basic chat features
 */
const playgroundBasics: TutorialDefinition = {
  id: "playground-basics",
  routePattern: "/chat",
  labelKey: "tutorials:playground.basics.label",
  labelFallback: "Chat Basics",
  descriptionKey: "tutorials:playground.basics.description",
  descriptionFallback: "Learn how to select models and send your first message",
  icon: MessageSquare,
  priority: 1,
  steps: [
    {
      target: '[data-testid="model-selector"], [data-testid="chat-input"]',
      titleKey: "tutorials:playground.basics.modelTitle",
      titleFallback: "Choose a Model",
      contentKey: "tutorials:playground.basics.modelContent",
      contentFallback:
        "Select an AI model to chat with. Different models have different capabilities - look for tags like 'Vision' or 'Fast'. Star your favorites for quick access.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="chat-input"]',
      titleKey: "tutorials:playground.basics.inputTitle",
      titleFallback: "Start a Conversation",
      contentKey: "tutorials:playground.basics.inputContent",
      contentFallback:
        "Type your message here and press Enter to send. You can ask questions, request help with tasks, or have a conversation.",
      placement: "top"
    },
    {
      target: '[data-testid="chat-input"]',
      titleKey: "tutorials:playground.basics.slashTitle",
      titleFallback: "Slash Commands",
      contentKey: "tutorials:playground.basics.slashContent",
      contentFallback:
        "Type / to see available commands. Try /search to find content in your knowledge base, or /web to search the internet.",
      placement: "top"
    },
    {
      target:
        '[data-testid="composer-casual-advanced-chip"], [data-testid="chat-input"]',
      titleKey: "tutorials:playground.basics.customizeTitle",
      titleFallback: "Customize Responses",
      contentKey: "tutorials:playground.basics.customizeContent",
      contentFallback:
        "A system prompt guides how the assistant replies. Use chat customization controls to choose one or adjust the response; create reusable prompts in the Prompts workspace.",
      placement: "bottom"
    },
    {
      target: '[data-testid="chat-input"]',
      titleKey: "tutorials:playground.basics.newChatTitle",
      titleFallback: "Start Fresh",
      contentKey: "tutorials:playground.basics.newChatContentV2",
      contentFallback:
        "Start a new conversation from the sidebar or with the New chat shortcut listed in keyboard help. Saved conversations can be reopened with Find conversations. A Temporary conversation asks before you discard it.",
      placement: "bottom"
    }
  ]
}

/**
 * Playground Tools Tutorial
 * Covers: RAG/knowledge search, web search, file uploads, MCP tools
 */
const playgroundTools: TutorialDefinition = {
  id: "playground-tools",
  routePattern: "/chat",
  labelKey: "tutorials:playground.tools.label",
  labelFallback: "Tools & Attachments",
  descriptionKey: "tutorials:playground.tools.description",
  descriptionFallback:
    "Discover knowledge search, file uploads, and advanced tools",
  icon: Wrench,
  priority: 2,
  prerequisites: ["playground-basics"],
  steps: [
    {
      target: '[data-testid="tools-button"]',
      titleKey: "tutorials:playground.tools.toolsButtonTitle",
      titleFallback: "Tools & Attachments",
      contentKey: "tutorials:playground.tools.toolsButtonContent",
      contentFallback:
        "Access knowledge search, upload files, configure MCP tools, and more advanced features here.",
      placement: "top",
      disableBeacon: true
    },
    {
      target: '[data-testid="knowledge-search-toggle"]',
      titleKey: "tutorials:playground.tools.knowledgeTitle",
      titleFallback: "Knowledge Search",
      contentKey: "tutorials:playground.tools.knowledgeContent",
      contentFallback:
        "Enable knowledge search to let the AI reference your ingested documents, notes, and media. Great for research and fact-checking.",
      placement: "top"
    },
    {
      target: '[data-testid="web-search-toggle"]',
      titleKey: "tutorials:playground.tools.webSearchTitle",
      titleFallback: "Web Search",
      contentKey: "tutorials:playground.tools.webSearchContent",
      contentFallback:
        "Toggle web search to let the AI search the internet for current information. Perfect for recent news or topics not in your knowledge base.",
      placement: "top"
    },
    {
      target: '[data-testid="attachment-button"]',
      titleKey: "tutorials:playground.tools.uploadTitle",
      titleFallback: "Attach Files",
      contentKey: "tutorials:playground.tools.uploadContent",
      contentFallback:
        "Upload images or documents to discuss them with the AI. Vision-capable models can analyze images directly.",
      placement: "top"
    },
    {
      target: '[data-testid="mcp-tools-toggle"]',
      titleKey: "tutorials:playground.tools.mcpTitle",
      titleFallback: "MCP Tools",
      contentKey: "tutorials:playground.tools.mcpContent",
      contentFallback:
        "MCP (Model Context Protocol) tools let the AI interact with external services and execute actions on your behalf.",
      placement: "top"
    }
  ]
}

/**
 * Playground Voice Tutorial
 * Covers: Voice chat, dictation
 */
const playgroundVoice: TutorialDefinition = {
  id: "playground-voice",
  routePattern: "/chat",
  labelKey: "tutorials:playground.voice.label",
  labelFallback: "Voice Chat",
  descriptionKey: "tutorials:playground.voice.description",
  descriptionFallback:
    "Learn to use voice input and have spoken conversations",
  icon: Mic,
  priority: 3,
  prerequisites: ["playground-basics"],
  steps: [
    {
      target: '[data-testid="voice-chat-button"]',
      titleKey: "tutorials:playground.voice.voiceChatTitle",
      titleFallback: "Voice Chat",
      contentKey: "tutorials:playground.voice.voiceChatContent",
      contentFallback:
        "Start a hands-free voice conversation. Speak naturally and hear AI responses aloud.",
      placement: "top",
      isFixed: true,
      disableBeacon: true
    },
    {
      target: '[data-testid="dictation-button"]',
      titleKey: "tutorials:playground.voice.dictationTitle",
      titleFallback: "Dictation",
      contentKey: "tutorials:playground.voice.dictationContent",
      contentFallback:
        "Use dictation to speak your messages instead of typing. Great for longer inputs or when your hands are busy.",
      placement: "top"
    }
  ]
}

/**
 * Export all playground tutorials
 */
export const playgroundTutorials: TutorialDefinition[] = [
  playgroundBasics,
  playgroundTools,
  playgroundVoice
]
