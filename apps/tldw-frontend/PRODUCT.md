# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

People researching, learning from, and interacting with their own media and knowledge sources.

## Product Purpose

tldw combines media ingestion, transcription, search, and conversations with AI models in a self-hosted research assistant.

## Operating Context

The Next.js WebUI shares React components, actions, and state with the browser extension. Users choose configured models, compose messages, attach sources, and continue saved conversations.

## Capabilities and Constraints

Chat supports model selection, prompts, personas/characters, source retrieval, tools, voice, queued messages, and temporary conversations. Availability depends on server capabilities and connection state. Preserve draft ownership, IME input, slash commands, mentions, queue/stop behavior, privacy locks, and actionable warnings.

## Product Principles

- The composer prioritizes a calm writing space, with tools revealed when needed (user confirmed 2026-10-23).
- Essential actions and active state remain available on small screens.
- Reuse the established visual tokens, components, and behavior.
- Preserve existing user preferences and conversation data.

## Evidence on Hand

Product facts are established by the repository AGENTS.md and existing Chat implementation. TASK-13275 records the composer redesign and verification.
