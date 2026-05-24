# Product

## Register

product

## Users

tldw serves self-hosters, researchers, journalists, OSINT analysts, students, educators, knowledge workers, developers, and operators who need to ingest, transcribe, search, analyze, and reuse media and documents under their own control. Users may arrive through the WebUI, browser extension, API, or deployment tooling. Their context is often complex: local data, provider keys, model choices, GPU or sidecar setup, long-running jobs, privacy constraints, and source-heavy workflows.

They are usually trying to turn raw sources into grounded understanding: searchable libraries, transcripts, notes, chat answers with citations, study materials, reports, watchlist outputs, evaluations, and durable work products. The interface must support both first-time setup and repeated expert use without hiding important system state.

## Product Purpose

tldw is a local and self-hosted research assistant and media-analysis platform. It helps users collect sources, process audio, video, documents, and web content, retrieve relevant context, chat with grounded answers, create notes and outputs, administer providers and workers, and inspect system readiness.

Success means a user can move from messy source material to trustworthy, reusable artifacts without losing provenance, privacy, control, or operational clarity. The product should make hard infrastructure visible enough to be understood, while keeping the primary research and learning workflows efficient.

## Brand Personality

tldw should support multiple personality modes rather than one flat voice:

- Calm expert for setup, admin, health, recovery, security, and deployment. The tone is precise, factual, and oriented around what to do next.
- Primer-like mentor for onboarding, learning, study, research guidance, and adaptive assistance. The tone is humane and structured, with useful prompts and scaffolding rather than cheerleading.
- Power-user cockpit for chat, RAG, media libraries, evaluations, watchlists, jobs, and dense operational screens. The tone is compact, scannable, configurable, and explicit about state.

Across modes, the product should feel privacy-respecting, technically honest, source-aware, and humble about uncertainty. It should not pretend model output is magic or conceal the machinery that users need to trust.

## Anti-references

Avoid generic AI-app gloss: pastel gradients, vague sparkle language, oversized marketing metrics, decorative agent mascots, and feature cards that say little. Avoid toy-like chat surfaces that make serious research feel unserious. Avoid hidden-controls simplicity where critical provider, source, permission, cost, or processing state disappears behind "smart" defaults.

Also avoid cluttered enterprise admin screens that bury the next action, raw traceback dumps without recovery paths, modal-first flows for routine work, and designs that make the WebUI and extension feel like unrelated products.

## Design Principles

1. Source trust before model spectacle. Ground answers, generated artifacts, and recommendations in visible sources, citations, status, and retrieval context.
2. Show system state plainly. Setup, auth, provider readiness, background jobs, degraded services, permissions, and failures should be legible without digging through logs.
3. Keep expert workflows fast. Dense screens are acceptable when hierarchy, keyboard access, defaults, and progressive disclosure make repeated work efficient.
4. Teach through the workflow. Onboarding and empty states should help users take the next useful step in context, not explain the whole product at once.
5. Preserve user control. Privacy, self-hosting, local data ownership, configuration, and reversibility are product values, not implementation details.

## Accessibility & Inclusion

Target WCAG AA for product surfaces. Maintain keyboard-operable controls, visible focus states, semantic landmarks, accessible names for icon-only controls, and text contrast that works in both light and dark themes. Support reduced motion and avoid relying on color alone to communicate state.

Design for users with large libraries, long-running jobs, intermittent backend availability, smaller screens, screen readers, and mixed technical fluency. Error and empty states should pair clear language with recovery actions.
