# Tutorial Per-Page Coverage Design

## Purpose

Define the release rationale and route-by-route scope for baseline (`*-basics`) tutorials so reviewers can validate onboarding coverage before shipping.

## Coverage Scope (P0/P1)

| Route | Basics Tutorial ID | Release Tier |
|------|---------------------|--------------|
| `/chat` | `playground-basics` | P0 |
| `/research-workspace` | `research-workspace-basics` | P0 |
| `/media` | `media-basics` | P0 |
| `/knowledge` | `knowledge-basics` | P0 |
| `/characters` | `characters-basics` | P0 |
| `/prompts` | `prompts-basics` | P1 |
| `/evaluations` | `evaluations-basics` | P1 |
| `/notes` | `notes-basics` | P1 |
| `/flashcards` | `flashcards-basics` | P1 |
| `/world-books` | `world-books-basics` | P1 |

## Design Rationale

1. Prioritize P0 routes where first-run confusion is highest (`/chat`, `/research-workspace`, `/media`, `/knowledge`, `/characters`).
2. Ensure each route has one clear entry tutorial before expanding to advanced/tutorial chains.
3. Keep route ownership explicit via one canonical basics ID per page to simplify QA and completion tracking.
4. Gate release readiness on this map plus the manual QA checklist in `Docs/Design/Tutorial_Per_Page_Coverage.md`.

## Release Review Checklist

- Verify every route in this table resolves to a registered tutorial definition.
- Ensure Page Help (`?`) and Quick Chat `Browse Guides` show the mapped tutorial on each route.
- Validate that completion state persists and replay works for each mapped basics tutorial.
