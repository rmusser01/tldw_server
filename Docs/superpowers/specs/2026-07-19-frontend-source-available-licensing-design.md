# Time-Delayed Source-Available Frontend Licensing Design

- **Date:** 2026-07-19
- **Backlog task:** TASK-12117
- **Status:** Approved design; implementation and legal review pending
- **Licensor:** Robert Benjamin Jake Musser

## Purpose

Protect future tldw frontend releases from direct commercial cloning during an
initial customer-development window while preserving broad self-hosting,
internal commercial use, community modification, redistribution, consulting,
and commercial adoption of the backend.

The selected model is Business Source License 1.1 (`BUSL-1.1`) for protected
frontend releases, with each release changing automatically to
`AGPL-3.0-only` 24 months after publication. The server and all repository
areas not explicitly included in the protected frontend boundary remain
`GPL-3.0-only`.

This document records product and engineering policy. It is not final license
language or legal advice. An attorney must approve the BSL parameters,
Additional Use Grant, CLA, trademark policy, and commercial-license template
before the first protected release.

## Goals

- Delay direct commercial clones of the official WebUI and browser extension.
- Keep source visible and useful for personal, internal, self-hosted, and
  customer-specific deployments.
- Keep the GPL backend available for commercial adoption and independent
  client development.
- Accept community frontend contributions without losing the ability to offer
  commercial exceptions or complete the planned AGPL transition.
- Make license scope, version dates, third-party notices, and branding rights
  unambiguous in source and release artifacts.
- Convert every protected release to an OSI open-source copyleft license on a
  predictable rolling schedule.

## Non-Goals

- Preventing independently developed clients from using the public GPL API.
- Retroactively changing the license of any version already published under
  GPL.
- Restricting commercial use of the backend.
- Claiming that a BSL-covered release is open source before its Change Date.
- Creating a custom software license or modifying the standard BSL terms.
- Replacing review by qualified legal counsel.

## Considered Approaches

### 1. BSL 1.1 with a rolling AGPL transition (selected)

This model best matches the business goal. It allows a broad Additional Use
Grant while withholding permission for a substituting commercial product or
hosted service. Each version becomes `AGPL-3.0-only` after 24 months.

Costs include source-available rather than open-source status during the
restricted window, strict separation from GPL implementation code, release
metadata, a frontend CLA, and legal review.

### 2. Immediate AGPL-3.0-only

This is simpler and ensures hosted modifications are offered to network users,
but it permits immediate commercial hosting, redistribution, and competition.
It protects openness rather than a market-entry window.

### 3. Proprietary frontend with delayed AGPL publication

This offers stronger control but conflicts with the desired self-hosting and
contribution model. It adds more friction than the business goal requires.

FSL was rejected because its standard two-year transition is limited to MIT or
Apache 2.0. A custom license was rejected because it would add avoidable legal
and adoption uncertainty.

## License Boundary

The repository becomes an explicitly multi-license monorepo.

| Scope | License before Change Date | License after Change Date |
| --- | --- | --- |
| Repository-authored code, tests, scripts, and original assets under `admin-ui/**` | `BUSL-1.1` | `AGPL-3.0-only` |
| Repository-authored code, tests, scripts, and original assets under `apps/tldw-frontend/**` | `BUSL-1.1` | `AGPL-3.0-only` |
| Repository-authored code, tests, scripts, and original assets under `apps/extension/**` | `BUSL-1.1` | `AGPL-3.0-only` |
| Repository-authored code, tests, scripts, and original assets under `apps/packages/ui/**` | `BUSL-1.1` | `AGPL-3.0-only` |
| Frontend-only workspace and release definitions classified in the root matrix | `BUSL-1.1` | `AGPL-3.0-only` |
| `tldw_Server_API/**` and other unlisted repository paths | `GPL-3.0-only` | `GPL-3.0-only` |
| Markdown documentation, including nested frontend documentation | `GPL-3.0-only` unless explicitly listed as a release asset | `GPL-3.0-only` unless explicitly listed as a release asset |
| Third-party code and assets | Existing upstream terms | Existing upstream terms |

The root GPL license remains. A root license matrix must override the mistaken
impression that one license governs every path. Full GPL, AGPL, and BSL texts
will live under a dedicated `LICENSES/` directory, and every protected package
will carry a local notice pointing to its applicable BSL parameters and Change
Date.

Frontend workspace files are classified individually. The design must not put
all of `apps/**` under BSL because that directory also contains components that
remain GPL. Documentation stays GPL unless it is an inseparable frontend
release asset explicitly classified otherwise.

The backend and protected frontends remain separate programs communicating
through documented HTTP and WebSocket interfaces. Protected code must not
import, bundle, copy, or link GPL backend implementation code before its Change
Date. Generated API clients, copied schemas, shared scripts, fonts, icons, and
other ambiguous inputs require provenance and compatibility review before they
are included in the protected scope.

## Additional Use Grant Policy

Counsel will translate this policy into a compliant BSL Additional Use Grant.
The grant permits the following during the 24-month restricted window:

- Personal, household, educational, evaluation, and non-production use.
- Internal production use by individuals, companies, nonprofits, and
  government entities.
- Self-hosting for the licensee's own organization.
- Modification and redistribution subject to the applicable BSL terms.
- Paid consulting, customization, deployment, and management of a dedicated
  instance for one identified customer's internal use.
- Integration with, and commercial use of, the separately licensed GPL server.

The grant does not authorize a Competing Offering. A use is a Competing
Offering when protected frontend code or a derivative is used as:

- A commercial product whose primary purpose substantially replaces the
  official tldw WebUI or browser extension.
- A hosted or managed service offered generally to multiple unrelated
  customers that substantially replaces an official tldw frontend service.
- A browser-store or app-store product positioned as a substitute for the
  official extension or WebUI.

A consultant may be paid to deploy, customize, or manage a dedicated instance
for a named customer's internal use. The grant must distinguish that engagement
from a repeatable provider-controlled or multi-tenant competing service.

The restriction reaches only the protected code and its derivatives. It does
not prevent a third party from independently developing a clean-room client
against the GPL API.

Standard BSL does not impose AGPL-style corresponding-source duties on every
modified hosted or binary-distributed copy during the restricted window. The
model instead withholds permission for the competing hosted use. AGPL source
sharing takes effect when the version reaches its Change Date.

## Rolling Release Lifecycle

Each tagged frontend release has a concrete release record containing:

- Product and semantic version.
- Release date.
- Change Date exactly 24 calendar months later.
- Current license identifier: `BUSL-1.1`.
- Change License identifier: `AGPL-3.0-only`.
- Applicable Additional Use Grant revision.
- Source commit and release-artifact identifiers.

The first BSL release starts a new frontend version line. Existing GPL tags,
commits, artifacts, and grants remain unchanged and irrevocably available under
their existing terms.

After a stable release, active development advances to the next prerelease
version with prospective release metadata. Release automation must replace
prospective values with the concrete release and Change Dates before publishing
artifacts. A release cannot proceed with missing, stale, or inconsistent dates.

The BSL terms cause an automatic license transition on the Change Date; no
relicensing commit is required. A maintained release registry and scheduled
check will nevertheless identify upcoming and completed transitions and update
public documentation for discoverability.

## Source and Artifact Notices

Every source archive, WebUI image, browser-extension package, and admin UI
artifact must include:

- The governing BSL text and release-specific parameters.
- Product version, release date, and Change Date.
- The `AGPL-3.0-only` Change License text or an unambiguous pointer to it.
- A source repository link.
- Required third-party notices and licenses.
- A statement that no tldw trademark rights are granted.

Package manifests use `BUSL-1.1` while a release is in its restricted window.
Because package metadata cannot fully express a timed transition, the
release-specific license record is authoritative. Each repository-authored
source file that supports comments receives an `SPDX-License-Identifier:
BUSL-1.1` notice and a short reference to its package's release-specific BSL
parameters. Non-commentable original assets are mapped explicitly in the root
license matrix. Generated and vendor files retain their upstream notices and
are never assigned a blanket frontend license. Counsel must approve the final
notice form before the first protected release.

Public README files, documentation, container metadata, About/Legal screens,
and browser-store descriptions must use the phrase "source-available under
Business Source License 1.1." They must not describe a current BSL version as
open source. The repository-level description must say that the server is GPLv3
and the protected frontends are time-delayed source-available software.

## Contributions and CLA

Frontend contributions require a signed, path-aware contributor license
agreement before merge. Contributors retain copyright. They grant Robert
Benjamin Jake Musser a permanent, worldwide, nonexclusive, irrevocable, and
sublicensable copyright and patent license sufficient to:

- Distribute contributions under BSL 1.1.
- Apply the scheduled `AGPL-3.0-only` transition.
- Grant separate commercial licenses.
- Maintain and relicense the combined frontend work consistently with this
  design.

The CLA must confirm that the contributor, and when applicable the
contributor's employer, has authority to make the grant. It applies to the
protected frontend boundary. Existing GPL backend contribution terms remain
unchanged.

Before adoption, repository history must be audited for authors, commit
trailers, imported projects, and prior contribution terms. Any material for
which the Licensor lacks necessary rights must stay under its existing license,
be separated, be replaced, or receive permission before the first BSL release.

## Trademark Policy

Licensing the code grants no rights to tldw product names, logos, icons,
official domains, signing identities, or extension-store identities. Forks and
redistributed builds must remove official branding and must not imply
endorsement or official status. Accurate nominative statements such as
"compatible with tldw_server" remain permitted.

The initial policy owner is Robert Benjamin Jake Musser. The policy uses the
`TM` designation and must not use the registered-trademark symbol unless a mark
is registered. A later assignment to a legal entity does not revoke existing
software licenses.

## Selective Commercial Licensing

Uses outside the Additional Use Grant may receive a separate commercial
license at the Licensor's sole discretion. There is no automatic right to buy
an exception and no published universal price.

Each agreement is limited to named legal entities, products, deployments,
territories when relevant, and time periods. It may authorize a Competing
Offering without changing the public BSL rights of any other party. Support,
warranty, privacy, security, service levels, and trademark permissions are
separate contractual subjects rather than implied software-license terms.

## Verification and Enforcement Gates

### Provenance gate

- Audit human authors and co-author trailers for every protected path.
- Inventory copied, generated, and vendored material.
- Scan runtime and build dependencies for license compatibility.
- Preserve the Page Assist MIT copyright and permission notice.
- Stop the affected relicense when ownership or compatibility is unresolved.

### Boundary gate

- Maintain an allowlist of BSL-protected paths and explicit GPL exclusions.
- Reject protected-code imports of GPL backend implementation modules.
- Review generated OpenAPI inputs and outputs rather than assuming that an API
  boundary resolves copied-code licensing.
- Require an explicit classification for shared workspace files.

### Contribution gate

- Require CLA completion for protected-path pull requests.
- Confirm that automated dependency PRs do not alter license scope or notices.
- Keep backend contribution checks independent from frontend CLA checks.

### Artifact gate

- Build the WebUI, all supported extension targets, and admin UI.
- Inspect archives, images, and store packages for BSL parameters, dates,
  source links, trademark disclaimers, and third-party notices.
- Fail release verification when metadata conflicts with the license matrix.

### Language and release gate

- Detect inaccurate "open source" claims for current BSL frontend releases.
- Verify that each Change Date is exactly 24 months after its Release Date.
- Verify that published commit and artifact identifiers match the release
  registry.
- Require documented counsel approval before the first BSL release.

## Implementation Stages

### Stage 1: Provenance and legal preparation

**Goal:** Establish that the selected frontend material can be relicensed and
turn the approved policies into attorney-reviewed documents.

**Success criteria:** Protected paths are inventoried; ambiguous material is
resolved; BSL parameters, Additional Use Grant, CLA, trademark policy, and
commercial template receive legal approval.

### Stage 2: Repository license structure

**Goal:** Establish the path-based license matrix and package notices without
changing existing GPL release history.

**Success criteria:** Every repository path has an unambiguous governing
license; third-party notices are preserved; public descriptions accurately
represent the multi-license project.

### Stage 3: Product and artifact disclosure

**Goal:** Make the applicable license visible wherever protected software is
received or used.

**Success criteria:** Source trees, About/Legal surfaces, containers, archives,
and extension packages expose consistent version, Change Date, source, and
notice information.

### Stage 4: Automated governance

**Goal:** Prevent accidental boundary, contribution, metadata, and release
violations.

**Success criteria:** CI enforces path classification, CLA status, import
boundaries, release dates, artifact notices, and approved public terminology.

### Stage 5: First protected release and transition monitoring

**Goal:** Publish the first BSL versions and maintain the rolling AGPL schedule.

**Success criteria:** Release records and artifacts pass all gates; historical
GPL versions remain intact; conversion checks track every published Change
Date.

## Risks and Mitigations

- **Existing GPL forks remain usable:** Accept this as an unavoidable property
  of prior grants; make the protected release materially better and clearly
  versioned.
- **"Substantially replaces" is ambiguous:** Use counsel-reviewed objective
  examples and a commercial inquiry channel; do not improvise custom BSL text.
- **BSL/GPL incompatibility before conversion:** Preserve an arms-length API
  boundary and statically enforce protected-path imports.
- **Contributor friction:** Use a short, plain-language CLA with retained
  contributor ownership and a transparent explanation of commercial rights.
- **Marketing confusion:** Put the license matrix and source-available wording
  in the README, product legal surfaces, and release listings.
- **Missed transitions:** Keep release dates in machine-readable records and
  monitor approaching Change Dates automatically.
- **Trademark overreach:** Permit truthful compatibility statements and have
  counsel review rebranding requirements.

## Authoritative References

- GNU GPL FAQ: <https://www.gnu.org/licenses/gpl-faq.en.html>
- GNU GPL/AGPL compatibility guidance:
  <https://www.gnu.org/licenses/license-compatibility.en.html>
- Business Source License 1.1: <https://mariadb.com/bsl11/>
- BSL adoption guidance: <https://mariadb.com/bsl-faq-adopting/>
- Functional Source License comparison: <https://fsl.software/>

## Approval Record

The user approved the selected BSL model, protected path boundary, Additional
Use Grant policy, rolling 24-month release lifecycle, `AGPL-3.0-only` Change
License, CLA, trademark policy, selective commercial licensing, and rollout
gates in the design conversation on 2026-07-19.
