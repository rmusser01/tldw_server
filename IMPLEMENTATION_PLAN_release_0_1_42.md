# Release 0.1.42 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox syntax for tracking.

**Goal:** Publish reviewed release `0.1.42` from PR #2761 with complete metadata
and a compliant source-only protected-frontend release record.

**Architecture:** Keep the frozen `dev` and `main` ancestry already integrated.
Record the unchanged protected frontend snapshot at exact commit
`0f3983788c413e0d17ffe7eabe8cff4a9f6ae723`, while tagging the final reviewed
`main` merge commit. Publish only source, Python, and GPL backend images.

**Tech Stack:** Git, GitHub Actions/CLI, Python/pytest, MkDocs, SHA-256.

---

### Task 1: Record the protected source release

**Files:**
- Create: `LICENSES/releases/0.1.42/release.json`
- Create: `LICENSES/releases/0.1.42/PolyForm-Countdown-1.0.0.txt`
- Create: `LICENSES/releases/0.1.42/protected-files.sha256`
- Modify: `tldw_Server_API/tests/CI/test_licensing_policy.py`

- [x] Replace the obsolete “no release directories” assertion with tests for
  release ID `0.1.42`, source revision
  `0f3983788c413e0d17ffe7eabe8cff4a9f6ae723`, release date `2026-07-26`,
  Countdown timestamp `2028-07-26T12:00:00Z`, scopes, notices, source-only
  publication, and the SHA-256 digest of `protected-files.sha256`.
- [x] Run the focused test and confirm it fails because `0.1.42` is absent.
- [x] Add the record, fill the Countdown template with
  `2028-07-26` plus the verbatim AGPL text, and generate SHA-256 entries for all
  tracked protected files at the frozen source revision.
- [x] Run the focused test and exact protected-tree comparison; expect success.
- [x] Commit the record and test with `TASK-12988` in the message.

### Task 2: Add reviewed release metadata

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `Docs/RELEASE_NOTES.md`
- Modify: `Docs/Site/RELEASE_NOTES.md`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `tldw_Server_API/app/main.py`
- Modify: `Docs/mkdocs.yml`
- Regenerate: `Docs/Published/**`

- [x] Add concise `0.1.42` notes covering provider credentials, embeddings
  workflows, frontend licensing/license-first CI, Jobs hardening, and Skills
  certification.
- [x] Change all visible release surfaces from `0.1.41` to `0.1.42` without
  running the patch helper, which would calculate `0.1.43`.
- [x] Refresh `Docs/Published` twice, require an empty second diff, and run the
  strict documentation build.
- [x] Commit the metadata and generated documentation with `TASK-12988`.

### Task 3: Verify and refresh PR #2761

- [ ] Run focused release, documentation, workflow, and licensing tests.
- [ ] Run Actionlint, Bandit on touched executable Python, and
  `git diff --check`.
- [ ] Prove frozen `origin/dev=0f3983788c413e0d17ffe7eabe8cff4a9f6ae723`
  and `origin/main=d9c245ac14c40df855d1ab6cd19b3c137b16b47b`
  are both ancestors of the exact final PR head.
- [ ] Push the exact release head and require
  `frontend-license-policy/trusted/main` success on that SHA.
- [ ] Keep the PR draft until the requester reviews the legal record and writes
  the required Change summary in their own words.
- [ ] If merge/publication moves past `2026-07-26`, stop and obtain requester
  approval for new release and Countdown dates before changing either date.

### Task 4: Merge, tag, publish, and synchronize

- [ ] Merge only after Task 3; admin bypass is limited to deliberately
  cancelled ordinary checks and cannot bypass the trusted license or human
  review gates.
- [ ] Before tagging, require that neither local/remote `v0.1.42` nor GitHub
  Release `v0.1.42` already exists.
- [ ] Require `origin/main` to equal PR #2761's merge SHA, then create and push
  annotated tag `v0.1.42` on that exact commit and verify its local target.
- [ ] Create the GitHub Release with `--verify-tag` from the tagged changelog
  and verify automatic PyPI plus `app`, `worker`, and `audio-worker`
  publication and attestations.
- [ ] Confirm no protected frontend binary was published.
- [ ] Recover narrowly: if only the tag exists, create only the missing GitHub
  Release; if PyPI or Docker publication fails, rerun only that failed workflow
  against the immutable tag/commit. Never recreate the tag, release, or
  version.
- [ ] Merge released `main` back into `dev`, prove ancestry, complete
  `TASK-12988`, and resume `TASK-12986`.
