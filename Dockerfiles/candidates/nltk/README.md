# NLTK 3.10.3 candidate source

This directory prepares the reviewed candidate-only source for
TASK-13013.7.17. It does not download, install, build, publish, waive, or admit
NLTK. Upstream still identifies the resulting runtime and metadata version as
`3.10.3`; later wheel construction uses build tag `1tldw1`.

## Prepare from retained local inputs

```bash
python Dockerfiles/candidates/nltk/prepare.py \
  --inputs /path/to/verified-inputs \
  --output /new/output/path
```

The input directory must contain the source archive and every primary,
prerequisite, and parent-source file declared by `source-inputs.json`. The
script verifies their SHA256 values, validates every tar member before manually
extracting regular files/directories, applies `backport.patch` with fuzz disabled,
and rejects any source inventory or pre/post hash drift. The output path must not
already exist.

Successful output has exactly this interface for Task 2:

```text
output/
├── source/                  # complete prepared nltk-3.10.3 source root
└── source-provenance.json   # schemaVersion 1, admitted:false
```

`source-provenance.json` binds the source archive, all original upstream inputs,
the adapted patch, and the ten changed-file pre/post hashes. Its scope is
`nltk-candidate-source-not-release-admission`. The exact selected and omitted
hunk map is recorded in
`Docs/Design/TASK-13013.7.17-nltk-candidate-backport.md`.

## Current review status

The dependency-isolated prepared-source import closure passed in the pinned
no-network Linux/amd64 Python 3.11.16 container. Review controls also found that
a stateful path-like value made `PerceptronTagger.save_to_json` create a
forbidden sibling directory before its later descriptor check rejected the
operation; no model or canary bytes changed. After that concrete result, the
requester approved a minimal five-caller correction to consume the exact strings
returned by `validate_tool_path`/`validate_tool_dir`. The local correction is
documented separately from the exact upstream hunks in the design. The current
patch SHA256 is
`d5071d450b140c5d37ad5db59c05aa2b9542c62e92ff885739d425baf878c251`.
Stage 1 is independently reviewed and remains candidate-only; nothing here
admits the candidate or changes a production dependency.

The partial Task 2 artifact verifier statically checks candidate wheel ZIP
members, top-level metadata, RECORD hashes, and byte identity against the
caller-supplied prepared source. It records the caller-supplied provenance hash
but does not independently authenticate upstream source. Its tests use
synthetic archives only. No candidate wheel has been built or validated, and no
installed-environment or runtime-closure claim follows.

The 2026-09-09 independent static review and scoped re-review are complete.
The verifier rejects noncanonical ZIP/RECORD paths and file/directory aliases,
requires one expected Name, Version and Build header, and supports exactly one
`Wheel-Version: 1.0` header. The corrections were demonstrated with failing
synthetic fixtures before implementation. Final focused verification passed
29 tests with five existing warnings; Black, Ruff and compile-only checks
passed. Verifier Bandit reported zero findings. Test-only Bandit retained 13
Low-severity B101 assertion findings without suppressions.

All 22 pins in `requirements-qualification.txt` were rechecked against retained
wheel SHA256 values and exactly one top-level METADATA name/version per wheel.
This is an artifact identity audit, not dependency-closure verification.
The existing implementation plan and TASK-13013.7.17 retain the evidence paths,
review results and platform restriction. Installed-wheel qualification and
application/scanner evidence remain incomplete; AC2/AC3 stay open and
`admitted:false` is unchanged. The full Supply_Chain suite was not rerun for
this static follow-up.
