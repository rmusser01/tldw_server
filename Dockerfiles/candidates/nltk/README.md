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
