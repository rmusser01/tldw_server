# Native frontend runtime candidate renderer

This directory generates candidate-only Dockerfiles for WebUI and Admin UI. It
does not modify either canonical Dockerfile, build an application, publish an
image, or admit this design to production.

Run it from any directory and choose an output outside the canonical recipe:

```bash
python Dockerfiles/candidates/frontend/render.py \
  --application webui \
  --output /tmp/Dockerfile.webui.candidate
```

The renderer accepts only the exact Node 24.20.0 builder and runtime markers. It
preserves all bytes before the runtime marker and all bytes after it, replacing
only that marker with the candidate runtime block. Missing, duplicated, renamed,
or repinned canonical stages fail rather than producing an unreviewed recipe.

## Candidate runtime contract

The generated runtime is pinned to the reviewed Ubuntu 24.04 digest and installs
only `zlib1g=1:1.3.dfsg-3.1ubuntu2.2` through normal signed APT indexes. The build
retains the signed `InRelease` files and their SHA256 values, the exact acquired
zlib package SHA256, APT policy, and installed zlib/libc/libstdc++ versions under
`/usr/local/share/tldw-candidate-evidence`. It also requires libc6 to remain
`2.39-0ubuntu8.8`. These records bind the acquisition used by the experiment;
they are not a snapshot-reproducibility claim.

The unchanged builder supplies `/usr/local/bin/node`,
`/usr/local/bin/docker-entrypoint.sh`, and `/usr/local/LICENSE`; the candidate
restores `ENTRYPOINT ["docker-entrypoint.sh"]` explicitly. The official
[Node 24.20.0 Bookworm slim recipe](https://github.com/nodejs/docker-node/blob/c4eb0858f5c522521768d5b6dc1d9f1631d4854d/24/bookworm-slim/Dockerfile)
extracts the authenticated Node archive into `/usr/local` and copies the
entrypoint there. Direct inspection of the exact pinned amd64 artifact confirmed
all three paths and Node `v24.20.0`.

That exact artifact has no `/etc/ssl/certs`, `/usr/share/ca-certificates`, or
system CA bundle: the recipe auto-purges its temporary `ca-certificates` package.
The candidate therefore does not emit a broken CA copy. Copying the exact Node
executable preserves Node's embedded root store; native qualification must compare
its `tls.rootCertificates` hash and count and separately report any system bundle.

Unlike the baseline Node image, the candidate does not carry npm, npx, Yarn, or
the `nodejs` convenience symlink into the runtime. Both canonical applications
start with `node`, and their complete application `ENV`, `COPY`, `USER`,
`WORKDIR`, `EXPOSE`, `HEALTHCHECK`, and `CMD` remainder stays byte-for-byte
unchanged. This is not a bit-identical operating system and does not imply that
findings absent from another distribution's feed were fixed.
