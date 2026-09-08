#!/bin/bash
# Actual candidate controls. Caller supplies only a writable evidence directory.
set -euo pipefail
test "$(id -u)" = 10001
test "$(readlink -f /opt/tldw-venv/bin/python)" = /usr/local/bin/python3.12
cd /
sha256sum --check --strict /opt/combined-evidence/installed-binaries.sha256 > /evidence/python-binaries.log
dpkg --verify libexpat1 > /evidence/expat-package-verify.txt
test ! -s /evidence/expat-package-verify.txt
test "$(dpkg-query -W -f='${Version}' libexpat1)" = '2.8.4-1~deb13u1+tldw1'
python /opt/combined/python-controls.py controls --prefix /usr/local > /evidence/python-controls.json
env -u LD_LIBRARY_PATH /opt/tldw-venv/bin/python /opt/combined/python-controls.py controls --prefix /usr/local > /evidence/python-controls-no-ld-path.json
python - <<'PY' > /evidence/system-parsers.json
import ctypes
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import runpy

expected_tools = runpy.run_path("/opt/combined/test-tools.py")["VERSIONS"]
installed_tools = {}
for distribution in importlib.metadata.distributions(path=["/opt/expat-test-tools"]):
    name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
    if name in installed_tools:
        raise ValueError("duplicate isolated test tool")
    installed_tools[name] = distribution.version
if installed_tools != expected_tools:
    raise ValueError("isolated test tools differ from the reviewed cohort")
Path("/evidence/test-tools.json").write_text(json.dumps(installed_tools, sort_keys=True) + "\n")

result = {}
for name in ("libexpat.so.1", "libexpatw.so.1"):
    path = Path("/usr/lib/x86_64-linux-gnu") / name
    library = ctypes.CDLL(str(path))
    library.XML_ExpatVersion.restype = ctypes.c_char_p
    version = library.XML_ExpatVersion().decode()
    if version != "expat_2.8.4":
        raise ValueError("wrong installed Expat version")
    result[name] = {"path": str(path.resolve()), "version": version, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps(result, sort_keys=True))
PY
cd /app
python /opt/combined/run-tests.py --report /evidence/application-tests.json
fc-match sans --format '%{file}\n' > /evidence/font-match.txt
test -s /evidence/font-match.txt
test -f "$(head -n 1 /evidence/font-match.txt)"
ffmpeg -hide_banner -nostdin -f lavfi -i color=c=black:s=160x90:d=0.1 \
    -vf 'drawtext=text=candidate:fontcolor=white:fontsize=12' -frames:v 1 -f null - \
    > /evidence/drawtext.log 2>&1
printf '1\n00:00:00,000 --> 00:00:01,000\ncandidate\n' > /evidence/control.srt
ffmpeg -hide_banner -nostdin -f lavfi -i color=c=black:s=160x90:d=0.1 \
    -vf subtitles=/evidence/control.srt -frames:v 1 -f null - \
    > /evidence/subtitles.log 2>&1
for binary in /opt/tldw-ffmpeg9/bin/* /opt/tldw-ffmpeg9/lib/*.so; do
    ldd "$binary"
done > /evidence/ffmpeg-ldd.txt
if grep -Fq 'not found' /evidence/ffmpeg-ldd.txt; then
    echo 'Unresolved FFmpeg library' >&2
    exit 1
fi
if grep -Eq 'lib(avcodec\.so\.61|avdevice\.so\.61|avfilter\.so\.10|avformat\.so\.61|avutil\.so\.59|postproc\.so\.58|swresample\.so\.5|swscale\.so\.8)' /evidence/ffmpeg-ldd.txt; then
    echo 'Unexpected retired FFmpeg library' >&2
    exit 1
fi
printf '0\n' > /evidence/runtime-controls.exit
