"""Bounded, locally generated CVE-2026-66046 CPU-scaling control for libexpat.

This measures the explicitly named system library, never Python's bundled
parser. Comparisons fail closed; timings alone do not qualify the other copy.
"""

import argparse
import ctypes
import json
import math
import resource
import statistics
import time
from pathlib import Path


def document(count: int) -> bytes:
    """Generate tokenized attributes that enter the vulnerable default lookup."""
    if not 1 <= count <= 16000:
        raise ValueError("attribute count outside bounded control")
    declarations = "".join(f"<!ATTLIST tag a{i} NMTOKENS #IMPLIED>" for i in range(count))
    attributes = " ".join(f"a{i}=' \tvalue  '" for i in range(count))
    return f"<!DOCTYPE tag [{declarations}]><tag {attributes}/>".encode()


def measure(library: str) -> dict:
    """Measure median process CPU time with explicit memory and CPU limits."""
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    lib = ctypes.CDLL(library)
    lib.XML_ParserCreate.argtypes = [ctypes.c_char_p]
    lib.XML_ParserCreate.restype = ctypes.c_void_p
    lib.XML_Parse.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.XML_Parse.restype = ctypes.c_int
    lib.XML_ParserFree.argtypes = [ctypes.c_void_p]
    lib.XML_ParserFree.restype = None
    lib.XML_ExpatVersion.restype = ctypes.c_char_p
    result = {"version": lib.XML_ExpatVersion().decode(), "measurements": {}}
    for mode in ("whole", "incremental"):
        result["measurements"][mode] = []
        for count in (4000, 8000, 16000):
            payload = document(count)
            samples = []
            for _ in range(5):
                parser = lib.XML_ParserCreate(None)
                if not parser:
                    raise ValueError("parser allocation failed")
                try:
                    start = time.process_time()
                    step = len(payload) if mode == "whole" else 65536
                    for offset in range(0, len(payload), step):
                        piece = payload[offset : offset + step]
                        if lib.XML_Parse(parser, piece, len(piece), offset + step >= len(payload)) != 1:
                            raise ValueError("generated control XML failed to parse")
                    samples.append(time.process_time() - start)
                finally:
                    lib.XML_ParserFree(parser)
            result["measurements"][mode].append(statistics.median(samples))
    return result


def compare(baseline: dict, candidate: dict) -> None:
    """Require the pinned copies, baseline reproduction, and improved scaling."""
    if baseline.get("version") != "expat_2.8.3" or candidate.get("version") != "expat_2.8.4":
        raise ValueError("unexpected parser versions")
    for result in (baseline, candidate):
        timings = result.get("measurements", {})
        if set(timings) != {"whole", "incremental"}:
            raise ValueError("missing parsing mode")
        for values in timings.values():
            if len(values) != 3 or any(
                not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0 for value in values
            ):
                raise ValueError("invalid timing evidence")
    for mode in ("whole", "incremental"):
        old = baseline["measurements"][mode]
        new = candidate["measurements"][mode]
        if old[2] / old[1] <= 3:
            raise ValueError(f"baseline quadratic behavior was not reproduced: {mode}")
        if new[2] / new[1] >= 3 or new[2] >= old[2] * 0.6:
            raise ValueError(f"candidate scaling or improvement failed: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("measure", "compare"))
    parser.add_argument("first")
    parser.add_argument("second", nargs="?")
    args = parser.parse_args()
    if args.mode == "measure":
        print(json.dumps(measure(args.first), indent=2))
    else:
        if not args.second:
            parser.error("compare requires both baseline and candidate JSON paths")
        compare(json.loads(Path(args.first).read_text()), json.loads(Path(args.second).read_text()))
        print("PASS: bounded whole/incremental attribute scaling")
