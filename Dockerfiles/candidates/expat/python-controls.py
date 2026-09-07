"""Bundled-parser behavior and owning-runtime checks; never a system-library test."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import pyexpat
import resource
import statistics
import sys
import time

# Probe the actual CPython parser version; no input is supplied to ElementTree.
import xml.etree.ElementTree as ET  # nosec B405
import _elementtree


def controls() -> dict:
    """Exercise legitimate XML and the non-null-context child-parser regression."""
    counts = {"default-precedence": 0, "child-dtd-copy": 0, "namespace": 0}
    for chunk in (0, 1, 7):
        cases = (
            (
                "default-precedence",
                b"<!DOCTYPE tag [<!ATTLIST tag first CDATA ' a  b '><!ATTLIST tag first NMTOKENS 'ignored'><!ATTLIST tag second NMTOKENS ' a  b '>]><tag/>",
                [("tag", {"first": " a  b ", "second": "a b"})],
            ),
            (
                "child-dtd-copy",
                b"<!DOCTYPE doc [<!ENTITY e SYSTEM 'in-memory'><!ATTLIST tag first CDATA #IMPLIED><!ATTLIST tag second NMTOKENS #IMPLIED>]><doc>&e;</doc>",
                [("doc", {}), ("tag", {"second": "a b"})],
            ),
            (
                "namespace",
                b"<n:tag xmlns:n='urn:test' n:attribute='value'/>",
                [("urn:test|tag", {"urn:test|attribute": "value"})],
            ),
        )
        for name, payload, expected in cases:
            seen = []
            parser = pyexpat.ParserCreate(namespace_separator="|")
            parser.StartElementHandler = lambda tag, attributes: seen.append((tag, attributes))

            def external(context: str | None, base: str | None, system_id: str | None, public_id: str | None) -> int:
                if context is None or system_id != "in-memory":
                    raise ValueError("unexpected external entity boundary")
                child = parser.ExternalEntityParserCreate(context)
                child.Parse(b"<tag second=' a  b '/>", True)
                return 1

            parser.ExternalEntityRefHandler = external
            step = chunk or len(payload)
            for offset in range(0, len(payload), step):
                parser.Parse(payload[offset : offset + step], offset + step >= len(payload))
            if seen != expected:
                raise ValueError(f"legitimate XML control failed: {name}")
            counts[name] += 1
    return counts


def validate_identity(identity: dict, prefix: str) -> None:
    """Require the rebuilt interpreter, shared library, and both XML extensions."""
    if (identity.get("python_version"), identity.get("expat_version"), identity.get("elementtree_version")) != (
        "3.12.14",
        "expat_2.8.4",
        "Expat 2.8.4",
    ):
        raise ValueError("wrong Python or bundled parser version")
    for name in ("executable", "pyexpat", "elementtree", "libpython"):
        if not Path(identity[name]).resolve().is_relative_to(Path(prefix).resolve()):
            raise ValueError(f"runtime escaped candidate prefix: {name}")


def identity(prefix: str) -> dict:
    """Read the actual ELF mapping rather than a configured library search path."""
    libraries = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libpython3.12.so" in line and line.split()[-1].startswith("/")
    }
    if len(libraries) != 1:
        raise ValueError("expected exactly one mapped libpython")
    result = {
        "python_version": platform.python_version(),
        "expat_version": pyexpat.EXPAT_VERSION,
        # No XML input: inspect only the owning parser's version property.
        "elementtree_version": ET.XMLParser().version,  # nosec B314
        "executable": os.path.realpath(sys.executable),
        "pyexpat": os.path.realpath(pyexpat.__file__),
        "elementtree": os.path.realpath(_elementtree.__file__),
        "libpython": libraries.pop(),
    }
    validate_identity(result, prefix)
    return result


def measure() -> dict:
    """Use the same bounded input and timing contract, but the bundled parser."""
    spec = importlib.util.spec_from_file_location("scaling", Path(__file__).with_name("attribute-scaling.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {"version": pyexpat.EXPAT_VERSION, "measurements": {}}
    for mode in ("whole", "incremental"):
        values = []
        for count in (4000, 8000, 16000):
            payload = module.document(count)
            samples = []
            for _ in range(5):
                parser = pyexpat.ParserCreate()
                start = time.process_time()
                step = len(payload) if mode == "whole" else 65536
                for offset in range(0, len(payload), step):
                    parser.Parse(payload[offset : offset + step], offset + step >= len(payload))
                samples.append(time.process_time() - start)
            values.append(statistics.median(samples))
        result["measurements"][mode] = values
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("controls", "measure"))
    parser.add_argument("--prefix")
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    if args.mode == "controls":
        if not args.prefix:
            parser.error("controls require the expected candidate prefix")
        print(json.dumps({"identity": identity(args.prefix), "controls": controls()}, sort_keys=True))
    else:
        print(json.dumps(measure(), sort_keys=True))
