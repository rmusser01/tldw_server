"""Compare retained CPython JUnit results; fail on missing coverage or new skips."""

import argparse
from collections import Counter
import json
from pathlib import Path

# Only bounded UTF-8, DTD-free local test reports reach this parser (below).
import xml.etree.ElementTree as ET  # nosec B405

SUITES = {"test_pyexpat", "test_xml_etree", "test_xml_etree_c", "test_minidom", "test_sax"}


def read_suite(path: Path) -> dict:
    """Read only locally generated test evidence, including exact skip reasons."""
    with path.open("rb") as stream:
        data = stream.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("test report exceeds size limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("test report must use UTF-8") from error
    if "\x00" in text:
        raise ValueError("test report must use UTF-8 without null bytes")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("DTD is not valid test evidence")
    # Input has been bounded, decoded as UTF-8 and checked for DTDs above.
    root = ET.fromstring(text)  # nosec B314
    cases = root.findall(".//testcase")
    represented = set()
    skipped = []
    for case in cases:
        name = case.get("name", "")
        parts = name.split(".")
        if len(parts) < 3 or parts[0] != "test" or parts[1] not in SUITES:
            raise ValueError("unexpected XML test identity")
        if any(case.find(tag) is not None for tag in ("failure", "error", "output", "outcome")):
            raise ValueError("XML suite contains a failure or unexpected outcome")
        skip = case.find("skipped")
        if skip is None:
            represented.add(parts[1])
        else:
            skipped.append((name, skip.text or ""))
    if represented != SUITES:
        raise ValueError("missing executed XML suite")
    return {"tests": len(cases), "skipped": sorted(skipped), "names": Counter(case.get("name") for case in cases)}


def compare(baseline: Path, candidate: Path) -> dict:
    """Require every suite and no new skips compared with the same source tests."""
    old, new = read_suite(baseline), read_suite(candidate)
    if old["names"] - new["names"] or not set(new["skipped"]).issubset(set(old["skipped"])):
        raise ValueError("candidate lost tests or introduced skips")
    return {"tests": new["tests"], "skipped": new["skipped"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate), sort_keys=True))
