"""
PMC_OAI.py

PMC OAI-PMH provider adapter for Identify, ListSets, ListIdentifiers, ListRecords, GetRecord.

Uses the production base URL: https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/

Parses oai_dc metadata to extract title, creators, identifiers and license URLs when present.
Returns normalized dictionaries and resumption tokens where applicable.
"""
from __future__ import annotations

import contextlib
from typing import Any
from defusedxml import ElementTree as DET
from defusedxml.common import DefusedXmlException

from tldw_Server_API.app.core.http_client import fetch

BASE_URL = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
_PMC_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    DET.ParseError,
    DefusedXmlException,
)


def _get_xml(params: dict[str, Any]) -> Any:
    """Perform a GET to PMC OAI-PMH and return parsed XML root."""
    r = fetch(method="GET", url=BASE_URL, params=params, headers={"Accept": "application/xml"}, timeout=20)
    if r.status_code >= 400:
        # Let caller handle via generic error path
        raise RuntimeError(f"HTTP {r.status_code}")
    try:
        return DET.fromstring(r.text)
    finally:
        with contextlib.suppress(AttributeError, OSError):
            r.close()


def pmc_oai_identify() -> tuple[dict[str, Any] | None, str | None]:
    try:
        root = _get_xml({"verb": "Identify"})
        info: dict[str, Any] = {}
        ident = root.find(".//{http://www.openarchives.org/OAI/2.0/}Identify")
        if ident is None:
            return {"raw_xml": r_text(root)}, None
        # Extract some common fields
        def text(tag: str) -> str | None:
            el = ident.find(f"{{http://www.openarchives.org/OAI/2.0/}}{tag}")
            return el.text if el is not None else None

        info.update({
            "repositoryName": text("repositoryName"),
            "baseURL": text("baseURL"),
            "protocolVersion": text("protocolVersion"),
            "earliestDatestamp": text("earliestDatestamp"),
            "deletedRecord": text("deletedRecord"),
            "granularity": text("granularity"),
        })
        return info, None
    except TimeoutError:
        return None, "PMC OAI-PMH Identify request timed out."
    except _PMC_NONCRITICAL_EXCEPTIONS:
        return None, "PMC OAI-PMH Identify request failed."


def r_text(el: Any) -> str:
    return DET.tostring(el, encoding="unicode")


def _parse_resumption(root: Any) -> str | None:
    ns = {"oai": "http://www.openarchives.org/OAI/2.0/"}
    r = root.find(".//oai:resumptionToken", ns)
    return r.text.strip() if r is not None and r.text else None


def pmc_oai_list_sets(resumption_token: str | None = None) -> tuple[list[dict[str, Any]] | None, str | None, str | None]:
    try:
        params: dict[str, Any] = {"verb": "ListSets"}
        if resumption_token:
            params = {"verb": "ListSets", "resumptionToken": resumption_token}
        root = _get_xml(params)
        sets: list[dict[str, Any]] = []
        ns = {"oai": "http://www.openarchives.org/OAI/2.0/"}
        for se in root.findall(".//oai:set", ns):
            spec = se.find("oai:setSpec", ns)
            name = se.find("oai:setName", ns)
            sets.append({
                "setSpec": spec.text if spec is not None else None,
                "setName": name.text if name is not None else None,
            })
        return sets, _parse_resumption(root), None
    except TimeoutError:
        return None, None, "PMC OAI-PMH ListSets request timed out."
    except _PMC_NONCRITICAL_EXCEPTIONS:
        return None, None, "PMC OAI-PMH ListSets request failed."


def _parse_dc_metadata(md: Any) -> dict[str, Any]:
    # oai_dc namespace
    ns = {
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    out: dict[str, Any] = {
        "title": None,
        "creators": [],
        "identifiers": [],
        "rights": [],
        "license_urls": [],
        "date": None,
        "pmcid": None,
        "pmid": None,
        "doi": None,
    }
    t = md.find(".//dc:title", ns)
    out["title"] = t.text if t is not None else None
    for c in md.findall(".//dc:creator", ns):
        if c.text:
            out["creators"].append(c.text)
    for i in md.findall(".//dc:identifier", ns):
        if i.text:
            val = i.text.strip()
            out["identifiers"].append(val)
            # Extract pmcid/pmid/doi from canonical URLs when present
            if "pmc.ncbi.nlm.nih.gov" in val and "/PMC" in val:
                try:
                    # canonical PMC URL looks like https://pmc.ncbi.nlm.nih.gov/PMC1234567
                    pmcid = val.split("/PMC")[-1]
                    out["pmcid"] = pmcid
                except (AttributeError, IndexError, TypeError, ValueError):
                    pass
            if "pubmed.ncbi.nlm.nih.gov" in val:
                with contextlib.suppress(AttributeError, IndexError, TypeError, ValueError):
                    out["pmid"] = val.rstrip('/').split('/')[-1]
            if "doi.org/" in val:
                with contextlib.suppress(AttributeError, IndexError, TypeError, ValueError):
                    out["doi"] = val.split("doi.org/")[-1]
    for r in md.findall(".//dc:rights", ns):
        if r.text:
            txt = r.text.strip()
            out["rights"].append(txt)
            if txt.startswith("http://") or txt.startswith("https://"):
                out["license_urls"].append(txt)
    d = md.find(".//dc:date", ns)
    out["date"] = d.text if d is not None else None
    return out


def _parse_records(root: Any) -> list[dict[str, Any]]:
    ns = {"oai": "http://www.openarchives.org/OAI/2.0/"}
    items: list[dict[str, Any]] = []
    for rec in root.findall(".//oai:record", ns):
        header = rec.find("oai:header", ns)
        meta = rec.find("oai:metadata", ns)
        item: dict[str, Any] = {}
        if header is not None:
            id_el = header.find("oai:identifier", ns)
            ds = header.find("oai:datestamp", ns)
            ss = [se.text for se in header.findall("oai:setSpec", ns) if se is not None and se.text]
            item["header"] = {
                "identifier": id_el.text if id_el is not None else None,
                "datestamp": ds.text if ds is not None else None,
                "setSpecs": ss if ss else None,
            }
        if meta is not None and len(meta):
            # expect child to be oai_dc:dc or pmc/pmc_fm
            md_child = list(meta)[0]
            if md_child.tag.endswith("dc"):
                item["metadata"] = _parse_dc_metadata(md_child)
            else:
                item["raw_xml"] = r_text(md_child)
        items.append(item)
    return items


def pmc_oai_list_records(
    metadata_prefix: str = "oai_dc",
    from_date: str | None = None,
    until_date: str | None = None,
    set_name: str | None = None,
    resumption_token: str | None = None,
) -> tuple[list[dict[str, Any]] | None, str | None, str | None]:
    try:
        params: dict[str, Any] = {"verb": "ListRecords"}
        if resumption_token:
            params["resumptionToken"] = resumption_token
        else:
            params["metadataPrefix"] = metadata_prefix
            if from_date:
                params["from"] = from_date
            if until_date:
                params["until"] = until_date
            if set_name:
                params["set"] = set_name
        root = _get_xml(params)
        items = _parse_records(root)
        return items, _parse_resumption(root), None
    except TimeoutError:
        return None, None, "PMC OAI-PMH ListRecords request timed out."
    except _PMC_NONCRITICAL_EXCEPTIONS:
        return None, None, "PMC OAI-PMH ListRecords request failed."


def pmc_oai_list_identifiers(
    metadata_prefix: str = "oai_dc",
    from_date: str | None = None,
    until_date: str | None = None,
    set_name: str | None = None,
    resumption_token: str | None = None,
) -> tuple[list[dict[str, Any]] | None, str | None, str | None]:
    try:
        params: dict[str, Any] = {"verb": "ListIdentifiers"}
        if resumption_token:
            params["resumptionToken"] = resumption_token
        else:
            params["metadataPrefix"] = metadata_prefix
            if from_date:
                params["from"] = from_date
            if until_date:
                params["until"] = until_date
            if set_name:
                params["set"] = set_name
        root = _get_xml(params)
        ns = {"oai": "http://www.openarchives.org/OAI/2.0/"}
        items: list[dict[str, Any]] = []
        for he in root.findall(".//oai:header", ns):
            id_el = he.find("oai:identifier", ns)
            ds = he.find("oai:datestamp", ns)
            ss = [se.text for se in he.findall("oai:setSpec", ns) if se is not None and se.text]
            items.append({
                "identifier": id_el.text if id_el is not None else None,
                "datestamp": ds.text if ds is not None else None,
                "setSpecs": ss if ss else None,
            })
        return items, _parse_resumption(root), None
    except TimeoutError:
        return None, None, "PMC OAI-PMH ListIdentifiers request timed out."
    except _PMC_NONCRITICAL_EXCEPTIONS:
        return None, None, "PMC OAI-PMH ListIdentifiers request failed."


def pmc_oai_get_record(
    identifier: str,
    metadata_prefix: str = "oai_dc",
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if not identifier or not identifier.strip():
            return None, "Identifier cannot be empty"
        params = {"verb": "GetRecord", "identifier": identifier.strip(), "metadataPrefix": metadata_prefix}
        root = _get_xml(params)
        items = _parse_records(root)
        if not items:
            return None, None
        return items[0], None
    except TimeoutError:
        return None, "PMC OAI-PMH GetRecord request timed out."
    except _PMC_NONCRITICAL_EXCEPTIONS:
        return None, "PMC OAI-PMH GetRecord request failed."
