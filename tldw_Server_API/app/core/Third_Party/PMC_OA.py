"""
PMC_OA.py

PMC OA Web Service adapter (https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi)

Supports identification and result set queries with resumptionToken. Also
exposes a simple PDF download helper for PMC articles.
"""
from __future__ import annotations

import contextlib
from typing import Any
from defusedxml import ElementTree as DET

from tldw_Server_API.app.core.http_client import fetch

BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


def _get_xml(params: dict[str, Any]) -> Any:
    r = fetch(method="GET", url=BASE_URL, params=params, headers={"Accept": "application/xml"}, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}")
    try:
        return DET.fromstring(r.text)
    finally:
        with contextlib.suppress(Exception):
            r.close()


def pmc_oa_identify() -> tuple[dict[str, Any] | None, str | None]:
    try:
        root = _get_xml({})
        info: dict[str, Any] = {}
        # Extract simple counts and formats when present
        resp_date = root.find("responseDate")
        repo = root.find("repositoryName")
        formats = [f.text for f in root.findall("formats/format") if f is not None and f.text]
        latest = root.find("records/latest")
        info.update({
            "responseDate": resp_date.text if resp_date is not None else None,
            "repositoryName": repo.text if repo is not None else None,
            "formats": formats or None,
            "latest": latest.text if latest is not None else None,
        })
        return info, None
    except TimeoutError:
        return None, "PMC OA Identify request timed out."
    except Exception:
        return None, "PMC OA Identify request failed."


def _parse_resumption(root: Any) -> str | None:
    res = root.find("resumption")
    if res is None:
        return None
    link = res.find("link")
    if link is not None:
        token = link.attrib.get("token")
        return token
    return None


def pmc_oa_query(
    from_date: str | None = None,
    until_date: str | None = None,
    fmt: str | None = None,  # 'pdf' or 'tgz'
    resumption_token: str | None = None,
    id_param: str | None = None,
) -> tuple[list[dict[str, Any]] | None, str | None, str | None]:
    """Query OA Web Service for record links.

    Returns (items, resumption_token, error_message)
    """
    try:
        params: dict[str, Any] = {}
        if id_param:
            params["id"] = id_param
        elif resumption_token:
            params["resumptionToken"] = resumption_token
        else:
            if from_date:
                params["from"] = from_date
            if until_date:
                params["until"] = until_date
            if fmt:
                params["format"] = fmt
        root = _get_xml(params)
        records: list[dict[str, Any]] = []
        for rec in root.findall("records/record"):
            rid = rec.attrib.get("id")
            citation = rec.attrib.get("citation")
            license_attr = rec.attrib.get("license")
            retracted = rec.attrib.get("retracted")
            links = []
            for link in rec.findall("link"):
                links.append({
                    "format": link.attrib.get("format"),
                    "updated": link.attrib.get("updated"),
                    "href": link.attrib.get("href"),
                })
            records.append({
                "id": rid,
                "citation": citation,
                "license": license_attr,
                "retracted": (retracted == "yes") if retracted is not None else None,
                "links": links,
            })
        return records, _parse_resumption(root), None
    except TimeoutError:
        return None, None, "PMC OA query request timed out."
    except Exception:
        return None, None, "PMC OA query request failed."


def download_pmc_pdf(pmcid: str) -> tuple[bytes | None, str | None, str | None]:
    """Download a PMC PDF by PMCID numeric portion.

    Returns (content_bytes, filename, error_message)
    """
    try:
        pmcid_num = str(pmcid).strip().lstrip("PMC")
        if not pmcid_num:
            return None, None, "PMCID cannot be empty"
        url = f"https://pmc.ncbi.nlm.nih.gov/PMC{pmcid_num}/pdf"
        r = fetch(method="GET", url=url, timeout=30)
        if r.status_code >= 400:
            return None, None, f"PMC PDF HTTP error: {r.status_code}"
        # Best-effort filename from headers; default to PMC{pmcid}.pdf
        filename = f"PMC{pmcid_num}.pdf"
        cd = r.headers.get("Content-Disposition")
        if cd and "filename=" in cd:
            filename = cd.split("filename=")[-1].strip('"')
        return r.content, filename, None
    except TimeoutError:
        return None, None, "PMC PDF download timed out."
    except Exception:
        return None, None, "PMC PDF download failed."
