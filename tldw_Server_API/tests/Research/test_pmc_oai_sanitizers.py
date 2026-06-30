import pytest

from tldw_Server_API.app.core.Third_Party import PMC_OAI as pmc_oai


pytestmark = pytest.mark.unit


def test_pmc_oai_identify_sanitizes_xml_failures(monkeypatch):
    def fail_get_xml(_params):
        raise RuntimeError("pmc oai token at /private/pmc-oai.key")

    monkeypatch.setattr(pmc_oai, "_get_xml", fail_get_xml)

    info, error = pmc_oai.pmc_oai_identify()

    assert info is None
    assert error == "PMC OAI-PMH Identify request failed."
    assert "pmc oai token" not in error
    assert "/private/pmc-oai.key" not in error


def test_pmc_oai_identify_preserves_timeout_classification(monkeypatch):
    def fail_get_xml(_params):
        raise TimeoutError("timed out at /private/pmc-oai-timeout.key")

    monkeypatch.setattr(pmc_oai, "_get_xml", fail_get_xml)

    info, error = pmc_oai.pmc_oai_identify()

    assert info is None
    assert error == "PMC OAI-PMH Identify request timed out."
    assert "timed out at" not in error
    assert "/private/pmc-oai-timeout.key" not in error


def test_pmc_oai_list_sets_sanitizes_xml_failures(monkeypatch):
    def fail_get_xml(_params):
        raise RuntimeError("pmc oai sets token at /private/pmc-oai-sets.key")

    monkeypatch.setattr(pmc_oai, "_get_xml", fail_get_xml)

    items, token, error = pmc_oai.pmc_oai_list_sets()

    assert items is None
    assert token is None
    assert error == "PMC OAI-PMH ListSets request failed."
    assert "pmc oai sets token" not in error
    assert "/private/pmc-oai-sets.key" not in error


def test_pmc_oai_list_sets_preserves_timeout_classification(monkeypatch):
    def fail_get_xml(_params):
        raise TimeoutError("timed out at /private/pmc-oai-sets-timeout.key")

    monkeypatch.setattr(pmc_oai, "_get_xml", fail_get_xml)

    items, token, error = pmc_oai.pmc_oai_list_sets()

    assert items is None
    assert token is None
    assert error == "PMC OAI-PMH ListSets request timed out."
    assert "timed out at" not in error
    assert "/private/pmc-oai-sets-timeout.key" not in error


def test_pmc_oai_list_records_sanitizes_xml_failures(monkeypatch):
    def fail_get_xml(_params):
        raise RuntimeError("pmc oai records token at /private/pmc-oai-records.key")

    monkeypatch.setattr(pmc_oai, "_get_xml", fail_get_xml)

    items, token, error = pmc_oai.pmc_oai_list_records()

    assert items is None
    assert token is None
    assert error == "PMC OAI-PMH ListRecords request failed."
    assert "pmc oai records token" not in error
    assert "/private/pmc-oai-records.key" not in error


def test_pmc_oai_list_records_preserves_timeout_classification(monkeypatch):
    def fail_get_xml(_params):
        raise TimeoutError("timed out at /private/pmc-oai-records-timeout.key")

    monkeypatch.setattr(pmc_oai, "_get_xml", fail_get_xml)

    items, token, error = pmc_oai.pmc_oai_list_records()

    assert items is None
    assert token is None
    assert error == "PMC OAI-PMH ListRecords request timed out."
    assert "timed out at" not in error
    assert "/private/pmc-oai-records-timeout.key" not in error


def test_pmc_oai_list_identifiers_sanitizes_xml_failures(monkeypatch):
    def fail_get_xml(_params):
        raise RuntimeError("pmc oai ids token at /private/pmc-oai-ids.key")

    monkeypatch.setattr(pmc_oai, "_get_xml", fail_get_xml)

    items, token, error = pmc_oai.pmc_oai_list_identifiers()

    assert items is None
    assert token is None
    assert error == "PMC OAI-PMH ListIdentifiers request failed."
    assert "pmc oai ids token" not in error
    assert "/private/pmc-oai-ids.key" not in error


def test_pmc_oai_list_identifiers_preserves_timeout_classification(monkeypatch):
    def fail_get_xml(_params):
        raise TimeoutError("timed out at /private/pmc-oai-ids-timeout.key")

    monkeypatch.setattr(pmc_oai, "_get_xml", fail_get_xml)

    items, token, error = pmc_oai.pmc_oai_list_identifiers()

    assert items is None
    assert token is None
    assert error == "PMC OAI-PMH ListIdentifiers request timed out."
    assert "timed out at" not in error
    assert "/private/pmc-oai-ids-timeout.key" not in error


def test_pmc_oai_get_record_sanitizes_xml_failures(monkeypatch):
    def fail_get_xml(_params):
        raise RuntimeError("pmc oai record token at /private/pmc-oai-record.key")

    monkeypatch.setattr(pmc_oai, "_get_xml", fail_get_xml)

    item, error = pmc_oai.pmc_oai_get_record("oai:pubmedcentral.nih.gov:123")

    assert item is None
    assert error == "PMC OAI-PMH GetRecord request failed."
    assert "pmc oai record token" not in error
    assert "/private/pmc-oai-record.key" not in error
    assert "oai:pubmedcentral.nih.gov:123" not in error


def test_pmc_oai_get_record_preserves_timeout_classification(monkeypatch):
    def fail_get_xml(_params):
        raise TimeoutError("timed out at /private/pmc-oai-record-timeout.key")

    monkeypatch.setattr(pmc_oai, "_get_xml", fail_get_xml)

    item, error = pmc_oai.pmc_oai_get_record("oai:pubmedcentral.nih.gov:123")

    assert item is None
    assert error == "PMC OAI-PMH GetRecord request timed out."
    assert "timed out at" not in error
    assert "/private/pmc-oai-record-timeout.key" not in error
