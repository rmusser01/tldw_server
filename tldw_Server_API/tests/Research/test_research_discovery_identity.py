def test_fingerprint_prefers_doi_over_url_and_title():
    from tldw_Server_API.app.core.Research.discovery.identity import build_fingerprint

    first = build_fingerprint(
        {"doi": "10.1000/Example", "url": "https://a.test/paper", "title": "A"}
    )
    second = build_fingerprint(
        {"doi": "https://doi.org/10.1000/example", "url": "https://b.test/other", "title": "B"}
    )

    assert first == second
    assert first.startswith("doi:")


def test_merge_records_preserves_all_provenance_and_primary_source():
    from tldw_Server_API.app.core.Research.discovery.identity import normalize_and_merge_records

    results = normalize_and_merge_records(
        [
            {
                "source_id": "openalex",
                "provider": "openalex",
                "doi": "10.1000/example",
                "title": "Paper",
            },
            {
                "source_id": "crossref",
                "provider": "crossref",
                "doi": "10.1000/example",
                "title": "Paper",
            },
        ],
        catalog_version="research-discovery-v1",
    )

    assert len(results) == 1
    assert results[0].primary_source_id == "openalex"
    assert {item.source_id for item in results[0].merged_provenance} == {
        "openalex",
        "crossref",
    }


def test_safe_provider_metadata_drops_raw_url_containers():
    from tldw_Server_API.app.core.Research.discovery.identity import safe_provider_metadata

    metadata = safe_provider_metadata(
        {
            "title": "Paper",
            "raw_urls": ["https://repo.example/paper.pdf?token=SECRET"],
            "url": "https://repo.example/paper.pdf?X-Amz-Signature=SECRET",
            "links": [{"url": "https://repo.example/other.pdf?token=SECRET"}],
        }
    )

    assert metadata == {"title": "Paper"}


def test_safe_provider_metadata_drops_reviewer_sensitive_key_probe():
    from tldw_Server_API.app.core.Research.discovery.identity import safe_provider_metadata

    metadata = safe_provider_metadata(
        {
            "access_token": "SECRET",
            "api-key": "SECRET",
            "header": "Bearer SECRET",
            "file": "raw.pdf",
            "link": "https://x.test?token=SECRET",
            "safe": "ok",
        }
    )

    assert metadata == {"safe": "ok"}


def test_safe_provider_metadata_drops_common_sensitive_key_separators():
    from tldw_Server_API.app.core.Research.discovery.identity import safe_provider_metadata

    metadata = safe_provider_metadata(
        {
            "download-url": "https://x.test/paper.pdf?token=SECRET",
            "pdf url": "https://x.test/paper.pdf?token=SECRET",
            "apiKey": "SECRET",
            "raw-files": ["paper.pdf"],
            "safe_field": "ok",
        }
    )

    assert metadata == {"safe_field": "ok"}


def test_safe_provider_metadata_drops_url_like_values_with_sensitive_material():
    from tldw_Server_API.app.core.Research.discovery.identity import safe_provider_metadata

    metadata = safe_provider_metadata(
        {
            "href": "https://repo.example/paper.pdf?token=SECRET",
            "uri": "https://repo.example/paper.pdf#SECRET",
            "openAccessPdf": {
                "href": "https://repo.example/other.pdf?authToken=SECRET",
                "label": "repository copy",
            },
            "safe": "ok",
        }
    )

    assert metadata == {"openAccessPdf": {"label": "repository copy"}, "safe": "ok"}


def test_normalize_and_merge_records_drops_unsafe_urls_from_safe_metadata_and_provenance():
    from tldw_Server_API.app.core.Research.discovery.identity import normalize_and_merge_records

    results = normalize_and_merge_records(
        [
            {
                "source_id": "openalex",
                "provider": "openalex",
                "doi": "10.1000/example",
                "title": "Paper",
                "href": "https://repo.example/paper.pdf?token=SECRET",
                "best_oa_location": {
                    "href": "https://repo.example/best.pdf?authToken=SECRET",
                    "host_type": "repository",
                },
            }
        ],
        catalog_version="research-discovery-v1",
    )

    result = results[0]
    assert "href" not in result.safe_metadata
    assert result.safe_metadata["best_oa_location"] == {"host_type": "repository"}
    assert "href" not in result.merged_provenance[0].safe_metadata
    assert result.merged_provenance[0].safe_metadata["best_oa_location"] == {
        "host_type": "repository"
    }
    assert "SECRET" not in str(result.safe_metadata)
    assert "SECRET" not in str(result.merged_provenance[0].safe_metadata)


def test_signed_oa_url_is_redacted_from_response_snapshot_and_candidate_id():
    from tldw_Server_API.app.core.Research.discovery.oa import build_oa_candidates

    raw_url = "https://repo.example/files/paper.pdf?X-Amz-Signature=SECRET&Expires=999"
    candidates = build_oa_candidates(
        result_fingerprint="doi:10.1000/example",
        source_id="openalex",
        provider="openalex",
        doi="10.1000/example",
        raw_urls=[raw_url],
    )

    candidate = candidates[0]
    assert candidate.url_redacted is True
    assert candidate.safe_url == "https://repo.example/files/paper.pdf"
    assert "SECRET" not in candidate.candidate_id
    assert "X-Amz-Signature" not in candidate.candidate_id
    assert candidate.resolver_reference is not None
    assert candidate.requires_reresolution is True


def test_oa_candidate_url_strips_userinfo_and_all_query_strings():
    from tldw_Server_API.app.core.Research.discovery.oa import build_oa_candidates

    candidates = build_oa_candidates(
        result_fingerprint="doi:10.1000/example",
        source_id="openalex",
        provider="openalex",
        doi="10.1000/example",
        raw_urls=["https://user:pass@repo.example:8443/files/paper.pdf?download=1"],
    )

    candidate = candidates[0]
    assert candidate.url_redacted is True
    assert candidate.safe_url == "https://repo.example:8443/files/paper.pdf"
    assert candidate.requires_reresolution is True
    assert "user" not in candidate.candidate_id
    assert "pass" not in candidate.candidate_id
    assert "download=1" not in candidate.candidate_id


def test_oa_candidate_url_strips_unknown_tokenish_query_names():
    from tldw_Server_API.app.core.Research.discovery.oa import build_oa_candidates

    candidates = build_oa_candidates(
        result_fingerprint="doi:10.1000/example",
        source_id="openalex",
        provider="openalex",
        doi="10.1000/example",
        raw_urls=["https://repo.example/files/paper.pdf?authToken=SECRET"],
    )

    candidate = candidates[0]
    assert candidate.url_redacted is True
    assert candidate.safe_url == "https://repo.example/files/paper.pdf"
    assert "authToken" not in candidate.candidate_id
    assert "SECRET" not in candidate.candidate_id


def test_unpaywall_resolver_wraps_doi_lookup_and_sanitizes_signed_pdf_url():
    from tldw_Server_API.app.core.Research.discovery.oa import ResearchOAResolver

    calls = []

    def fake_resolve_oa_pdf(doi):
        calls.append(doi)
        return "https://repo.example/paper.pdf?token=SECRET", None

    resolver = ResearchOAResolver(resolve_oa_pdf_fn=fake_resolve_oa_pdf)
    candidates = resolver.resolve_for_result(
        result_fingerprint="doi:10.1000/example",
        source_id="unpaywall",
        provider="unpaywall",
        doi="10.1000/example",
        provider_ids={"doi": "10.1000/example"},
        raw_urls=[],
    )

    assert calls == ["10.1000/example"]
    assert candidates[0].url_redacted is True
    assert candidates[0].safe_url == "https://repo.example/paper.pdf"
    assert "SECRET" not in candidates[0].candidate_id
