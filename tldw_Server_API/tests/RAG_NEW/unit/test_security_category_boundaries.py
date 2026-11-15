"""Sensitivity categories match words without treating accountability as an account."""

import pytest

from tldw_Server_API.app.core.RAG.rag_service.security_filters import ContentFilter, SensitivityLevel


@pytest.mark.unit
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("The agency is accountable to the attorney.", SensitivityLevel.INTERNAL),
        ("The attorney recommends accountability.", SensitivityLevel.INTERNAL),
        ("Accountability and accountable leadership.", SensitivityLevel.PUBLIC),
        ("The attorney reviews a financial account.", SensitivityLevel.CONFIDENTIAL),
        ("The attorneys review bank accounts.", SensitivityLevel.CONFIDENTIAL),
        ("Banking and litigation records.", SensitivityLevel.CONFIDENTIAL),
        ("Credited payments and contracted services.", SensitivityLevel.CONFIDENTIAL),
        ("Accounting and attorneys.", SensitivityLevel.CONFIDENTIAL),
        ("Medically prescribed treatment.", SensitivityLevel.INTERNAL),
        ("Top secret: public policy discussion.", SensitivityLevel.RESTRICTED),
        ("Sensitive compartmented information.", SensitivityLevel.RESTRICTED),
        ("Confidential: ordinary prose.", SensitivityLevel.CONFIDENTIAL),
        ("A trade secret.", SensitivityLevel.CONFIDENTIAL),
    ],
)
def test_category_words_preserve_inflections_without_unrelated_substrings(content, expected):
    assert ContentFilter().classify_content(content) == expected
