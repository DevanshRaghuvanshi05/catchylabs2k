"""
Unit tests for Gemini API rate-limiting, exponential backoff, and fallback handling.
All Gemini API interactions are fully mocked. No live network calls are made.
"""

from unittest.mock import MagicMock, patch
import pytest

from evidence_extractor import (
    parse_retry_after,
    is_transient_error,
    is_hard_quota_exhausted,
    calculate_backoff,
    extract_evidence_from_doc,
    generate_queries,
    check_evidence_coverage,
)
from pipeline import extract_all_evidence
import inspect


class MockResponseWithHeaders:
    def __init__(self, headers):
        self.headers = headers


class DummyExceptionWithHeader(Exception):
    def __init__(self, message, headers):
        super().__init__(message)
        self.response = MockResponseWithHeaders(headers)


def test_parse_retry_after_from_header():
    exc = DummyExceptionWithHeader("Rate limit exceeded", {"Retry-After": "4.5"})
    assert parse_retry_after(exc) == 4.5

    exc_lower = DummyExceptionWithHeader("Rate limit exceeded", {"retry-after": "10"})
    assert parse_retry_after(exc_lower) == 10.0


def test_parse_retry_after_from_message():
    exc1 = Exception("429 Too Many Requests: Please retry after 12s.")
    assert parse_retry_after(exc1) == 12.0

    exc2 = Exception("Resource exhausted, please retry in 5.5 seconds.")
    assert parse_retry_after(exc2) == 5.5

    exc3 = Exception("Regular generic error with no delay hints")
    assert parse_retry_after(exc3) is None


def test_is_transient_error():
    assert is_transient_error(Exception("429 RESOURCE_EXHAUSTED"))
    assert is_transient_error(Exception("503 UNAVAILABLE: high demand"))
    assert is_transient_error(Exception("500 Internal Server Error"))
    assert is_transient_error(Exception("Rate limit exceeded"))
    assert not is_transient_error(Exception("Invalid argument: bad prompt format"))


def test_is_hard_quota_exhausted():
    quota_err = Exception(
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details'}}"
    )
    assert is_hard_quota_exhausted(quota_err)

    temp_rate_limit = Exception("429 Too Many Requests: Per-minute rate limit exceeded.")
    assert not is_hard_quota_exhausted(temp_rate_limit)


def test_calculate_backoff():
    # Respects Retry-After within bounds
    delay = calculate_backoff(attempt=1, retry_after=3.0, max_delay=6.0)
    assert delay == 3.0

    # Caps excessive Retry-After at max_delay
    capped_delay = calculate_backoff(attempt=1, retry_after=600.0, max_delay=5.0)
    assert capped_delay == 5.0

    # Exponential growth with jitter
    d1 = calculate_backoff(attempt=1, base_delay=1.0, max_delay=10.0, jitter_max=0.1)
    d2 = calculate_backoff(attempt=2, base_delay=1.0, max_delay=10.0, jitter_max=0.1)
    assert 1.0 <= d1 <= 1.2
    assert 2.0 <= d2 <= 2.2
    assert d2 > d1


def test_extract_evidence_bounded_retry_on_same_model():
    """Verify that a transient 429 triggers a retry on the same model instead of switching immediately to fallback."""
    client = MagicMock()
    mock_success_response = MagicMock()
    mock_success_response.text = (
        '{"claim": "Running shoe demand is up 15%", '
        '"evidence": "Demand is up 15%.", "topic": "consumer demand", "confidence": 0.8, '
        '"source": {"title": "Invented report", "url": "https://fake.example"}}'
    )
    mock_success_response.usage_metadata = None

    # Attempt 1: 429 transient error, Attempt 2: Success
    client.models.generate_content.side_effect = [
        Exception("429 RESOURCE_EXHAUSTED: temporary rate limit"),
        mock_success_response,
    ]

    doc = {
        "title": "Sport News",
        "url": "http://test.com",
        "publisher": "Search Provider",
        "snippet": "Demand is up 15%."
    }
    brief = {"brand": "Adidas", "market": "Germany"}

    with patch("time.sleep") as mock_sleep:
        res = extract_evidence_from_doc(client, doc, brief, max_retries=2)

    assert res["status"] == "success"
    assert res["data"]["claim"] == "Running shoe demand is up 15%"
    assert res["data"]["source"] == {
        "title": "Sport News",
        "url": "http://test.com",
        "publisher": "Search Provider",
        "published_date": None
    }
    assert res["data"]["evidence"] == "Demand is up 15%."
    assert client.models.generate_content.call_count == 2
    # Verify both calls targeted the first model
    first_model = client.models.generate_content.call_args_list[0].kwargs.get("model")
    second_model = client.models.generate_content.call_args_list[1].kwargs.get("model")
    assert first_model == second_model
    assert mock_sleep.called


def test_extract_evidence_rejects_unquoted_model_claim():
    client = MagicMock()
    response = MagicMock()
    response.text = (
        '{"claim": "An unsupported statistic", "evidence": "A made-up quotation.", '
        '"topic": "market", "confidence": 0.9}'
    )
    response.usage_metadata = None
    client.models.generate_content.return_value = response

    result = extract_evidence_from_doc(
        client,
        {"title": "Source", "url": "https://source.example", "snippet": "The actual source snippet."},
        {"brand": "Test", "market": "Test", "audience": "Test", "objective": "Test"},
    )

    assert result["status"] == "no_claim"
    assert "quotation" in result["reason"]


def test_extract_evidence_rejects_invalid_source_url_without_ai_call():
    client = MagicMock()
    result = extract_evidence_from_doc(
        client,
        {"title": "Source", "url": "javascript:alert(1)", "snippet": "A source snippet."},
        {"brand": "Test", "market": "Test", "audience": "Test", "objective": "Test"},
    )

    assert result["status"] == "no_claim"
    client.models.generate_content.assert_not_called()


def test_extract_evidence_hard_quota_no_fallback():
    """Verify that a hard quota exhaustion error fails gracefully immediately without calling fallback models."""
    client = MagicMock()
    quota_err = Exception("429 RESOURCE_EXHAUSTED: You exceeded your current quota, please check your plan and billing details")
    client.models.generate_content.side_effect = quota_err

    doc = {"title": "Sport News", "url": "http://test.com", "snippet": "Demand is up 15%."}
    brief = {"brand": "Adidas", "market": "Germany"}

    with patch("time.sleep"):
        res = extract_evidence_from_doc(client, doc, brief, max_retries=2)

    assert res["status"] == "failed"
    assert "Quota exhausted" in res["reason"]
    # Only 1 call made; did not retry or hit fallback models
    assert client.models.generate_content.call_count == 1


def test_extract_evidence_retries_exhausted_on_429_no_fallback():
    """Verify that if retries on model 1 are exhausted on 429, it fails gracefully without hammering fallback models."""
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

    doc = {"title": "Sport News", "url": "http://test.com", "snippet": "Demand is up 15%."}
    brief = {"brand": "Adidas", "market": "Germany"}

    with patch("time.sleep"):
        res = extract_evidence_from_doc(client, doc, brief, max_retries=2)

    assert res["status"] == "failed"
    # Exactly max_retries (2) calls on the primary model, 0 calls on fallback model
    assert client.models.generate_content.call_count == 2
    model1 = client.models.generate_content.call_args_list[0].kwargs.get("model")
    model2 = client.models.generate_content.call_args_list[1].kwargs.get("model")
    assert model1 == model2


def test_generate_queries_bounded_retry_and_fallback():
    """Verify query generation handles 429 with bounded retries and falls back to default queries gracefully."""
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

    brief = {"brand": "Pringles", "market": "India", "audience": "Youth"}

    with patch("time.sleep"):
        queries = generate_queries(client, brief, max_retries=2)

    assert isinstance(queries, list)
    assert len(queries) >= 2
    # Verify bounded call count (did not make dozens of calls)
    assert client.models.generate_content.call_count == 2


def test_extract_all_evidence_concurrency_defaults_to_one():
    """Verify extract_all_evidence defaults to max_concurrency=1."""
    sig = inspect.signature(extract_all_evidence)
    assert sig.parameters["max_concurrency"].default == 1

    client = MagicMock()
    docs = [
        {"title": "Doc 1", "url": "http://test.com/1", "snippet": "Snippet 1"},
        {"title": "Doc 2", "url": "http://test.com/2", "snippet": "Snippet 2"},
    ]
    brief = {"brand": "Pringles", "market": "India"}

    with patch("pipeline.extract_evidence_from_doc") as mock_extract:
        mock_extract.return_value = {
            "status": "success",
            "data": {"claim": "A valid claim", "source": {}},
        }
        evidence, stats = extract_all_evidence(client, docs, brief, max_concurrency=1)

    assert len(evidence) == 2
    assert stats["documents_processed"] == 2
    assert stats["documents_successful_extraction"] == 2
    assert mock_extract.call_count == 2


def test_researcher_extract_evidence_bounded_retries():
    """Verify researcher.py uses bounded retries and backoff on 429."""
    import researcher

    client = MagicMock()
    client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

    doc = {"title": "Doc", "url": "http://test.com", "snippet": "Snippet"}
    brief = {"brand": "Puma", "market": "UK"}

    with patch("time.sleep") as mock_sleep:
        res = researcher.extract_evidence_from_doc(client, doc, brief)

    assert res["status"] == "failed"
    # Verifies max 2 attempts (not 3) and bounded sleep
    assert client.models.generate_content.call_count == 2
    assert mock_sleep.call_count == 1
    sleep_arg = mock_sleep.call_args[0][0]
    assert 1.0 <= sleep_arg <= 6.0  # bounded max delay, not 8s/16s/24s


def test_researcher_extract_evidence_hard_quota_no_retry():
    """Verify researcher.py fails immediately on hard quota exhaustion."""
    import researcher

    client = MagicMock()
    client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: You exceeded your current quota, please check your plan")

    doc = {"title": "Doc", "url": "http://test.com", "snippet": "Snippet"}
    brief = {"brand": "Puma", "market": "UK"}

    with patch("time.sleep") as mock_sleep:
        res = researcher.extract_evidence_from_doc(client, doc, brief)

    assert res["status"] == "failed"
    assert client.models.generate_content.call_count == 1
    assert not mock_sleep.called


def test_check_evidence_coverage_handles_429_gracefully():
    """Verify check_evidence_coverage handles rate limits without cascading calls and returns sufficient=False."""
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: You exceeded your current quota")

    brief = {"brand": "Pringles", "market": "India"}
    evidence_items = [{"claim": "test claim", "topic": "market"}]

    with patch("time.sleep"):
        res = check_evidence_coverage(client, brief, evidence_items, max_retries=2)

    assert res["sufficient"] is False
    assert res["status"] == "failed"
    assert res["evaluated"] is False
    assert res["gaps"] == []
    assert res["additional_queries"] == []
    assert "quota" in res["error"].lower() or "rate limit" in res["error"].lower()
    assert client.models.generate_content.call_count == 1
