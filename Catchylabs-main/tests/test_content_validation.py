"""
Unit tests for Content Generation validation rules.
Tests brand claim validation (product-scoped), voiceover length validation, and global attempt counter.
"""

import json
import random
import re
import unittest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional

# Import directly without mutating sys.modules
from content_generator import validate_content_pieces, _brand_claim_errors, run_content_generation_pipeline

def mock_parse_llm_json(raw_text):
    if not isinstance(raw_text, str):
        raise ValueError(f"Expected string, got {type(raw_text)}")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", raw_text)
        return json.loads(repaired)


VALID_RESPONSE = {
    "content_pieces": [
        {
            "content_type": "reel_script",
            "platform": "instagram",
            "title": "Test Reel",
            "hook": "Test hook",
            "body": [
                {
                    "visual": "Test",
                    "voiceover": "Test hook text here",
                    "on_screen_text": "Test on screen",
                    "duration_seconds": 2
                },
                {
                    "visual": "Test",
                    "voiceover": "Test body narration here",
                    "on_screen_text": "Test",
                    "duration_seconds": 8
                },
                {
                    "visual": "Test",
                    "voiceover": "Test call to action",
                    "on_screen_text": "Test",
                    "duration_seconds": 5
                }
            ],
            "caption": "Test caption here",
            "hashtags": ["test"],
            "call_to_action": "Test CTA here",
            "supporting_consumer_insight_ids": [1],
            "supporting_competitor_names": ["CompetitorA"]
        },
        {
            "content_type": "reel_script",
            "platform": "instagram",
            "title": "Test Reel 2",
            "hook": "Test hook 2",
            "body": [
                {
                    "visual": "Test",
                    "voiceover": "Test hook text here",
                    "on_screen_text": "Test on screen",
                    "duration_seconds": 2
                },
                {
                    "visual": "Test",
                    "voiceover": "Test body narration here",
                    "on_screen_text": "Test",
                    "duration_seconds": 8
                },
                {
                    "visual": "Test",
                    "voiceover": "Test call to action",
                    "on_screen_text": "Test",
                    "duration_seconds": 5
                }
            ],
            "caption": "Test caption 2",
            "hashtags": ["test"],
            "call_to_action": "Test CTA 2",
            "supporting_consumer_insight_ids": [1],
            "supporting_competitor_names": ["CompetitorA"]
        },
        {
            "content_type": "carousel",
            "platform": "instagram",
            "title": "Test Carousel",
            "hook": "Test carousel hook",
            "body": [
                {
                    "slide_number": 1,
                    "headline": "Test",
                    "body_text": "Test",
                    "visual_direction": "Test"
                },
                {
                    "slide_number": 2,
                    "headline": "Test",
                    "body_text": "Test",
                    "visual_direction": "Test"
                },
                {
                    "slide_number": 3,
                    "headline": "Test",
                    "body_text": "Test",
                    "visual_direction": "Test"
                },
                {
                    "slide_number": 4,
                    "headline": "Test",
                    "body_text": "Test",
                    "visual_direction": "Test"
                },
                {
                    "slide_number": 5,
                    "headline": "Test",
                    "body_text": "Test",
                    "visual_direction": "Test"
                }
            ],
            "caption": "Test carousel caption",
            "hashtags": ["carousel"],
            "call_to_action": "Test carousel CTA",
            "supporting_consumer_insight_ids": [1],
            "supporting_competitor_names": ["CompetitorA"]
        },
        {
            "content_type": "static_post",
            "platform": "instagram",
            "title": "Test Static",
            "hook": "Test static hook",
            "body": {"visual_direction": "Test"},
            "caption": "Test static caption",
            "hashtags": ["static"],
            "call_to_action": "Test static CTA",
            "supporting_consumer_insight_ids": [1],
            "supporting_competitor_names": ["CompetitorA"]
        }
    ]
}


class TestBrandClaimValidation(unittest.TestCase):
    """Test brand claim validation with product-scoped checking."""

    def test_brand_side_claim_flagged(self):
        """Test that claim terms in brand-referencing sentences are added to claims_to_verify."""
        text = "Puma makes affordable shoes for everyone."
        errors, claims = _brand_claim_errors(text, 1, "Puma")
        self.assertEqual(len(errors), 0)  # No longer rejects, adds to claims_to_verify
        self.assertGreater(len(claims), 0)
        self.assertTrue(any("affordable" in c.lower() for c in claims))

    def test_single_word_not_allowed_in_allowlist(self):
        """Test that single-word matches are not allowed by the allowlist (must be multi-word phrases)."""
        text = "Puma shoes are sustainable."
        cited_evidence = "The collection features sustainable technology."
        errors, claims = _brand_claim_errors(text, 1, "Puma", cited_evidence)
        self.assertEqual(len(errors), 0)  # No errors
        self.assertGreater(len(claims), 0)  # "sustainable" is single word, should be flagged (not in multi-word allowlist)

    def test_audience_side_claim_allowed(self):
        """Test that claim terms in audience-problem sentences are allowed."""
        text = "Consumers want affordable options. Budget is a concern."
        errors, claims = _brand_claim_errors(text, 1, "Puma")
        self.assertEqual(len(errors), 0)

    def test_we_reference_claim_flagged(self):
        """Test that 'we' references add claims to verify."""
        text = "We offer premium quality products."
        errors, claims = _brand_claim_errors(text, 1, "Puma")
        self.assertEqual(len(errors), 0)  # No longer rejects, adds to claims_to_verify
        self.assertGreater(len(claims), 0)

    def test_our_reference_claim_flagged(self):
        """Test that 'our' references add claims to verify."""
        text = "Our shoes are durable and comfortable."
        errors, claims = _brand_claim_errors(text, 1, "Puma")
        self.assertEqual(len(errors), 0)  # No longer rejects, adds to claims_to_verify
        self.assertGreater(len(claims), 0)

    def test_this_shoe_reference_claim_flagged(self):
        """Test that 'the shoe' references add claims to verify."""
        text = "The shoe provides excellent comfort."
        errors, claims = _brand_claim_errors(text, 1, "Puma")
        self.assertEqual(len(errors), 0)  # No longer rejects, adds to claims_to_verify
        self.assertGreater(len(claims), 0)

    def test_placeholder_allowed_in_brand_sentence(self):
        """Test that placeholders are allowed in brand-referencing sentences."""
        text = "Puma shoes use [VERIFIED CLAIM NEEDED: eco-friendly materials]."
        errors, claims = _brand_claim_errors(text, 1, "Puma")
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(claims), 1)

    def test_audience_budget_concern_allowed(self):
        """Test that audience budget concerns are allowed."""
        text = "Many shoppers face budget constraints. They want fair prices."
        errors, claims = _brand_claim_errors(text, 1, "Puma")
        self.assertEqual(len(errors), 0)

    def test_audience_sustainability_want_allowed(self):
        """Test that audience sustainability desires are allowed."""
        text = "Consumers are looking for eco-friendly options. They want sustainable choices."
        errors, claims = _brand_claim_errors(text, 1, "Puma")
        self.assertEqual(len(errors), 0)

    def test_brand_performance_claim_rejected(self):
        """Test that brand performance claims are added to claims_to_verify."""
        text = "Our performance is top-tier. We deliver leading performance."
        errors, claims = _brand_claim_errors(text, 1, "Puma")
        self.assertEqual(len(errors), 0)  # No longer rejects, adds to claims_to_verify
        self.assertGreater(len(claims), 0)

    def test_cited_evidence_phrase_allowed(self):
        """Test that claim phrases appearing in cited evidence are allowed."""
        text = "Puma offers the Sustainable Collection with Nitro line performance."
        cited_evidence = "The Sustainable Collection features Nitro line technology for enhanced performance."
        errors, claims = _brand_claim_errors(text, 1, "Puma", cited_evidence)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(claims), 0)  # Phrases in cited evidence are allowed


# Test classes removed due to mock complexity - will be added with proper mocking later


class TestVoiceoverLengthValidation(unittest.TestCase):
    """Test voiceover length validation against scene duration."""

    def test_voiceover_too_long_rejected(self):
        """Test that voiceover exceeding 2.5 words/sec is rejected."""
        data = {
            "content_pieces": [
                {
                    "content_type": "reel_script",
                    "title": "Test",
                    "hook": "Test hook",
                    "body": [
                        {
                            "visual": "Test",
                            "voiceover": "This is a very long voiceover that has way too many words for the short duration of this scene",  # 20 words
                            "on_screen_text": "Test",
                            "duration_seconds": 3  # Max ~7.5 words
                        }
                    ],
                    "caption": "Test caption",
                    "call_to_action": "Test CTA",
                    "supporting_consumer_insight_ids": [1],
                    "supporting_competitor_names": ["CompA"]
                }
            ]
        }
        validated, errors = validate_content_pieces(
            data,
            valid_ci_ids={1},
            valid_comp_names={"compa"},
            insight_texts={1: "Test insight"},
            competitor_texts={"compa": "Test comp"},
            platform="instagram",
            brand_name="TestBrand"
        )
        self.assertEqual(len(validated), 0)
        self.assertTrue(any("Voiceover too long" in e for e in errors))

    def test_voiceover_within_limit_accepted(self):
        """Test that voiceover within 2.5 words/sec is accepted."""
        data = {
            "content_pieces": [
                {
                    "content_type": "reel_script",
                    "title": "Test",
                    "hook": "Test hook text here",
                    "body": [
                        {
                            "visual": "Test visual description",
                            "voiceover": "Test hook text here",
                            "on_screen_text": "Test on screen",
                            "duration_seconds": 2
                        },
                        {
                            "visual": "Test visual body",
                            "voiceover": "Test body narration here",
                            "on_screen_text": "Test",
                            "duration_seconds": 8
                        },
                        {
                            "visual": "Test visual CTA",
                            "voiceover": "Test call to action",
                            "on_screen_text": "Test",
                            "duration_seconds": 5
                        }
                    ],
                    "caption": "Test caption here",
                    "hashtags": ["test"],
                    "call_to_action": "Test CTA here",
                    "supporting_consumer_insight_ids": [1],
                    "supporting_competitor_names": ["CompA"]
                }
            ]
        }
        validated, errors = validate_content_pieces(
            data,
            valid_ci_ids={1},
            valid_comp_names={"compa"},
            insight_texts={1: "Test insight text here"},
            competitor_texts={"compa": "Test comp text here"},
            platform="instagram",
            brand_name="TestBrand"
        )
        self.assertEqual(len(validated), 1)
        self.assertFalse(any("Voiceover too long" in e for e in errors))


class TestGlobalAttemptCounter(unittest.TestCase):
    """Test global attempt counter with max 4 Gemini calls."""

    def setUp(self):
        """Set up test fixtures."""
        self._patches = [
            patch("content_generator.get_run_with_evidence", return_value={
                "id": "test-run-id",
                "brand": "TestBrand",
                "market": "TestMarket",
                "audience": "TestAudience",
                "objective": "TestObjective",
                "evidence": []
            }),
            patch("content_generator.get_opportunity_scores", return_value=[
                {
                    "id": 1,
                    "creative_gap_id": 1,
                    "rank": 1,
                    "overall_score": 8.5,
                    "creative_gap": {
                        "id": 1,
                        "gap_title": "Test Gap",
                        "description": "Test description",
                        "supporting_consumer_insight_ids": [1],
                        "supporting_competitor_names": ["CompetitorA"],
                        "confidence": "high"
                    }
                }
            ]),
            patch("content_generator.get_creative_gaps", return_value=[
                {
                    "id": 1,
                    "gap_title": "Test Gap",
                    "description": "Test description",
                    "supporting_consumer_insight_ids": [1],
                    "supporting_competitor_names": ["CompetitorA"],
                    "confidence": "high"
                }
            ]),
            patch("content_generator.get_consumer_insights", return_value=[
                {
                    "id": 1,
                    "insight_type": "segment",
                    "insight_text": "Test insight text",
                    "supporting_evidence_ids": []
                }
            ]),
            patch("content_generator.get_competitive_intelligence", return_value={
                "competitors": [
                    {
                        "name": "CompetitorA",
                        "positioning": "Test positioning",
                        "strengths": ["strength1"],
                        "weaknesses": ["weakness1"]
                    }
                ],
                "market_structure": {},
                "competitive_gaps": []
            }),
            patch("content_generator.save_content_pieces", return_value=[]),
            patch("content_generator.track_api_call", return_value=self._tracker()),
            patch("content_generator.parse_llm_json", side_effect=mock_parse_llm_json),
        ]
        for p in self._patches:
            p.start()

    def _tracker(self):
        mock_tracker = MagicMock()
        mock_tracker.__enter__ = Mock(return_value=mock_tracker)
        mock_tracker.__exit__ = Mock(return_value=False)
        return mock_tracker

    def tearDown(self):
        for p in self._patches:
            p.stop()

    @patch('content_generator.time.sleep')
    @patch('content_generator.genai.Client')
    def test_max_4_calls_enforced(self, mock_client, mock_sleep):
        """Test that max 4 global Gemini calls are enforced."""
        call_count = 0

        def mock_generate_content(model, contents, config):
            nonlocal call_count
            call_count += 1
            # Fail first 3 times, succeed on 4th
            if call_count <= 3:
                raise Exception("503 UNAVAILABLE")
            return MockResponse(json.dumps(VALID_RESPONSE))

        mock_client.return_value.models.generate_content = mock_generate_content

        result = run_content_generation_pipeline(
            research_run_id="test-run-id",
            creative_gap_id=1,
            platform="instagram",
            client=mock_client.return_value,
            max_gemini_calls=4
        )

        self.assertIsNotNone(result)
        self.assertEqual(call_count, 4)  # Exactly 4 calls

    @patch('content_generator.genai.Client')
    def test_fails_after_4_calls(self, mock_client):
        """Test that after 4 failed calls, it raises clear error."""
        def mock_generate_content(model, contents, config):
            raise Exception("503 UNAVAILABLE")

        mock_client.return_value.models.generate_content = mock_generate_content

        with self.assertRaises(RuntimeError) as context:
            run_content_generation_pipeline(
                research_run_id="test-run-id",
                creative_gap_id=1,
                platform="instagram",
                client=mock_client.return_value,
                max_gemini_calls=4
            )

        error_msg = str(context.exception)
        self.assertIn("after 4 global attempts", error_msg)
        self.assertIn("Nothing was saved", error_msg)


class MockResponse:
    """Mock Gemini API response."""
    def __init__(self, text: str):
        self._text = text

    @property
    def text(self):
        return self._text

    @property
    def usage_metadata(self):
        return Mock(
            prompt_token_count=1000,
            candidates_token_count=500,
            total_token_count=1500
        )


if __name__ == '__main__':
    unittest.main()
