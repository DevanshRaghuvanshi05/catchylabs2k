"""
Standalone content-validator unit tests.
=========================================
Does NOT touch the database and does NOT need TEST_DATABASE_URL.
All Gemini and SerpAPI calls are mocked.

Covers:
  1. Whole-phrase allowlist (cited multi-word phrases pass, bare single words don't).
  2. Price/affordability terms are captured as brand claims.
  3. 3 words/second voiceover rule.
  4. Per-scene word budgets (from the prompt specification).
  5. Five NEW brand-claim sentences that MUST be flagged (never used to tune rules).
  6. Five NEW audience-side sentences that MUST pass (never used to tune rules).
  7. Rejection reason logged for every failed validation attempt.
  8. Mocked failing run — full rejection log captured and asserted.
  9. Gemini call cap = 4 total (API retries + validation retries combined).
"""

import io
import json
import logging
import re
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# ---------------------------------------------------------------------------
# Stub out heavy dependencies so we never hit the real DB / config / Gemini
# ---------------------------------------------------------------------------
_mock_config = MagicMock()
_mock_config.GEMINI_API_KEY = "test-key"
_mock_config.GEMINI_MODEL_NAME = "gemini-test"
_mock_config.GEMINI_FALLBACK_MODELS = ["gemini-test"]

_mock_db = MagicMock()
_mock_ee = MagicMock()
_mock_ee.clean_json_response = lambda x: x

sys.modules.setdefault("config", _mock_config)
sys.modules.setdefault("database", _mock_db)
sys.modules.setdefault("evidence_extractor", _mock_ee)

# ---------------------------------------------------------------------------
# Import after stubbing
# ---------------------------------------------------------------------------
from content_generator import (  # noqa: E402
    _brand_claim_errors,
    validate_content_pieces,
    run_content_generation_pipeline,
)
import content_generator as _cg  # noqa: E402

# Patch parse_llm_json so we can pass dicts directly in tests
def _parse_passthrough(raw_text):
    if isinstance(raw_text, str):
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", raw_text)
            return json.loads(repaired)
    return raw_text

_cg.parse_llm_json = _parse_passthrough

# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------

def _vol_piece(voiceover_words: int, duration_s: float) -> dict:
    """Build a minimal reel piece with one long scene and one filler scene."""
    return {
        "content_type": "reel_script",
        "title": "Test Reel",
        "hook": "Catchy hook line",
        "body": [
            {
                "visual": "Opening shot",
                "voiceover": " ".join(f"word{i}" for i in range(voiceover_words)),
                "on_screen_text": "Text",
                "duration_seconds": duration_s,
            },
            {
                "visual": "Closing shot",
                "voiceover": "Short line here",
                "on_screen_text": "CTA",
                "duration_seconds": 20 - duration_s,  # total >= 15 s
            },
        ],
        "caption": "Caption text here",
        "call_to_action": "Learn more now",
        "supporting_consumer_insight_ids": [1],
        "supporting_competitor_names": ["CompA"],
    }


def _validate_single_reel(voiceover_words: int, duration_s: float):
    data = {"content_pieces": [_vol_piece(voiceover_words, duration_s)]}
    return validate_content_pieces(
        data,
        valid_ci_ids={1},
        valid_comp_names={"CompA"},
        insight_texts={1: "test insight"},
        competitor_texts={"compa": "test comp"},
        platform="instagram",
        brand_name="TestBrand",
    )


# ---------------------------------------------------------------------------
# 1. Whole-phrase allowlist
# ---------------------------------------------------------------------------

class TestWholePhrase(unittest.TestCase):
    """Whole-phrase allowlist: only multi-word verbatim cited phrases are exempted."""

    def test_single_word_sustainable_flagged(self):
        _, claims = _brand_claim_errors(
            "Puma shoes are sustainable.", 1, "Puma", cited_evidence=""
        )
        self.assertGreater(len(claims), 0)

    def test_multiword_cited_phrase_passes(self):
        text = "Puma offers the Sustainable Collection with Nitro line performance."
        cited = "The Sustainable Collection features Nitro line technology for enhanced performance."
        _, claims = _brand_claim_errors(text, 1, "Puma", cited_evidence=cited)
        self.assertEqual(len(claims), 0)

    def test_partial_cited_word_not_enough(self):
        text = "Our shoes use sustainable materials and eco-friendly processes."
        cited = "Some sustainable notes from the research."
        _, claims = _brand_claim_errors(text, 1, "Puma", cited_evidence=cited)
        self.assertGreater(len(claims), 0)


# ---------------------------------------------------------------------------
# 2. Price / affordability terms
# ---------------------------------------------------------------------------

class TestPriceAffordabilityTerms(unittest.TestCase):
    BRAND_PRICE_LINES = [
        "Puma offers prices that are hard to beat.",
        "Our shoes cost less without sacrificing quality.",
        "These sneakers are affordable for everyone.",
        "We keep our costs low so you don't have to compromise.",
        "Puma delivers premium performance at a fair price.",
    ]

    AUDIENCE_PRICE_LINES = [
        "Many consumers struggle with rising prices.",
        "Budget-conscious shoppers are looking for value.",
        "Affordability is a top concern for Gen Z buyers.",
        "People want good quality without paying too much.",
        "Rising costs make it hard for families to buy new gear.",
    ]

    def test_brand_price_claims_flagged(self):
        for line in self.BRAND_PRICE_LINES:
            _, claims = _brand_claim_errors(line, 1, "Puma")
            self.assertGreater(len(claims), 0,
                               f"Brand price claim not flagged: {line!r}")

    def test_audience_price_concerns_pass(self):
        for line in self.AUDIENCE_PRICE_LINES:
            errors, claims = _brand_claim_errors(line, 1, "Puma")
            self.assertEqual(len(errors), 0)
            self.assertEqual(len(claims), 0,
                             f"Audience price concern incorrectly flagged: {line!r}")


# ---------------------------------------------------------------------------
# 3. Voiceover 3 words/second rule
# ---------------------------------------------------------------------------

class TestVoiceoverWordBudget(unittest.TestCase):
    def test_exactly_at_limit_passes(self):
        validated, errors = _validate_single_reel(voiceover_words=9, duration_s=3)
        self.assertFalse(any("Voiceover too long" in e for e in errors))

    def test_one_over_limit_rejected(self):
        validated, errors = _validate_single_reel(voiceover_words=10, duration_s=3)
        self.assertTrue(any("Voiceover too long" in e for e in errors))

    def test_long_scene_high_budget_passes(self):
        validated, errors = _validate_single_reel(voiceover_words=30, duration_s=10)
        self.assertFalse(any("Voiceover too long" in e for e in errors))

    def test_long_scene_over_budget_rejected(self):
        validated, errors = _validate_single_reel(voiceover_words=31, duration_s=10)
        self.assertTrue(any("Voiceover too long" in e for e in errors))


# ---------------------------------------------------------------------------
# 4. Per-scene word budget (each scene checked independently)
# ---------------------------------------------------------------------------

class TestPerSceneWordBudget(unittest.TestCase):
    def test_only_over_budget_scene_flagged(self):
        data = {
            "content_pieces": [
                {
                    "content_type": "reel_script",
                    "title": "Multi-scene reel",
                    "hook": "Strong hook line",
                    "body": [
                        # Scene 1: 3 words / 2 s (1.5 w/s) — passes
                        {"visual": "Opening", "voiceover": "Three words only",
                         "on_screen_text": "OST", "duration_seconds": 2},
                        # Scene 2: 25 words / 5 s (5 w/s) — rejected
                        {"visual": "Middle",
                         "voiceover": "one two three four five six seven eight nine ten "
                                      "eleven twelve thirteen fourteen fifteen sixteen "
                                      "seventeen eighteen nineteen twenty twenty-one "
                                      "twenty-two twenty-three twenty-four twenty-five",
                         "on_screen_text": "OST", "duration_seconds": 5},
                        # Scene 3: 6 words / 8 s — passes
                        {"visual": "Closing", "voiceover": "Shop Puma today and enjoy",
                         "on_screen_text": "CTA", "duration_seconds": 8},
                    ],
                    "caption": "Caption here",
                    "call_to_action": "Buy now",
                    "supporting_consumer_insight_ids": [1],
                    "supporting_competitor_names": ["CompA"],
                }
            ]
        }
        validated, errors = validate_content_pieces(
            data, valid_ci_ids={1}, valid_comp_names={"CompA"},
            insight_texts={1: "test"}, competitor_texts={"compa": "test"},
            brand_name="TestBrand",
        )
        vo_errors = [e for e in errors if "Voiceover too long" in e]
        self.assertEqual(len(vo_errors), 1)
        self.assertIn("scene #2", vo_errors[0])


# ---------------------------------------------------------------------------
# 5. Five NEW brand-claim sentences — must be flagged
# ---------------------------------------------------------------------------

class TestNewBrandClaimsFlagged(unittest.TestCase):
    """Five brand-claim sentences never used to tune the rules."""

    CASES = [
        ("Puma crafts every sneaker from premium recycled fibres.",
         "unverified material/premium claim"),
        ("Our latest collection is the most affordable running shoe on the market.",
         "unverified affordability superlative"),
        ("These trainers guarantee three years of durable wear under any condition.",
         "unverified durability guarantee"),
        ("We are an eco-friendly brand committed to sustainable production.",
         "unverified eco/environmental claim"),
        ("This shoe's cushioning delivers unmatched comfort for everyday runners.",
         "unverified comfort superlative"),
    ]

    def _check(self, text):
        return _brand_claim_errors(text, 1, "Puma", cited_evidence="")

    def test_claim_1_premium_recycled_fibres(self):
        _, claims = self._check(self.CASES[0][0])
        self.assertGreater(len(claims), 0, self.CASES[0][1])

    def test_claim_2_most_affordable(self):
        _, claims = self._check(self.CASES[1][0])
        self.assertGreater(len(claims), 0, self.CASES[1][1])

    def test_claim_3_durable_three_years(self):
        _, claims = self._check(self.CASES[2][0])
        self.assertGreater(len(claims), 0, self.CASES[2][1])

    def test_claim_4_carbon_neutral(self):
        _, claims = self._check(self.CASES[3][0])
        self.assertGreater(len(claims), 0, self.CASES[3][1])

    def test_claim_5_unmatched_comfort(self):
        _, claims = self._check(self.CASES[4][0])
        self.assertGreater(len(claims), 0, self.CASES[4][1])

    @classmethod
    def tearDownClass(cls):
        print("\n" + "=" * 110)
        print("BRAND-CLAIM PASS/FAIL TABLE")
        print("{:<72} {:<35} {}".format("Sentence", "Expected", "Result"))
        print("-" * 110)
        for text, reason in cls.CASES:
            _, claims = _brand_claim_errors(text, 1, "Puma", cited_evidence="")
            result = "FLAGGED OK" if claims else "MISSED X"
            print("{:<72} {:<35} {}".format(text[:70], reason[:33], result))
        print("=" * 110)


# ---------------------------------------------------------------------------
# 6. Five NEW audience-side sentences — must pass
# ---------------------------------------------------------------------------

class TestNewAudienceLinePass(unittest.TestCase):
    """Five audience-side sentences never used to tune the rules."""

    CASES = [
        "Millions of athletes replace worn-out trainers every six months.",
        "Young professionals prioritise comfort when commuting on foot.",
        "Eco-conscious buyers research a brand's sustainability record before purchasing.",
        "Price is often the deciding factor for first-time sneaker buyers.",
        "Shoppers in urban markets are increasingly aware of quality differences between brands.",
    ]

    def _check(self, text):
        return _brand_claim_errors(text, 1, "Puma", cited_evidence="")

    def test_audience_1(self):
        e, c = self._check(self.CASES[0])
        self.assertEqual(c, [], self.CASES[0])

    def test_audience_2(self):
        e, c = self._check(self.CASES[1])
        self.assertEqual(c, [], self.CASES[1])

    def test_audience_3(self):
        e, c = self._check(self.CASES[2])
        self.assertEqual(c, [], self.CASES[2])

    def test_audience_4(self):
        e, c = self._check(self.CASES[3])
        self.assertEqual(c, [], self.CASES[3])

    def test_audience_5(self):
        e, c = self._check(self.CASES[4])
        self.assertEqual(c, [], self.CASES[4])

    @classmethod
    def tearDownClass(cls):
        print("\n" + "=" * 100)
        print("AUDIENCE-LINE PASS/FAIL TABLE")
        print("{:<88} {}".format("Sentence", "Result"))
        print("-" * 100)
        for text in cls.CASES:
            _, claims = _brand_claim_errors(text, 1, "Puma", cited_evidence="")
            result = "PASS" if not claims else "FAIL ({})".format(claims[0][:30])
            print("{:<88} {}".format(text[:86], result))
        print("=" * 100)


# ---------------------------------------------------------------------------
# 7. Rejection reason logging
# ---------------------------------------------------------------------------

class TestRejectionReasonLogging(unittest.TestCase):
    """validate_content_pieces emits structured WARNING logs for each rejection."""

    def _bad_data(self):
        # 25 words in 3 s exceeds the 9-word budget
        return {
            "content_pieces": [{
                "content_type": "reel_script",
                "title": "Bad Reel",
                "hook": "Opening hook",
                "body": [
                    {"visual": "V",
                     "voiceover": " ".join(f"w{i}" for i in range(25)),
                     "on_screen_text": "OST", "duration_seconds": 3},
                    {"visual": "V2", "voiceover": "Short close",
                     "on_screen_text": "CTA", "duration_seconds": 12},
                ],
                "caption": "Cap text",
                "call_to_action": "Buy now",
                "supporting_consumer_insight_ids": [1],
                "supporting_competitor_names": ["CompA"],
            }]
        }

    def test_rejection_reasons_logged_as_warnings(self):
        cg_logger = logging.getLogger("CatchyLabContentGenerator")
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        cg_logger.addHandler(handler)
        cg_logger.setLevel(logging.WARNING)

        try:
            validated, errors = validate_content_pieces(
                self._bad_data(),
                valid_ci_ids={1}, valid_comp_names={"CompA"},
                insight_texts={1: "insight"}, competitor_texts={"compa": "comp"},
                brand_name="Puma",
            )
        finally:
            cg_logger.removeHandler(handler)

        log_output = log_stream.getvalue()

        self.assertEqual(len(validated), 0, "Bad piece should not pass validation")
        self.assertIn("Content validation REJECTED", log_output)

        rejection_lines = [l for l in log_output.splitlines()
                           if "REJECTION REASON" in l]
        self.assertGreater(len(rejection_lines), 0,
                           "At least one REJECTION REASON line must be logged")

        print("\n" + "=" * 80)
        print("CAPTURED REJECTION LOG (mocked failing run)")
        print("=" * 80)
        print(log_output.strip())
        print("=" * 80)


# ---------------------------------------------------------------------------
# 8. Gemini call cap = 4
# ---------------------------------------------------------------------------

class TestGeminiCallCap(unittest.TestCase):
    """run_content_generation_pipeline must never exceed max_gemini_calls=4."""

    _DB = {
        "run": {"id": "r1", "brand": "Puma", "market": "India",
                "audience": "Gen Z", "objective": "Growth", "evidence": []},
        "opps": [{"id": 1, "creative_gap_id": 1, "rank": 1,
                  "overall_score": 8.0, "is_stale": False}],
        "gaps": [{"id": 1, "gap_title": "Gap", "description": "Desc",
                  "supporting_consumer_insight_ids": [1],
                  "supporting_competitor_names": ["CompA"],
                  "confidence": "high"}],
        "cis": [{"id": 1, "insight_type": "segment", "insight_text": "Test",
                 "supporting_evidence_ids": []}],
        "comp": {"competitors": [{"name": "CompA", "positioning": "P",
                                   "strengths": [], "weaknesses": []}],
                 "competitive_gaps": []},
    }

    def _tracker(self):
        t = MagicMock()
        t.__enter__ = Mock(return_value=t)
        t.__exit__ = Mock(return_value=False)
        return t

    def _patches(self):
        d = self._DB
        return [
            patch("content_generator.get_run_with_evidence", return_value=d["run"]),
            patch("content_generator.get_opportunity_scores", return_value=d["opps"]),
            patch("content_generator.get_creative_gaps", return_value=d["gaps"]),
            patch("content_generator.get_consumer_insights", return_value=d["cis"]),
            patch("content_generator.get_competitive_intelligence", return_value=d["comp"]),
            patch("content_generator.save_content_pieces", return_value=[]),
            patch("content_generator.track_api_call", return_value=self._tracker()),
            patch("content_generator.time.sleep"),
        ]

    def test_api_failures_capped_at_4(self):
        """All 4 API calls fail with 503 — exactly 4 attempts made."""
        call_count = 0

        def always_fail(*a, **kw):
            nonlocal call_count
            call_count += 1
            raise Exception("503 UNAVAILABLE")

        client = MagicMock()
        client.models.generate_content.side_effect = always_fail

        ctx = self._patches()
        for p in ctx:
            p.start()
        try:
            with self.assertRaises((RuntimeError, ValueError)):
                run_content_generation_pipeline(
                    "r1", creative_gap_id=1, client=client, max_gemini_calls=4)
        finally:
            for p in ctx:
                p.stop()

        self.assertEqual(call_count, 4,
                         f"Expected 4 Gemini calls, made {call_count}")

    def test_validation_retries_count_toward_cap(self):
        """
        Validation retries consume the same global cap as API retries.

        When all 4 Gemini responses fail content validation, the pipeline
        makes exactly 4 calls (never more) and raises ValueError.
        This proves validation retries are counted against the 4-call cap.
        """
        # A response that always fails validation: voiceover way over budget
        bad_piece = {
            "content_type": "reel_script",
            "title": "BadReel",
            "hook": "Some hook",
            "body": [
                {"visual": "V",
                 "voiceover": " ".join("word{}".format(i) for i in range(40)),
                 "on_screen_text": "OST", "duration_seconds": 3},   # 40 w / 3 s
                {"visual": "V2", "voiceover": "ok here", "on_screen_text": "CTA",
                 "duration_seconds": 12},
            ],
            "caption": "Caption text",
            "call_to_action": "CTA here",
            "supporting_consumer_insight_ids": [1],
            "supporting_competitor_names": ["CompA"],
        }
        bad = {"content_pieces": [bad_piece]}

        call_count = 0

        def always_bad(*a, **kw):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            r.text = json.dumps(bad)
            r.usage_metadata = MagicMock(
                prompt_token_count=100, candidates_token_count=50,
                total_token_count=150)
            return r

        client = MagicMock()
        client.models.generate_content.side_effect = always_bad

        ctx = self._patches()
        for p in ctx:
            p.start()
        try:
            with self.assertRaises((RuntimeError, ValueError)):
                run_content_generation_pipeline(
                    "r1", creative_gap_id=1, client=client, max_gemini_calls=4)
        finally:
            for p in ctx:
                p.stop()

        self.assertEqual(call_count, 4,
                         "Validation retries must consume the same cap: "
                         "expected 4 calls, made {}".format(call_count))


if __name__ == '__main__':
    unittest.main(verbosity=2)
