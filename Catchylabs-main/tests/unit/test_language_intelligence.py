import unittest

from intelligence.dataset import (
    DATASET_VERSION,
    eligible_for_training,
    find_split_leakage,
    validate_example,
)
from intelligence.knowledge import retrieve_knowledge
from intelligence.language import detect_language, output_language_instruction, resolve_language
from intelligence.research_quality import build_research_context


class LanguageDetectionTests(unittest.TestCase):
    def test_detects_english_hindi_roman_hindi_and_hinglish(self):
        examples = [
            ("Build a campaign strategy for this product.", "english", "latin"),
            ("मुझे इस ब्रांड के लिए एक कहानी चाहिए।", "hindi", "devanagari"),
            ("Mujhe apne naye brand ke liye kahani banana hai.", "hindi", "latin"),
            (
                "Bhai, is product ko viral feel dene wala reel concept bana.",
                "hinglish",
                "latin",
            ),
            ("यह ad funny है, लेकिन message साफ़ नहीं है।", "hinglish", "mixed"),
        ]
        for text, language, script in examples:
            with self.subTest(text=text):
                result = detect_language(text)
                self.assertEqual(result["language"], language)
                self.assertEqual(result["script"], script)
                self.assertLessEqual(result["confidence"], 1)

    def test_explicit_preference_overrides_heuristic(self):
        result = resolve_language("यह एक विचार है", "english")
        self.assertEqual(result["language"], "english")
        self.assertEqual(result["selection_method"], "user_preference")
        with self.assertRaises(ValueError):
            resolve_language("text", "tamil")

    def test_generation_guidance_preserves_identifiers_and_cultural_uncertainty(self):
        instruction = output_language_instruction("hinglish")
        self.assertIn("Roman script", instruction)
        self.assertIn("technical identifiers", instruction)
        self.assertIn("Do not invent cultural", instruction)


class MultilingualKnowledgeTests(unittest.TestCase):
    def test_devanagari_and_romanized_hindi_terms_retrieve_original_guidance(self):
        devanagari = retrieve_knowledge("विज्ञापन और हास्य के लिए रचनात्मक रणनीति", limit=3)
        romanized = retrieve_knowledge("reel ka hook funny aur kahani creative", limit=3)
        self.assertIn("creative_strategy", {item["domain"] for item in devanagari})
        self.assertIn("creative_strategy", {item["domain"] for item in romanized})
        creative = next(item for item in devanagari if item["domain"] == "creative_strategy")
        self.assertEqual(creative["metadata"]["original_language"], "en")
        self.assertEqual(creative["metadata"]["translation_status"], "not_translated")
        self.assertFalse(creative["metadata"]["training_eligible"])

    def test_research_context_preserves_source_text_language_and_query_hints(self):
        context = build_research_context(
            {
                "objective": "यह ग्राहक की समस्या समझें",
                "language_preference": "hinglish",
            },
            [
                {
                    "claim": "Audience discussions describe late deliveries.",
                    "evidence": "मेरा ऑर्डर अभी तक नहीं आया।",
                    "source": {"url": "https://example.com/post"},
                }
            ],
            retrieved_at="2026-10-28T00:00:00+00:00",
            search_queries=["ग्राहक की समस्या"],
        )
        self.assertEqual(context["language"]["language"], "hinglish")
        entry = context["evidence_registry"][0]
        self.assertEqual(entry["source_language"]["language"], "hindi")
        self.assertEqual(entry["translation_status"], "not_asserted")
        self.assertEqual(entry["excerpt"], "मेरा ऑर्डर अभी तक नहीं आया।")
        self.assertEqual(context["search_language_coverage"]["queries"][0]["script"], "devanagari")


class SupervisedDatasetGuardTests(unittest.TestCase):
    def make_example(self, split="train"):
        return {
            "dataset_version": DATASET_VERSION,
            "task": "humor_style",
            "split": split,
            "input": "An original synthetic Hinglish ad brief.",
            "expected_output": {"humor_style": "observational"},
            "language": {"language": "hinglish", "script": "latin"},
            "annotation_language": "english",
            "synthetic": True,
            "provenance": {
                "source_type": "original_synthetic",
                "usage_rights": "internal_original",
            },
            "review": {"status": "unreviewed", "confidence": None},
        }

    def test_unreviewed_synthetic_example_is_not_training_eligible(self):
        example = self.make_example()
        self.assertEqual(validate_example(example), [])
        self.assertFalse(eligible_for_training(example))

    def test_human_review_requires_reviewer_provenance(self):
        example = self.make_example()
        example["review"] = {"status": "human_reviewed", "confidence": 0.9}
        self.assertIn("reviewer_reference_required", validate_example(example))

    def test_rights_cleared_reviewed_training_example_can_be_eligible(self):
        example = self.make_example()
        example["review"] = {
            "status": "human_reviewed",
            "reviewer_reference": "reviewer-pseudonym-01",
            "confidence": 0.95,
            "disagreement": False,
        }
        self.assertTrue(eligible_for_training(example))

    def test_exact_duplicates_must_not_cross_data_splits(self):
        training = self.make_example("train")
        validation = self.make_example("validation")
        validation["input"] = " an ORIGINAL synthetic Hinglish ad brief. "
        self.assertEqual(find_split_leakage([training, validation])[0]["prior_split"], "train")


if __name__ == "__main__":
    unittest.main()
