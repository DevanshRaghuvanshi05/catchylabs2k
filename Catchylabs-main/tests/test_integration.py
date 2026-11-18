"""
Integration tests for validation logic (no database operations).
"""

import pytest
from content_generator import _brand_claim_errors


class TestPriceWordingValidation:
    """Test price/affordability wording validation."""

    def test_price_terms_flagged_in_brand_context(self):
        """Test that price terms are flagged in brand-referencing context."""
        brand_name = "Puma"
        cited_evidence = "Puma offers sustainable products."

        # Brand context - should be flagged (sustainable is not in cited evidence as exact phrase)
        text = "Check out Puma's latest sustainable lineup today."
        errors, claims = _brand_claim_errors(text, 1, brand_name, cited_evidence)
        assert len(claims) > 0 or len(errors) > 0  # Should be flagged

        # Audience context - should pass
        text = "Consumers want affordable prices and low costs."
        errors, claims = _brand_claim_errors(text, 1, brand_name, cited_evidence)
        assert len(claims) == 0 and len(errors) == 0

