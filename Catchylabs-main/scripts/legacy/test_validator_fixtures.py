import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Test validator fixtures with exact strings - no Gemini calls.
"""

from content_generator import _brand_claim_errors

def test_validator_fixtures():
    """Test claim validator with exact fixture strings."""

    brand_name = "Puma"
    cited_evidence = "Puma offers sustainable products at affordable prices."  # Example evidence

    fixtures = [
        # Must be flagged
        ("Check out Puma's latest sustainable lineup today.", True),
        ("We're bridging the gap so you can flex sustainable activewear without the premium price barrier.", True),
        ("High Quality. Eco-conscious. Real Prices.", True),
        ("wearing new Puma eco-friendly running shoes ... low impact on the planet", True),
        ("Upgrade your run, protect your wallet.", True),
        # Must pass
        ("Think eco-friendly running shoes have to break your bank account?", False),
        ("Gen Z consumers are highly price-sensitive.", False),
    ]

    print("=== Validator Fixture Results ===\n")
    print(f"{'Text':<80} | {'Expected':<10} | {'Actual':<10} | {'Status':<10}")
    print("-" * 120)

    for text, expected_flagged in fixtures:
        errors, claims = _brand_claim_errors(text, 1, brand_name, cited_evidence)
        actual_flagged = len(claims) > 0 or len(errors) > 0

        status = "PASS" if actual_flagged == expected_flagged else "FAIL"
        print(f"{text:<80} | {str(expected_flagged):<10} | {str(actual_flagged):<10} | {status:<10}")

if __name__ == "__main__":
    test_validator_fixtures()
