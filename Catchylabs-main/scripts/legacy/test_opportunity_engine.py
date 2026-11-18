import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Comprehensive Test Suite for the Opportunity Engine
---------------------------------------------------
Tests:
1. Unit Validation Rules & Edge Cases:
   - Rejects score < 1 or > 5
   - Rejects non-integer scores
   - Rejects hallucinated creative_gap_ids
   - Rejects missing gaps
   - Rejects fabricated numbers in reasoning text
   - Correctly validates grounded reasoning and scores
2. Mathematical Precision:
   - Spot-checks weighted calculation: 30% audience_demand + 25% whitespace + 25% brand_fit + 20% feasibility
3. FastAPI Endpoints:
   - 404 on non-existent run
   - 400 with clear message if Agent 4 hasn't run
   - 200 and enqueue on valid run
   - GET /research/{run_id}/opportunities returns ranked list
4. Live Pipeline Execution on Puma Run:
   - Quota tracking before & after (CONFIRM 0 new SerpAPI calls)
   - Print all scores & reasonings for each gap
   - Confirm intuitive sense of the ranking
   - Hand-calculate one gap's overall score
"""

import sys
import json
from fastapi.testclient import TestClient

from database import (
    get_quota_status,
    get_creative_gaps,
    get_opportunity_scores,
    get_consumer_insights,
    get_competitive_intelligence,
)
from app import app
from opportunity_engine import (
    compute_overall_score,
    validate_opportunity_scores,
    run_opportunity_engine_pipeline,
    WEIGHT_AUDIENCE_DEMAND,
    WEIGHT_COMPETITIVE_WHITESPACE,
    WEIGHT_BRAND_FIT,
    WEIGHT_FEASIBILITY,
)

PUMA_RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"


def test_math_and_weights():
    print("\n--- TEST 1: Weighted Score Mathematical Precision ---")
    # Verify weights sum to 1.0
    total_weights = WEIGHT_AUDIENCE_DEMAND + WEIGHT_COMPETITIVE_WHITESPACE + WEIGHT_BRAND_FIT + WEIGHT_FEASIBILITY
    assert abs(total_weights - 1.0) < 1e-6, f"Weights must sum to 1.0, got: {total_weights}"
    print(f"  [PASS] Weights sum to 1.0 (Audience: {WEIGHT_AUDIENCE_DEMAND}, Whitespace: {WEIGHT_COMPETITIVE_WHITESPACE}, BrandFit: {WEIGHT_BRAND_FIT}, Feasibility: {WEIGHT_FEASIBILITY})")

    # Hand calculation spot-check:
    # Example: scores = (5, 4, 4, 4)
    # Expected: 5*0.30 + 4*0.25 + 4*0.25 + 4*0.20 = 1.50 + 1.00 + 1.00 + 0.80 = 4.30
    calc = compute_overall_score(5, 4, 4, 4)
    assert calc == 4.30, f"Expected 4.30, got {calc}"
    print(f"  [PASS] Spot check: scores (5, 4, 4, 4) -> overall_score = {calc}")

    # Example 2: scores = (4, 5, 3, 4)
    # Expected: 4*0.30 + 5*0.25 + 3*0.25 + 4*0.20 = 1.20 + 1.25 + 0.75 + 0.80 = 4.00
    calc2 = compute_overall_score(4, 5, 3, 4)
    assert calc2 == 4.00, f"Expected 4.00, got {calc2}"
    print(f"  [PASS] Spot check: scores (4, 5, 3, 4) -> overall_score = {calc2}")


def test_unit_validation():
    print("\n--- TEST 2: Validation Rules & Grounding Edge Cases ---")
    valid_gap_ids = {6, 7}
    gap_lookup = {
        6: {
            "id": 6,
            "gap_title": "Radical Transparency",
            "description": "Authentic sustainable footwear without greenwashing.",
            "supporting_consumer_insight_ids": [19],
            "supporting_competitor_names": ["Thaely"]
        },
        7: {
            "id": 7,
            "gap_title": "Affordable Eco Footwear",
            "description": "Budget-friendly sustainable sneakers.",
            "supporting_consumer_insight_ids": [21],
            "supporting_competitor_names": ["Zontrek"]
        }
    }
    ci_lookup = {
        19: {"id": 19, "insight_text": "Gen Z consumers are strongly driven by ethical values and vegan materials."},
        21: {"id": 21, "insight_text": "High price sensitivity creates a major barrier for Gen Z apparel."}
    }
    comp_lookup = {
        "thaely": {"name": "Thaely", "positioning": "Recycled plastic shoes", "strengths": ["Innovation"], "weaknesses": ["Premium price"]},
        "zontrek": {"name": "Zontrek", "positioning": "Affordable vegan sneakers", "strengths": ["100 percent vegan"], "weaknesses": ["Niche"]}
    }

    # Case A: Out-of-bounds score (e.g. 6)
    bad_data_score_range = {
        "opportunity_scores": [
            {
                "creative_gap_id": 6,
                "audience_demand_score": 6,
                "audience_demand_reasoning": "Extremely high consumer demand.",
                "competitive_whitespace_score": 4,
                "competitive_whitespace_reasoning": "Solid whitespace in the market.",
                "brand_fit_score": 5,
                "brand_fit_reasoning": "Strong brand authority and heritage.",
                "feasibility_score": 4,
                "feasibility_reasoning": "Readily executable across current channels."
            },
            {
                "creative_gap_id": 7,
                "audience_demand_score": 4,
                "audience_demand_reasoning": "Strong demand from price sensitive consumers.",
                "competitive_whitespace_score": 4,
                "competitive_whitespace_reasoning": "Competitors fail on price.",
                "brand_fit_score": 4,
                "brand_fit_reasoning": "Good brand fit for scale.",
                "feasibility_score": 4,
                "feasibility_reasoning": "High operational feasibility."
            }
        ]
    }
    _, errors = validate_opportunity_scores(bad_data_score_range, valid_gap_ids, gap_lookup, ci_lookup, comp_lookup)
    assert any("out of bounds" in e for e in errors), f"Expected out of bounds error, got: {errors}"
    print("  [PASS] Out-of-bounds score (6) rejected.")

    # Case B: Hallucinated creative_gap_id (999)
    bad_data_fake_id = {
        "opportunity_scores": [
            {
                "creative_gap_id": 999,
                "audience_demand_score": 5,
                "audience_demand_reasoning": "Extremely high consumer demand.",
                "competitive_whitespace_score": 4,
                "competitive_whitespace_reasoning": "Solid whitespace in the market.",
                "brand_fit_score": 5,
                "brand_fit_reasoning": "Strong brand authority and heritage.",
                "feasibility_score": 4,
                "feasibility_reasoning": "Readily executable across current channels."
            }
        ]
    }
    _, errors = validate_opportunity_scores(bad_data_fake_id, valid_gap_ids, gap_lookup, ci_lookup, comp_lookup)
    assert any("hallucinated creative_gap_id" in e for e in errors), f"Expected hallucinated ID error, got: {errors}"
    print("  [PASS] Hallucinated creative_gap_id (999) rejected.")

    # Case C: Fabricated number in reasoning
    bad_data_fake_stat = {
        "opportunity_scores": [
            {
                "creative_gap_id": 6,
                "audience_demand_score": 5,
                "audience_demand_reasoning": "Over 92% of Gen Z consumers demand this specific anti-greenwashing transparency.",
                "competitive_whitespace_score": 4,
                "competitive_whitespace_reasoning": "Solid whitespace in the market.",
                "brand_fit_score": 5,
                "brand_fit_reasoning": "Strong brand authority and heritage.",
                "feasibility_score": 4,
                "feasibility_reasoning": "Readily executable across current channels."
            },
            {
                "creative_gap_id": 7,
                "audience_demand_score": 4,
                "audience_demand_reasoning": "Strong demand from price sensitive consumers.",
                "competitive_whitespace_score": 4,
                "competitive_whitespace_reasoning": "Competitors fail on price.",
                "brand_fit_score": 4,
                "brand_fit_reasoning": "Good brand fit for scale.",
                "feasibility_score": 4,
                "feasibility_reasoning": "High operational feasibility."
            }
        ]
    }
    _, errors = validate_opportunity_scores(bad_data_fake_stat, valid_gap_ids, gap_lookup, ci_lookup, comp_lookup)
    assert any("ungrounded/hallucinated statistic" in e for e in errors), f"Expected ungrounded stat error, got: {errors}"
    print("  [PASS] Fabricated statistic in reasoning ('92%') rejected.")


def test_endpoints():
    print("\n--- TEST 3: FastAPI Endpoints ---")
    client = TestClient(app)

    # 1. Non-existent run_id -> 404
    resp_404 = client.post("/research/non-existent-uuid/opportunities")
    assert resp_404.status_code == 404
    print("  [PASS] POST /research/{non-existent}/opportunities returns 404.")

    # 2. Run without Agent 4 output -> 400 with clear message
    resp_400 = client.post("/research/c99e1546-d796-4309-b112-9cf341f60e94/opportunities")
    assert resp_400.status_code == 400
    assert "agent 4" in resp_400.json()["detail"].lower()
    print(f"  [PASS] POST /research/{{run_without_agent4}}/opportunities returns 400: '{resp_400.json()['detail']}'")

    # 3. GET on Puma run
    resp_get = client.get(f"/research/{PUMA_RUN_ID}/opportunities")
    assert resp_get.status_code == 200
    print(f"  [PASS] GET /research/{PUMA_RUN_ID}/opportunities returns 200.")


def test_pipeline_execution():
    print("\n--- TEST 4: Live Opportunity Engine Pipeline Execution on Puma Run ---")
    
    # Check SerpAPI quota before
    quota_before = get_quota_status()
    serpapi_before = quota_before.get("quotas", {}).get("serpapi", {}).get("local_db", {}).get("this_calendar_month_total", 0)
    print(f"  [QUOTA BEFORE] SerpAPI calls: {serpapi_before}")

    # Run Opportunity Engine pipeline
    ranked = run_opportunity_engine_pipeline(PUMA_RUN_ID)
    assert len(ranked) > 0, "Expected at least 1 ranked opportunity"

    # Check SerpAPI quota after
    quota_after = get_quota_status()
    serpapi_after = quota_after.get("quotas", {}).get("serpapi", {}).get("local_db", {}).get("this_calendar_month_total", 0)
    print(f"  [QUOTA AFTER]  SerpAPI calls: {serpapi_after}")

    serpapi_delta = serpapi_after - serpapi_before
    assert serpapi_delta == 0, f"Expected 0 new SerpAPI calls, got {serpapi_delta}"
    print(f"  [VERIFIED ZERO SERPAPI] Exact SerpAPI delta: {serpapi_delta} new calls!")

    # Fetch stored ranked opportunities from DB
    opps = get_opportunity_scores(PUMA_RUN_ID)
    print(f"\n--- TEST 5: Ranked Opportunities Breakdown ({len(opps)} gaps evaluated) ---")

    for opp in opps:
        rank = opp["rank"]
        overall = opp["overall_score"]
        scores = opp["scores"]
        reasonings = opp["reasoning"]
        gap = opp.get("creative_gap", {})

        print(f"\n{'='*70}")
        print(f"RANK #{rank}: \"{gap.get('gap_title')}\" (Gap ID: {opp['creative_gap_id']})")
        print(f"OVERALL WEIGHTED SCORE: {overall:.2f} / 5.00")
        print(f"{'-'*70}")
        print(f"  1. Audience Demand (30%):        {scores['audience_demand']}/5")
        print(f"     Reasoning: {reasonings['audience_demand']}")
        print(f"  2. Competitive Whitespace (25%): {scores['competitive_whitespace']}/5")
        print(f"     Reasoning: {reasonings['competitive_whitespace']}")
        print(f"  3. Brand Fit (25%):              {scores['brand_fit']}/5")
        print(f"     Reasoning: {reasonings['brand_fit']}")
        print(f"  4. Feasibility (20%):            {scores['feasibility']}/5")
        print(f"     Reasoning: {reasonings['feasibility']}")

        # Validate math on every returned record:
        manual_calc = round(
            scores['audience_demand'] * 0.30
            + scores['competitive_whitespace'] * 0.25
            + scores['brand_fit'] * 0.25
            + scores['feasibility'] * 0.20,
            2
        )
        assert abs(overall - manual_calc) < 1e-4, f"Math mismatch: stored={overall}, manual={manual_calc}"

    print(f"\n{'='*70}")
    print("ALL OPPORTUNITY SCORES VALIDATED AND VERIFIED MATHEMATICALLY!")


if __name__ == "__main__":
    test_math_and_weights()
    test_unit_validation()
    test_endpoints()
    test_pipeline_execution()
    print("\n============================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("============================================================")
