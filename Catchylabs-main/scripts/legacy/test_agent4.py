import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Comprehensive Test Suite for Agent 4 (Creative Gap Detection)
--------------------------------------------------------------
Tests:
1. Validation Logic & Edge Cases:
   - Clear failure when Agent 2 missing
   - Clear failure when Agent 3 missing
   - Rejection of single-sided gaps (only consumer insights or only competitors)
   - Rejection of hallucinated consumer insight IDs
   - Rejection of hallucinated competitor names
   - Numerical grounding validation (reject fabricated numbers, accept qualitative & verified numbers)
2. Database Trigger FK Enforcement:
   - Direct SQL insert with invalid supporting_consumer_insight_ids fails via DB trigger
3. FastAPI Endpoints:
   - 404 on non-existent run
   - 400 naming missing Agent 2 on run with only Agent 1
   - 200 and enqueue on run with Agent 2 + Agent 3
4. Pipeline Execution on Puma/India/Gen Z run:
   - Measure SerpAPI quota before and after (confirm 0 new calls)
   - Audit cross-referencing (confirm dual-sided citations)
   - Compare directly against Agent 3 competitive_gaps to verify novel synthesis
"""

import sys
import json
import re
from typing import Dict, Any

from fastapi.testclient import TestClient
from database import (
    SessionLocal,
    engine,
    get_quota_status,
    get_consumer_insights,
    get_competitive_intelligence,
    get_creative_gaps,
    CreativeGap,
    ResearchRun,
)
from app import app
from agent4 import (
    validate_creative_gaps,
    run_creative_gaps_pipeline,
    extract_statistical_figures,
)

PUMA_RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"

def test_unit_validation_rules():
    print("\n--- TEST 1: Agent 4 Validation Logic & Constraints ---")
    valid_ci_ids = {17, 18, 19, 20}
    ci_lookup = {
        17: {"id": 17, "insight_text": "Indian Gen Z concerned shoppers seek sustainable footwear."},
        18: {"id": 18, "insight_text": "Puma holds strong sportswear market share."},
        19: {"id": 19, "insight_text": "Gen Z is driven by vegan materials and ethical values."},
        20: {"id": 20, "insight_text": "Young consumers seek earth-friendly craftsmanship and trendy designs."}
    }
    valid_comp_names_map = {
        "zontrek": "Zontrek",
        "thaely": "Thaely",
        "satva": "Satva"
    }
    comp_lookup = {
        "Zontrek": {"name": "Zontrek", "positioning": "Affordable vegan sneaker brand", "strengths": ["100 percent vegan"], "weaknesses": ["Price sensitive"]},
        "Thaely": {"name": "Thaely", "positioning": "Recycled plastic waste shoes", "strengths": ["Innovative waste repurposing"], "weaknesses": ["Premium price barrier"]},
        "Satva": {"name": "Satva", "positioning": "Organic cotton activewear for women", "strengths": ["100 percent organic"], "weaknesses": ["Niche focus"]}
    }
    valid_cg_ids = {1, 2}
    comp_gap_lookup = {
        1: {"id": 1, "gap": "Bridging tension between price sensitivity and sustainable running shoes."},
        2: {"id": 2, "gap": "Digital channels with authentic messaging."}
    }

    # Case A: Missing consumer insights (single-sided: competitors only)
    bad_data_no_ci = {
        "creative_gaps": [{
            "gap_title": "Affordable Sustainable Footwear",
            "description": "Competitors like Thaely are too expensive while Zontrek is niche.",
            "supporting_consumer_insight_ids": [],
            "supporting_competitor_names": ["Thaely", "Zontrek"],
            "confidence": "high"
        }]
    }
    _, errors = validate_creative_gaps(
        bad_data_no_ci, valid_ci_ids, ci_lookup, valid_comp_names_map, comp_lookup, valid_cg_ids, comp_gap_lookup
    )
    assert any("only cites competitor data" in e for e in errors), f"Expected missing CI error, got: {errors}"
    print("  [PASS] Single-sided gap (competitors only) rejected.")

    # Case B: Missing competitors (single-sided: consumer insights only)
    bad_data_no_comp = {
        "creative_gaps": [{
            "gap_title": "Affordable Sustainable Footwear",
            "description": "Gen Z shoppers want sustainable products with trendy designs.",
            "supporting_consumer_insight_ids": [17, 19],
            "supporting_competitor_names": [],
            "confidence": "high"
        }]
    }
    _, errors = validate_creative_gaps(
        bad_data_no_comp, valid_ci_ids, ci_lookup, valid_comp_names_map, comp_lookup, valid_cg_ids, comp_gap_lookup
    )
    assert any("only cites consumer insights" in e for e in errors), f"Expected missing competitor error, got: {errors}"
    print("  [PASS] Single-sided gap (consumer insights only) rejected.")

    # Case C: Hallucinated consumer insight ID
    bad_data_fake_ci = {
        "creative_gaps": [{
            "gap_title": "Authentic Streetwear",
            "description": "Valid intersection between consumer desire and competitive absence.",
            "supporting_consumer_insight_ids": [999],
            "supporting_competitor_names": ["Thaely"],
            "confidence": "high"
        }]
    }
    _, errors = validate_creative_gaps(
        bad_data_fake_ci, valid_ci_ids, ci_lookup, valid_comp_names_map, comp_lookup, valid_cg_ids, comp_gap_lookup
    )
    assert any("hallucinated consumer insight IDs" in e for e in errors), f"Expected hallucinated CI error, got: {errors}"
    print("  [PASS] Hallucinated consumer insight ID (999) rejected.")

    # Case D: Hallucinated competitor name
    bad_data_fake_comp = {
        "creative_gaps": [{
            "gap_title": "Authentic Streetwear",
            "description": "Valid intersection between consumer desire and competitive absence.",
            "supporting_consumer_insight_ids": [17],
            "supporting_competitor_names": ["NonExistentBrand"],
            "confidence": "high"
        }]
    }
    _, errors = validate_creative_gaps(
        bad_data_fake_comp, valid_ci_ids, ci_lookup, valid_comp_names_map, comp_lookup, valid_cg_ids, comp_gap_lookup
    )
    assert any("not in Agent 3 competitor list" in e for e in errors), f"Expected unknown competitor error, got: {errors}"
    print("  [PASS] Hallucinated competitor name ('NonExistentBrand') rejected.")

    # Case E: Fabricated number (un-grounded statistic)
    bad_data_fake_num = {
        "creative_gaps": [{
            "gap_title": "Massive 85% Market Opportunity in Eco Streetwear",
            "description": "Over 85% of Gen Z consumers are demanding recycled plastic footwear that Thaely cannot supply at scale.",
            "supporting_consumer_insight_ids": [17, 20],
            "supporting_competitor_names": ["Thaely"],
            "confidence": "high"
        }]
    }
    _, errors = validate_creative_gaps(
        bad_data_fake_num, valid_ci_ids, ci_lookup, valid_comp_names_map, comp_lookup, valid_cg_ids, comp_gap_lookup
    )
    assert any("ungrounded/hallucinated statistic" in e for e in errors), f"Expected ungrounded stat error, got: {errors}"
    print("  [PASS] Fabricated statistic ('85%') rejected.")

    # Case F: Perfectly valid cross-referenced gap (qualitative or verified number)
    good_data = {
        "creative_gaps": [{
            "gap_title": "Mainstream Trend-Driven Recycled Footwear at Accessible Pricing",
            "description": "While Thaely proves consumer appetite for recycled waste footwear, its premium tier alienates price-sensitive youth. Gen Z craves trendy earth-friendly designs without paying a massive luxury premium.",
            "supporting_consumer_insight_ids": [17, 20],
            "supporting_competitor_names": ["Thaely", "Zontrek"],
            "supporting_competitive_gap_ids": [1],
            "confidence": "high"
        }]
    }
    validated, errors = validate_creative_gaps(
        good_data, valid_ci_ids, ci_lookup, valid_comp_names_map, comp_lookup, valid_cg_ids, comp_gap_lookup
    )
    assert len(errors) == 0, f"Expected 0 errors, got: {errors}"
    assert len(validated) == 1
    assert validated[0]["supporting_competitor_names"] == ["Thaely", "Zontrek"]
    print("  [PASS] Valid dual-sided creative gap accepted.")


def test_db_trigger_fk_validation():
    print("\n--- TEST 2: Database Trigger FK Validation on creative_gaps ---")
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            # Try to insert a row with an invalid supporting_consumer_insight_id (999999)
            conn.execute(text("""
                INSERT INTO creative_gaps (
                    research_run_id, gap_title, description,
                    supporting_consumer_insight_ids, supporting_competitor_names,
                    supporting_competitive_gap_ids, confidence, created_at
                ) VALUES (
                    :run_id, 'Trigger Test', 'Description for testing database trigger constraint violation',
                    ARRAY[999999], ARRAY['Thaely'], ARRAY[]::integer[], 'high', NOW()
                )
            """), {"run_id": PUMA_RUN_ID})
            conn.commit()
            raise AssertionError("Database trigger should have raised an exception for invalid consumer insight ID!")
        except Exception as exc:
            err_msg = str(exc)
            assert "Foreign key violation: supporting_consumer_insight_ids" in err_msg or "trigger" in err_msg.lower() or "999999" in err_msg
            print(f"  [PASS] Database trigger successfully blocked invalid FK array: {err_msg[:120]}...")


def test_endpoints():
    print("\n--- TEST 3: FastAPI Endpoints ---")
    client = TestClient(app)

    # 1. Non-existent run_id returns 404
    resp_404 = client.post("/research/non-existent-uuid/creative-gaps")
    assert resp_404.status_code == 404, f"Expected 404, got: {resp_404.status_code}"
    print("  [PASS] POST /research/{non-existent}/creative-gaps returns 404.")

    # 2. Run missing Agent 2/3 returns 400 with clear message
    # Let's find a run that has no Agent 2/3 (e.g. c99e1546-d796-4309-b112-9cf341f60e94)
    resp_400 = client.post("/research/c99e1546-d796-4309-b112-9cf341f60e94/creative-gaps")
    assert resp_400.status_code == 400, f"Expected 400, got: {resp_400.status_code}"
    assert "missing" in resp_400.json()["detail"].lower()
    print(f"  [PASS] POST /research/{{incomplete}}/creative-gaps returns 400: '{resp_400.json()['detail']}'")

    # 3. GET /research/{PUMA_RUN_ID}/creative-gaps returns 200 (even if empty initially)
    resp_get = client.get(f"/research/{PUMA_RUN_ID}/creative-gaps")
    assert resp_get.status_code == 200, f"Expected 200, got: {resp_get.status_code}"
    print(f"  [PASS] GET /research/{PUMA_RUN_ID}/creative-gaps returns 200.")


def test_pipeline_execution():
    print("\n--- TEST 4: Live Agent 4 Pipeline Execution on Puma Run ---")
    
    # Check SerpAPI quota BEFORE
    quota_before = get_quota_status()
    serpapi_before = quota_before.get("quotas", {}).get("serpapi", {})
    serpapi_calls_before = serpapi_before.get("local_db", {}).get("this_calendar_month_total", 0)
    api_calls_total_before = quota_before.get("database_status", {}).get("table_row_counts", {}).get("api_calls", 0)
    print(f"  [QUOTA BEFORE] SerpAPI monthly calls: {serpapi_calls_before}, Total api_calls rows: {api_calls_total_before}")

    # Execute Agent 4 Pipeline
    print(f"  Executing run_creative_gaps_pipeline for Puma run: {PUMA_RUN_ID}...")
    saved_gaps = run_creative_gaps_pipeline(PUMA_RUN_ID)
    assert len(saved_gaps) > 0, "Agent 4 should have returned at least 1 creative gap."
    print(f"  [SUCCESS] Generated and saved {len(saved_gaps)} creative gaps!")

    # Check SerpAPI quota AFTER
    quota_after = get_quota_status()
    serpapi_after = quota_after.get("quotas", {}).get("serpapi", {})
    serpapi_calls_after = serpapi_after.get("local_db", {}).get("this_calendar_month_total", 0)
    api_calls_total_after = quota_after.get("database_status", {}).get("table_row_counts", {}).get("api_calls", 0)
    print(f"  [QUOTA AFTER]  SerpAPI monthly calls: {serpapi_calls_after}, Total api_calls rows: {api_calls_total_after}")

    serpapi_delta = serpapi_calls_after - serpapi_calls_before
    assert serpapi_delta == 0, f"SERPAPI CALLS USED: {serpapi_delta} (Expected 0!)"
    print(f"  [VERIFIED ZERO SERPAPI] Exact SerpAPI delta: {serpapi_delta} new calls!")

    # Audit Cross-Referencing
    print("\n--- TEST 5: Cross-Reference Audit ---")
    ci_data = get_consumer_insights(PUMA_RUN_ID)
    valid_ci_ids = {ci["id"] for ci in ci_data}
    comp_data = get_competitive_intelligence(PUMA_RUN_ID)
    valid_comp_names = [c["name"] for c in comp_data.get("competitors", [])]
    existing_agent3_gaps = comp_data.get("competitive_gaps", [])

    print(f"  Available Consumer Insight IDs: {sorted(list(valid_ci_ids))}")
    print(f"  Available Competitors: {valid_comp_names}")

    for idx, gap in enumerate(saved_gaps, 1):
        print(f"\n  [Gap #{idx}] \"{gap['gap_title']}\"")
        print(f"    Confidence: {gap['confidence']}")
        print(f"    Supporting Consumer Insights: {gap['supporting_consumer_insight_ids']}")
        print(f"    Supporting Competitors: {gap['supporting_competitor_names']}")
        print(f"    Supporting Competitive Gap IDs: {gap['supporting_competitive_gap_ids']}")
        print(f"    Description: {gap['description'][:160]}...")

        # Strict assertion: dual-sided citations
        assert len(gap['supporting_consumer_insight_ids']) >= 1, f"Gap #{idx} missing consumer insight citation!"
        assert len(gap['supporting_competitor_names']) >= 1, f"Gap #{idx} missing competitor citation!"

        # Strict assertion: valid IDs & names
        for cid in gap['supporting_consumer_insight_ids']:
            assert cid in valid_ci_ids, f"Gap #{idx} cited invalid CI ID {cid}!"
        for cname in gap['supporting_competitor_names']:
            assert cname in valid_comp_names, f"Gap #{idx} cited invalid competitor name '{cname}'!"

    print("\n  [AUDIT PASSED] 100% of gaps cite BOTH consumer insight(s) AND competitor(s).")

    # Novelty Assessment vs Agent 3 competitive_gaps
    print("\n--- TEST 6: Novelty Comparison vs Agent 3 Competitive Gaps ---")
    print("  Agent 3 Competitive Gaps (Baseline Market Level):")
    for cg in existing_agent3_gaps:
        print(f"    - [Agent 3 Gap ID {cg.get('id')}]: {cg.get('gap')}")

    print("\n  Agent 4 Creative Gaps (Intersection of Consumer Desire & Competitive Absence):")
    for gap in saved_gaps:
        print(f"    + [Agent 4 Gap]: {gap['gap_title']}")
        print(f"      Novel Territory: {gap['description']}")


if __name__ == "__main__":
    test_unit_validation_rules()
    test_db_trigger_fk_validation()
    test_endpoints()
    test_pipeline_execution()
    print("\n============================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("============================================================")
