"""
Tests for generate_report.py.

Validates that:
1. The generated report contains NO number that is absent from the stored text.
2. Unscored opportunities are completely excluded from the report.
3. Scored opportunities appear in rank order.
4. Overall score is recomputed from stored component scores.
5. AI reasoning is explicitly labeled as 'AI reasoning (not evidence)'.
6. Weak evidence count is shown per opportunity with 'Evidence fit needs human review.'
7. Competitive landscape section lists competitors, competitive gaps, evidence count, and reused_evidence_count.
8. Claims to verify section lists claims from content pieces.
9. Source quality section lists domains and flags weak ones (scribd, linkedin, quora, reddit, unnamed surveys).
"""

import os
import re
import uuid
from typing import Set

import pytest

from database import (
    CompetitiveIntelligence,
    ConsumerInsight,
    ContentPiece,
    CreativeGap,
    Evidence,
    OpportunityScore,
    ResearchRun,
    Source,
)
from generate_report import extract_domain, generate_markdown_report, get_db_session
from opportunity_engine import compute_overall_score


def extract_numbers_from_text(text: str) -> Set[str]:
    """Extract all integer and floating-point number strings from text."""
    if not text:
        return set()
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    result = set()
    for m in matches:
        result.add(m)
        # Also include standard 2-decimal representation if float
        try:
            val = float(m)
            result.add(f"{val:.2f}")
            if val.is_integer():
                result.add(str(int(val)))
        except ValueError:
            pass
    return result


def seed_test_research_run(db_session):
    """
    Seeds a comprehensive test research run in the test DB with:
    - Brief with numbers (e.g. 2030)
    - Sources & Evidence with real-world statistics (CAGR, percentages, dollar values)
    - Weak sources (LinkedIn, Scribd with unnamed survey) and reputable sources
    - Competitive intelligence with competitors and competitive gaps
    - Consumer insights citing evidence
    - Scored creative gaps (Rank 1 and Rank 2)
    - An UNSCORED creative gap with distinct numbers (must NOT appear in report)
    - Content piece with claims_to_verify
    """
    run_id = str(uuid.uuid4())
    run = ResearchRun(
        id=run_id,
        brand="Lay's",
        market="India",
        audience="Indian youth and Gen Z snackers",
        objective="Analyze potato chips market opportunities toward 2030 targets",
    )
    db_session.add(run)
    db_session.flush()

    # Competitive Intelligence
    comp_intel = CompetitiveIntelligence(
        research_run_id=run_id,
        competitors=[
            {
                "name": "Bingo!",
                "positioning": "Targets Indian youth with bold desi flavors",
                "strengths": ["Bold flavors", "High decibel marketing"],
                "weaknesses": ["Smaller market share"],
            },
            {
                "name": "Balaji Wafers",
                "positioning": "Value pricing and strong regional distribution",
                "strengths": ["Affordable price points"],
                "weaknesses": ["Lower national advertising spend"],
            },
        ],
        market_structure={"leader": "Lay's"},
        competitive_gaps=[
            {"id": 1, "gap": "Expansion into healthier baked potato alternatives"},
            {"id": 2, "gap": "Under-penetration in extreme spicy snack formats"},
        ],
    )
    db_session.add(comp_intel)
    db_session.flush()

    # Source 1: Reputable research firm
    src1 = Source(
        url="https://www.imarcgroup.com/india-potato-chips-market",
        title="India Potato Chips Market Size, Share & Forecast 2034",
        publisher="IMARC Group",
        content_hash=f"{run_id}-hash-1",
    )
    db_session.add(src1)
    db_session.flush()

    ev1 = Evidence(
        research_run_id=run_id,
        source_id=src1.id,
        claim="The Indian potato chips market is valued at USD 1.4 billion in 2025 and is projected to expand to USD 2.0 billion by 2034 at a CAGR of 4.13%.",
        excerpt="The India potato chips market size valued at USD 1.4 Billion in 2025...",
        topic="market size",
        confidence=0.95,
    )
    db_session.add(ev1)

    # Source 2: Weak source (LinkedIn)
    src2 = Source(
        url="https://www.linkedin.com/posts/snack-market-share-44285781",
        title="Indian Potato Chips Market Overview",
        publisher="LinkedIn Member Post",
        content_hash=f"{run_id}-hash-2",
    )
    db_session.add(src2)
    db_session.flush()

    ev2 = Evidence(
        research_run_id=run_id,
        source_id=src2.id,
        claim="Lay's dominates the organized potato chips market with a 30% market share while main competitor Bingo holds approximately 10%.",
        excerpt="Lay's leading at 30% market share and Bingo holding around 10%...",
        topic="market share",
        confidence=0.90,
    )
    db_session.add(ev2)

    # Source 3: Weak source (Scribd with unnamed survey)
    src3 = Source(
        url="https://www.scribd.com/document/1048437737/market-survey-lays",
        title="Market Survey Lays | PDF",
        publisher="Unnamed survey uploaded to Scribd",
        content_hash=f"{run_id}-hash-3",
    )
    db_session.add(src3)
    db_session.flush()

    ev3 = Evidence(
        research_run_id=run_id,
        source_id=src3.id,
        claim="An unnamed survey of 500 consumers indicated that 56% prefer spicy masala flavors over classic salted options.",
        excerpt="Survey of 500 consumers indicated 56% preference...",
        topic="flavor preferences",
        confidence=0.85,
    )
    db_session.add(ev3)
    db_session.flush()

    # Consumer Insights
    ci1 = ConsumerInsight(
        research_run_id=run_id,
        insight_type="segment",
        insight_text="Gen Z snackers show strong preference for intense regional flavors with 56% affinity.",
        supporting_evidence_ids=[ev3.id],
    )
    ci2 = ConsumerInsight(
        research_run_id=run_id,
        insight_type="motivation",
        insight_text="Young professionals seek healthier snacking options while the overall market expands toward USD 2.0 billion.",
        supporting_evidence_ids=[ev1.id, ev2.id],
    )
    db_session.add_all([ci1, ci2])
    db_session.flush()

    # Creative Gaps: Gap 1 (Scored, Rank 1)
    gap1 = CreativeGap(
        research_run_id=run_id,
        gap_title="Baked Masala Potato Crisps",
        description="Launch baked potato chips retaining the iconic Magic Masala taste with 50% less oil.",
        supporting_consumer_insight_ids=[ci1.id],
        supporting_competitor_names=["Bingo!", "Too Yumm!"],
        confidence="high",
    )
    # Creative Gaps: Gap 2 (Scored, Rank 2)
    gap2 = CreativeGap(
        research_run_id=run_id,
        gap_title="Pocket-Friendly On-The-Go Snacking Packs",
        description="Introduce compact 15g single-serve packs tailored for daily student commuters.",
        supporting_consumer_insight_ids=[ci2.id],
        supporting_competitor_names=["Balaji Wafers"],
        confidence="medium",
    )
    # Creative Gaps: Gap 3 (UNSCORED — MUST BE EXCLUDED FROM REPORT)
    gap3 = CreativeGap(
        research_run_id=run_id,
        gap_title="Luxury Truffle Reserve Chips",
        description="Ultra premium chips priced at 999 rupees with 99.9% organic black truffle seasoning.",
        supporting_consumer_insight_ids=[ci1.id],
        supporting_competitor_names=["Pringles"],
        confidence="low",
    )
    db_session.add_all([gap1, gap2, gap3])
    db_session.flush()

    # Opportunity Scores: ONLY for Gap 1 and Gap 2 (Gap 3 has NO score)
    # Gap 1: scores 5, 4, 4, 3 -> overall = 5*0.30 + 4*0.25 + 4*0.25 + 3*0.20 = 1.50 + 1.00 + 1.00 + 0.60 = 4.10
    opp1 = OpportunityScore(
        research_run_id=run_id,
        creative_gap_id=gap1.id,
        rank=1,
        overall_score=4.10,
        audience_demand_score=5,
        audience_demand_reasoning="Urgent consumer demand for healthy guilt-free savory snacks",
        competitive_whitespace_score=4,
        competitive_whitespace_reasoning="Direct competitor Too Yumm has established the category but lacks masala authority",
        brand_fit_score=4,
        brand_fit_reasoning="Lay's flavor equity makes baked masala a natural brand extension",
        feasibility_score=3,
        feasibility_reasoning="Requires dedicated baking lines rather than frying kettles",
    )
    # Gap 2: scores 4, 3, 4, 4 -> overall = 4*0.30 + 3*0.25 + 4*0.25 + 4*0.20 = 1.20 + 0.75 + 1.00 + 0.80 = 3.75
    opp2 = OpportunityScore(
        research_run_id=run_id,
        creative_gap_id=gap2.id,
        rank=2,
        overall_score=3.75,
        audience_demand_score=4,
        audience_demand_reasoning="High frequency daily snacking demand among college students",
        competitive_whitespace_score=3,
        competitive_whitespace_reasoning="Balaji Wafers strongly dominates value price tiers in western markets",
        brand_fit_score=4,
        brand_fit_reasoning="Capitalizes on extensive kirana retail distribution",
        feasibility_score=4,
        feasibility_reasoning="Standard vertical form-fill-seal packaging machinery handles small pouch sizes",
    )
    db_session.add_all([opp1, opp2])
    db_session.flush()

    # Content Piece with claims_to_verify
    cp = ContentPiece(
        research_run_id=run_id,
        creative_gap_id=gap1.id,
        content_type="reel_script",
        platform="instagram",
        title="Masala Baked Revolution",
        hook="All the crunch, none of the guilt",
        body=[{"visual": "Close-up crisp", "voiceover": "Baked to perfection"}],
        caption="Try the new baked crunch!",
        hashtags=["baked", "lays"],
        call_to_action="Grab your pack today",
        supporting_consumer_insight_ids=[ci1.id],
        supporting_competitor_names=["Bingo!"],
        claims_to_verify=[
            "[VERIFIED CLAIM NEEDED: Contains 50% less fat than standard Lay's Classic]",
            "[VERIFIED CLAIM NEEDED: 100% locally sourced Indian potatoes]",
        ],
        status="draft",
    )
    db_session.add(cp)
    db_session.commit()

    return {
        "run_id": run_id,
        "run": run,
        "comp_intel": comp_intel,
        "sources": [src1, src2, src3],
        "evidence": [ev1, ev2, ev3],
        "insights": [ci1, ci2],
        "gaps": [gap1, gap2, gap3],
        "opps": [opp1, opp2],
        "content_pieces": [cp],
    }


def test_report_contains_no_number_absent_from_stored_text(db_session):
    """
    CRITICAL TEST: Asserts that every number in the generated markdown report
    is present in the stored text / database records for that research run.
    """
    data = seed_test_research_run(db_session)
    run_id = data["run_id"]

    # Generate the report
    report = generate_markdown_report(run_id, session=db_session)
    assert report, "Generated report must not be empty"

    # 1. Extract all numbers from the report
    report_numbers = extract_numbers_from_text(report)
    assert report_numbers, "Report should contain numbers (scores, ranks, claim statistics)"

    # 2. Collect all stored text from the database records for this run
    stored_texts = []

    # Brief
    run = data["run"]
    stored_texts.extend([run.brand, run.market, run.audience, run.objective])

    # Competitive Intelligence
    comp_intel = data["comp_intel"]
    for comp in comp_intel.competitors:
        stored_texts.extend([comp.get("name", ""), comp.get("positioning", "")])
        stored_texts.extend(comp.get("strengths", []))
        stored_texts.extend(comp.get("weaknesses", []))
    for cg in comp_intel.competitive_gaps:
        stored_texts.extend([str(cg.get("id", "")), cg.get("gap", "")])

    # Evidence count metrics (explicitly stored/computed for this run)
    total_ev = len(data["evidence"])
    stored_texts.extend([str(total_ev), "0"])  # total evidence, reused evidence

    # Weak evidence counts per opportunity
    # Gap 1 cites ev3 (from scribd: weak -> count = 1)
    # Gap 2 cites ev1 (imarc: ok) and ev2 (linkedin: weak -> count = 1)
    stored_texts.extend(["1", "0"])

    # Scored Gaps and Opportunity Scores
    for opp, gap in zip(data["opps"], [data["gaps"][0], data["gaps"][1]]):
        stored_texts.extend([
            gap.gap_title,
            gap.description,
            gap.confidence,
            " ".join(gap.supporting_competitor_names or []),
            str(gap.id),
            str(opp.rank),
            str(opp.audience_demand_score),
            opp.audience_demand_reasoning,
            str(opp.competitive_whitespace_score),
            opp.competitive_whitespace_reasoning,
            str(opp.brand_fit_score),
            opp.brand_fit_reasoning,
            str(opp.feasibility_score),
            opp.feasibility_reasoning,
            str(opp.overall_score),
            f"{opp.overall_score:.2f}",
        ])
        # Recomputed overall score
        recalc = compute_overall_score(
            opp.audience_demand_score,
            opp.competitive_whitespace_score,
            opp.brand_fit_score,
            opp.feasibility_score,
        )
        stored_texts.extend([str(recalc), f"{recalc:.2f}"])

    # Consumer Insights
    for ci in data["insights"]:
        stored_texts.extend([str(ci.id), ci.insight_type, ci.insight_text])

    # Evidence & Sources
    for ev in data["evidence"]:
        stored_texts.extend([str(ev.id), ev.claim or "", ev.excerpt or "", ev.topic or ""])
        if ev.confidence is not None:
            stored_texts.append(str(ev.confidence))
        if ev.source:
            stored_texts.extend([
                ev.source.url or "",
                ev.source.title or "",
                ev.source.publisher or "",
            ])

    # Content Pieces
    for cp in data["content_pieces"]:
        stored_texts.extend([cp.title, cp.hook, cp.caption])
        for ctv in (cp.claims_to_verify or []):
            stored_texts.append(ctv)

    # 3. Extract all numbers present anywhere in the stored text
    stored_numbers = set()
    for txt in stored_texts:
        stored_numbers.update(extract_numbers_from_text(txt))

    # 4. Check for absent numbers
    absent_numbers = [num for num in report_numbers if num not in stored_numbers]
    assert not absent_numbers, (
        f"Found numbers in generated report that are absent from the stored text: {absent_numbers}\n\n"
        f"Report content:\n{report}"
    )


def test_unscored_opportunities_are_completely_excluded(db_session):
    """
    Asserts that creative gaps without a stored opportunity score
    do not appear anywhere in the generated report.
    """
    data = seed_test_research_run(db_session)
    run_id = data["run_id"]

    report = generate_markdown_report(run_id, session=db_session)

    # Gap 3 was unscored
    unscored_gap = data["gaps"][2]
    assert unscored_gap.gap_title not in report, (
        f"Unscored gap title '{unscored_gap.gap_title}' must not appear in the report"
    )
    assert "999" not in report, "Number '999' from unscored gap description must not appear in the report"
    assert "99.9" not in report, "Number '99.9' from unscored gap description must not appear in the report"
    assert "Luxury Truffle" not in report


def test_opportunities_appear_in_rank_order(db_session):
    """Asserts that scored opportunities are ordered strictly by rank (1 then 2)."""
    data = seed_test_research_run(db_session)
    run_id = data["run_id"]

    report = generate_markdown_report(run_id, session=db_session)

    rank1_pos = report.find("Opportunity Rank 1")
    rank2_pos = report.find("Opportunity Rank 2")

    assert rank1_pos != -1, "Opportunity Rank 1 section missing"
    assert rank2_pos != -1, "Opportunity Rank 2 section missing"
    assert rank1_pos < rank2_pos, "Opportunity Rank 1 must appear before Opportunity Rank 2"


def test_overall_score_is_recomputed(db_session):
    """Asserts that overall_score is recomputed from component scores."""
    data = seed_test_research_run(db_session)
    run_id = data["run_id"]

    # Deliberately modify stored overall_score in DB to an outdated value
    opp1 = data["opps"][0]
    opp1.overall_score = 9.99
    db_session.flush()

    report = generate_markdown_report(run_id, session=db_session)

    # Correct recomputed score is 5*0.30 + 4*0.25 + 4*0.25 + 3*0.20 = 4.10
    assert "4.10" in report, "Recomputed overall score 4.10 must be in the scores table"
    assert "9.99" not in report, "Old/tampered overall score 9.99 must not appear"


def test_ai_reasoning_label_in_scores_table(db_session):
    """Asserts that stored scoring reasoning is labeled as 'AI reasoning (not evidence)'."""
    data = seed_test_research_run(db_session)
    run_id = data["run_id"]

    report = generate_markdown_report(run_id, session=db_session)

    assert "AI reasoning (not evidence)" in report, (
        "Scores table must label reasoning column as 'AI reasoning (not evidence)'"
    )


def test_per_opportunity_weak_evidence_count_and_human_review(db_session):
    """
    Asserts that each opportunity section shows the count of cited evidence items from weak sources
    and includes the required notice 'Evidence fit needs human review.'
    """
    data = seed_test_research_run(db_session)
    run_id = data["run_id"]

    report = generate_markdown_report(run_id, session=db_session)

    assert "Cited evidence items from weak sources" in report
    assert "Evidence fit needs human review." in report


def test_competitive_landscape_section(db_session):
    """
    Asserts that the 'Competitive landscape' section lists competitors and competitive gaps
    from stored data, along with evidence count and reused_evidence_count.
    """
    data = seed_test_research_run(db_session)
    run_id = data["run_id"]

    report = generate_markdown_report(run_id, session=db_session)

    assert "## Competitive Landscape" in report
    assert "**Total Evidence Count**: 3" in report
    assert "**Reused Evidence Count**: 0" in report
    assert "Bingo!" in report
    assert "Balaji Wafers" in report
    assert "Expansion into healthier baked potato alternatives" in report


def test_claims_to_verify_section(db_session):
    """Asserts that claims_to_verify from content pieces are listed in 'Claims to verify' section."""
    data = seed_test_research_run(db_session)
    run_id = data["run_id"]

    report = generate_markdown_report(run_id, session=db_session)

    assert "## Claims to Verify" in report
    assert "[VERIFIED CLAIM NEEDED: Contains 50% less fat than standard Lay's Classic]" in report
    assert "[VERIFIED CLAIM NEEDED: 100% locally sourced Indian potatoes]" in report
    assert "Masala Baked Revolution" in report


def test_source_quality_flags_weak_sources(db_session):
    """
    Asserts that the 'Source quality' section lists domains and flags weak ones
    (linkedin, scribd, unnamed surveys).
    """
    data = seed_test_research_run(db_session)
    run_id = data["run_id"]

    report = generate_markdown_report(run_id, session=db_session)

    assert "## Source Quality" in report
    # Reputable domain: imarcgroup.com
    assert "`imarcgroup.com`: OK" in report
    # Weak domain: linkedin.com
    assert "`linkedin.com`: WEAK" in report
    assert "linkedin" in report
    # Weak domain: scribd.com with unnamed survey
    assert "`scribd.com`: WEAK" in report
    assert "scribd" in report
    assert "unnamed survey" in report
