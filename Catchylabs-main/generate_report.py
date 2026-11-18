"""
Report Generator for CatchyLab Research Runs.

Given a research_run_id, reads stored data from PostgreSQL:
- Brief (brand, audience, market, objective)
- Competitive landscape (competitors, competitive gaps, evidence count, reused_evidence_count)
- Opportunity scores (in rank order)
- Creative gaps (only scored ones; unscored gaps are excluded)
- Consumer insights (cited by scored gaps)
- Competitors (cited by scored gaps)
- Evidence claims with source titles and URLs (cited by consumer insights)
- Number of cited evidence items from weak sources per opportunity
- Content pieces and their claims_to_verify
- Source quality assessment by domain (flagging scribd, linkedin, quora, reddit, unnamed surveys)

Strict constraint: NO free-written text or fabricated analysis.
Every statistic appears ONLY inside the claim it originated from, with its source.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import dotenv_values
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, joinedload, sessionmaker

from database import (
    CompetitiveIntelligence,
    ConsumerInsight,
    ContentPiece,
    CreativeGap,
    Evidence,
    OpportunityScore,
    ResearchRun,
    SessionLocal,
    Source,
)
from opportunity_engine import (
    WEIGHT_AUDIENCE_DEMAND,
    WEIGHT_BRAND_FIT,
    WEIGHT_COMPETITIVE_WHITESPACE,
    WEIGHT_FEASIBILITY,
    compute_overall_score,
)

# Known weak domain tokens
WEAK_DOMAIN_TOKENS = ("scribd", "linkedin", "quora", "reddit")


def extract_domain(url: str) -> str:
    """Extract clean domain hostname from a URL."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":", 1)[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def evaluate_domain_quality(
    domain: str, sources_for_domain: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Evaluates whether a domain is weak based on:
    - scribd, linkedin, quora, reddit
    - unnamed surveys (indicated in title, publisher, url, or claim text)
    Returns (is_weak, reasons).
    """
    reasons: List[str] = []
    lower_domain = domain.lower()

    for token in WEAK_DOMAIN_TOKENS:
        if token in lower_domain:
            reasons.append(token)

    # Check for unnamed survey indicators across sources under this domain
    unnamed_survey_detected = False
    for src in sources_for_domain:
        title = (src.get("title") or "").lower()
        publisher = (src.get("publisher") or "").lower()
        url = (src.get("url") or "").lower()
        claim = (src.get("claim") or "").lower()

        if "unnamed survey" in title or "unnamed survey" in publisher or "unnamed survey" in claim:
            unnamed_survey_detected = True
            break
        if "survey" in title or "survey" in url or "survey" in publisher:
            # Check if publisher is missing/unnamed or user-generated repository
            if not publisher or "unnamed" in publisher or any(tok in lower_domain for tok in WEAK_DOMAIN_TOKENS):
                unnamed_survey_detected = True
                break

    if unnamed_survey_detected and "unnamed survey" not in reasons:
        reasons.append("unnamed survey")

    is_weak = len(reasons) > 0
    return is_weak, reasons


def is_evidence_weak(ev: Evidence) -> bool:
    """Checks whether a single evidence item comes from a weak source."""
    if not ev.source or not ev.source.url:
        return False
    dom = extract_domain(ev.source.url)
    is_weak, _ = evaluate_domain_quality(
        dom,
        [{
            "title": ev.source.title or "",
            "publisher": ev.source.publisher or "",
            "url": ev.source.url or "",
            "claim": ev.claim or "",
        }]
    )
    return is_weak


def get_db_session(db_type: str = "test") -> Session:
    """
    Creates a SQLAlchemy session for the specified database ('test' or 'live').
    Default is 'test'.
    """
    if db_type == "test":
        test_url = os.environ.get("TEST_DATABASE_URL")
        if not test_url:
            test_env = dotenv_values(".env.test")
            test_url = test_env.get("TEST_DATABASE_URL")
        if not test_url:
            raise ValueError("TEST_DATABASE_URL is not set in environment or .env.test")
        engine = create_engine(test_url, pool_pre_ping=True)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return session_factory()
    elif db_type == "live":
        from config import DATABASE_URL
        live_url = os.environ.get("DATABASE_URL") or DATABASE_URL
        if not live_url:
            raise ValueError("DATABASE_URL is not configured.")
        engine = create_engine(live_url, pool_pre_ping=True)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return session_factory()
    else:
        raise ValueError(f"Unknown db_type: '{db_type}'. Must be 'test' or 'live'.")


def fetch_run_report_data(
    research_run_id: str, session: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Fetches all stored entities for the given research_run_id.
    Excludes opportunities that have no stored opportunity score.
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    try:
        run = session.query(ResearchRun).filter_by(id=research_run_id).first()
        if not run:
            raise ValueError(f"Research run '{research_run_id}' not found.")

        # 1. Scored opportunities only (non-superseded), ordered by rank asc
        opp_scores: List[OpportunityScore] = (
            session.query(OpportunityScore)
            .filter_by(research_run_id=research_run_id)
            .filter(OpportunityScore.superseded_at.is_(None))
            .order_by(OpportunityScore.rank.asc())
            .all()
        )

        # 2. Creative gaps mapped by ID
        creative_gaps_list: List[CreativeGap] = (
            session.query(CreativeGap)
            .filter_by(research_run_id=research_run_id)
            .filter(CreativeGap.superseded_at.is_(None))
            .all()
        )
        creative_gaps_map = {cg.id: cg for cg in creative_gaps_list}

        # 3. Consumer insights mapped by ID
        insights_list: List[ConsumerInsight] = (
            session.query(ConsumerInsight)
            .filter_by(research_run_id=research_run_id)
            .filter(ConsumerInsight.superseded_at.is_(None))
            .all()
        )
        insights_map = {ci.id: ci for ci in insights_list}

        # 4. Evidence items mapped by ID, joined with Source
        evidence_list: List[Evidence] = (
            session.query(Evidence)
            .options(joinedload(Evidence.source))
            .filter_by(research_run_id=research_run_id)
            .all()
        )
        evidence_map = {ev.id: ev for ev in evidence_list}

        # 5. Content pieces (non-superseded)
        content_pieces: List[ContentPiece] = (
            session.query(ContentPiece)
            .filter_by(research_run_id=research_run_id)
            .filter(ContentPiece.superseded_at.is_(None))
            .order_by(ContentPiece.id.asc())
            .all()
        )

        # 6. Competitive Intelligence
        ci_report: Optional[CompetitiveIntelligence] = (
            session.query(CompetitiveIntelligence)
            .filter_by(research_run_id=research_run_id)
            .filter(CompetitiveIntelligence.superseded_at.is_(None))
            .order_by(CompetitiveIntelligence.id.desc())
            .first()
        )
        competitors_data = ci_report.competitors if (ci_report and ci_report.competitors) else []
        competitive_gaps_data = ci_report.competitive_gaps if (ci_report and ci_report.competitive_gaps) else []

        # Calculate evidence counts and reused evidence count
        total_evidence_count = len(evidence_list)
        reused_evidence_count = 0
        if evidence_list:
            source_ids = [ev.source_id for ev in evidence_list]
            reused_source_ids = set(
                r[0] for r in session.query(Evidence.source_id)
                .filter(Evidence.source_id.in_(source_ids))
                .filter(Evidence.research_run_id != research_run_id)
                .all()
            )
            reused_evidence_count = sum(1 for ev in evidence_list if ev.source_id in reused_source_ids)

        # 7. Scored opportunity bundles (strict: ignore gaps with no score)
        scored_sections = []
        for opp in opp_scores:
            gap = creative_gaps_map.get(opp.creative_gap_id)
            if not gap:
                # If gap was superseded or missing, skip
                continue

            # Cited consumer insights
            cited_ci_ids = gap.supporting_consumer_insight_ids or []
            cited_insights = [insights_map[cid] for cid in cited_ci_ids if cid in insights_map]

            # Cited competitors
            cited_competitors = gap.supporting_competitor_names or []

            # Supporting evidence collected strictly through cited consumer insights
            cited_ev_ids: List[int] = []
            for ci in cited_insights:
                for eid in (ci.supporting_evidence_ids or []):
                    if eid in evidence_map and eid not in cited_ev_ids:
                        cited_ev_ids.append(eid)

            supporting_evidence = [evidence_map[eid] for eid in cited_ev_ids]

            # Count cited evidence items from weak sources
            weak_source_count = sum(1 for ev in supporting_evidence if is_evidence_weak(ev))

            # Recompute overall score deterministically from stored component scores
            recomputed_overall = compute_overall_score(
                audience_demand=opp.audience_demand_score,
                competitive_whitespace=opp.competitive_whitespace_score,
                brand_fit=opp.brand_fit_score,
                feasibility=opp.feasibility_score,
            )

            scored_sections.append({
                "opportunity_score": opp,
                "creative_gap": gap,
                "recomputed_overall_score": recomputed_overall,
                "cited_insights": cited_insights,
                "cited_competitors": cited_competitors,
                "supporting_evidence": supporting_evidence,
                "weak_source_count": weak_source_count,
            })

        # 8. Collect all unique sources across evidence in the run
        all_sources_by_domain: Dict[str, List[Dict[str, Any]]] = {}
        for ev in evidence_list:
            src = ev.source
            if src and src.url:
                dom = extract_domain(src.url)
                if dom:
                    all_sources_by_domain.setdefault(dom, []).append({
                        "url": src.url,
                        "title": src.title or "",
                        "publisher": src.publisher or "",
                        "claim": ev.claim or "",
                    })

        return {
            "run": run,
            "scored_sections": scored_sections,
            "content_pieces": content_pieces,
            "sources_by_domain": all_sources_by_domain,
            "competitors": competitors_data,
            "competitive_gaps": competitive_gaps_data,
            "total_evidence_count": total_evidence_count,
            "reused_evidence_count": reused_evidence_count,
        }
    finally:
        if close_session:
            session.close()


def generate_markdown_report(
    research_run_id: str,
    session: Optional[Session] = None,
    db_type: str = "test",
) -> str:
    """
    Generates a structured markdown report for a research run with NO free-written text.
    - One section per scored opportunity (in rank order)
    - Recomputed overall_score from stored component scores
    - AI reasoning labeled explicitly as "AI reasoning (not evidence)"
    - Cited consumer insights with exact stored text
    - Cited competitors
    - Supporting evidence claims with source title and URL
    - Weak evidence count per opportunity and 'Evidence fit needs human review.'
    - Competitive landscape section with competitors, competitive gaps, evidence count, and reused_evidence_count
    - Claims to verify section from content pieces
    - Source quality section listing domains and flagging weak ones
    - Excludes unscored opportunities entirely
    """
    close_session = False
    if session is None:
        session = get_db_session(db_type)
        close_session = True

    try:
        data = fetch_run_report_data(research_run_id, session=session)
    finally:
        if close_session:
            session.close()

    run: ResearchRun = data["run"]
    scored_sections = data["scored_sections"]
    content_pieces: List[ContentPiece] = data["content_pieces"]
    sources_by_domain: Dict[str, List[Dict[str, Any]]] = data["sources_by_domain"]
    competitors_list = data["competitors"]
    competitive_gaps_list = data["competitive_gaps"]
    total_evidence_count = data["total_evidence_count"]
    reused_evidence_count = data["reused_evidence_count"]

    lines: List[str] = []

    # 1. Header / Brief Section (stored text only)
    lines.append(f"# Research Report: {run.brand}")
    lines.append("")
    lines.append(f"- **Brand**: {run.brand}")
    lines.append(f"- **Market**: {run.market}")
    lines.append(f"- **Audience**: {run.audience}")
    lines.append(f"- **Objective**: {run.objective}")
    lines.append("")

    # 2. Competitive Landscape Section
    lines.append("## Competitive Landscape")
    lines.append("")
    lines.append(f"- **Total Evidence Count**: {total_evidence_count}")
    lines.append(f"- **Reused Evidence Count**: {reused_evidence_count}")
    lines.append("")
    lines.append("### Competitors")
    lines.append("")
    if competitors_list:
        for comp in competitors_list:
            if isinstance(comp, dict):
                cname = comp.get("name", "Competitor")
                pos = comp.get("positioning", "")
                strengths = comp.get("strengths", [])
                weaknesses = comp.get("weaknesses", [])
                lines.append(f"- **{cname}**: {pos}")
                if strengths:
                    lines.append(f"  - **Strengths**: {', '.join(strengths)}")
                if weaknesses:
                    lines.append(f"  - **Weaknesses**: {', '.join(weaknesses)}")
            else:
                lines.append(f"- **{comp}**")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("### Competitive Gaps")
    lines.append("")
    if competitive_gaps_list:
        for cg in competitive_gaps_list:
            if isinstance(cg, dict):
                gid = cg.get("id")
                gtext = cg.get("gap") or cg.get("description") or ""
                lines.append(f"- **Gap {gid}**: {gtext}")
            else:
                lines.append(f"- {cg}")
    else:
        lines.append("- None")
    lines.append("")

    # 3. Scored Opportunities in Rank Order
    for item in scored_sections:
        opp: OpportunityScore = item["opportunity_score"]
        gap: CreativeGap = item["creative_gap"]
        recomputed_score: float = item["recomputed_overall_score"]
        insights: List[ConsumerInsight] = item["cited_insights"]
        competitors: List[str] = item["cited_competitors"]
        evidence_items: List[Evidence] = item["supporting_evidence"]
        weak_count: int = item["weak_source_count"]

        lines.append(f"## Opportunity Rank {opp.rank}: {gap.gap_title}")
        lines.append("")
        lines.append(f"**Gap Description**: {gap.description}")
        lines.append("")

        # Scores Table (recompute overall_score; label reasoning as 'AI reasoning (not evidence)')
        lines.append("### Scores Table")
        lines.append("")
        lines.append("| Criterion | Score | AI reasoning (not evidence) |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(f"| Audience Demand | {opp.audience_demand_score} | {opp.audience_demand_reasoning} |")
        lines.append(f"| Competitive Whitespace | {opp.competitive_whitespace_score} | {opp.competitive_whitespace_reasoning} |")
        lines.append(f"| Brand Fit | {opp.brand_fit_score} | {opp.brand_fit_reasoning} |")
        lines.append(f"| Feasibility | {opp.feasibility_score} | {opp.feasibility_reasoning} |")
        lines.append(f"| Recomputed Overall Score | {recomputed_score:.2f} | Overall weighted evaluation |")
        lines.append("")

        # Cited Consumer Insights
        lines.append("### Cited Consumer Insights")
        lines.append("")
        if insights:
            for ci in insights:
                lines.append(f"- **Insight #{ci.id}** ({ci.insight_type}): {ci.insight_text}")
        else:
            lines.append("- None")
        lines.append("")

        # Cited Competitors
        lines.append("### Cited Competitors")
        lines.append("")
        if competitors:
            for comp in competitors:
                lines.append(f"- {comp}")
        else:
            lines.append("- None")
        lines.append("")

        # Supporting Evidence Claims with Source Title and URL
        lines.append("### Supporting Evidence")
        lines.append("")
        lines.append(f"- **Cited evidence items from weak sources**: {weak_count}")
        lines.append("- Evidence fit needs human review.")
        lines.append("")
        if evidence_items:
            for ev in evidence_items:
                src_title = ev.source.title if (ev.source and ev.source.title) else ""
                src_url = ev.source.url if (ev.source and ev.source.url) else ""
                lines.append(f"- **Claim**: {ev.claim}")
                lines.append(f"  - **Source Title**: {src_title}")
                lines.append(f"  - **Source URL**: {src_url}")
        else:
            lines.append("- None")
        lines.append("")

    # 4. Claims to Verify Section
    lines.append("## Claims to Verify")
    lines.append("")
    claims_found = False
    for cp in content_pieces:
        if cp.claims_to_verify:
            for claim_item in cp.claims_to_verify:
                claims_found = True
                lines.append(f"- {claim_item} (Content Piece: {cp.title})")
    if not claims_found:
        lines.append("- None")
    lines.append("")

    # 5. Source Quality Section
    lines.append("## Source Quality")
    lines.append("")
    if sources_by_domain:
        sorted_domains = sorted(sources_by_domain.keys())
        for dom in sorted_domains:
            is_weak, reasons = evaluate_domain_quality(dom, sources_by_domain[dom])
            if is_weak:
                reasons_str = ", ".join(reasons)
                lines.append(f"- `{dom}`: WEAK ({reasons_str})")
            else:
                lines.append(f"- `{dom}`: OK")
    else:
        lines.append("- None")
    lines.append("")

    return "\n".join(lines)


# Backwards compatibility alias
generate_report = generate_markdown_report


def save_report_to_file(
    research_run_id: str,
    output_path: str,
    session: Optional[Session] = None,
    db_type: str = "test",
) -> str:
    """Generates the report and writes it to the specified file path."""
    content = generate_markdown_report(research_run_id, session=session, db_type=db_type)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate markdown report for a CatchyLab research run.")
    parser.add_argument("research_run_id", help="UUID of the research run")
    parser.add_argument(
        "--db",
        choices=["test", "live"],
        default="test",
        help="Database to target: 'test' (default) or 'live'.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output markdown path. Default: outputs/report_<run_id>.md",
    )
    args = parser.parse_args()

    run_id = args.research_run_id
    out_path = args.out or f"outputs/report_{run_id}.md"

    if args.db == "live":
        print(f"ATTENTION: You have targeted the LIVE database for run '{run_id}'.")
        print("Live research run data may not exist or may be production-facing.")
        try:
            confirmation = input("Type 'CONFIRM' to proceed with the live database: ").strip()
        except EOFError:
            confirmation = ""
        if confirmation != "CONFIRM":
            print("Aborted. Typed confirmation 'CONFIRM' was not received.")
            sys.exit(1)

    session = get_db_session(args.db)
    try:
        report_md = generate_markdown_report(run_id, session=session, db_type=args.db)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Report successfully written to {out_path}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
