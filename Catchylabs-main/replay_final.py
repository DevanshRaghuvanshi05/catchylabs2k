"""
Replay validators on stored content pieces with updated rules.
"""

from database import SessionLocal, ContentPiece
from content_generator import validate_content_pieces, _brand_claim_errors

def replay_validators():
    """Replay all validators on stored content pieces."""
    session = SessionLocal()
    try:
        pieces = session.query(ContentPiece).filter(ContentPiece.id.in_([9, 10, 11, 12])).all()
        
        for piece in pieces:
            print(f"\n=== Piece #{piece.id}: {piece.title} ===")
            print(f"Content type: {piece.content_type}")
            
            # Reconstruct data for validation
            data = {
                "content_pieces": [{
                    "content_type": piece.content_type,
                    "title": piece.title,
                    "hook": piece.hook,
                    "body": piece.body,
                    "caption": piece.caption,
                    "call_to_action": piece.call_to_action,
                    "supporting_consumer_insight_ids": piece.supporting_consumer_insight_ids,
                    "supporting_competitor_names": piece.supporting_competitor_names,
                }]
            }
            
            # Get brand name from research run
            from database import ResearchRun
            run = session.query(ResearchRun).filter_by(id=piece.research_run_id).first()
            brand_name = run.brief.get("brand_name", "") if run else ""
            
            # Get cited evidence
            cited_evidence = ""
            # Collect from consumer insights
            for ci_id in piece.supporting_consumer_insight_ids:
                from database import ConsumerInsight
                ci = session.query(ConsumerInsight).filter_by(id=ci_id).first()
                if ci:
                    cited_evidence += ci.insight_text + " "
            
            # Run brand claim validation
            all_text = piece.hook + " " + piece.caption
            if piece.body:
                for scene in piece.body if isinstance(piece.body, list) else [piece.body]:
                    if isinstance(scene, dict):
                        all_text += " " + str(scene.get("voiceover", ""))
                        all_text += " " + str(scene.get("on_screen_text", ""))
            
            errors, claims = _brand_claim_errors(all_text, piece.id, brand_name, cited_evidence)
            
            print(f"\nBrand claims: {len(errors)} errors, {len(claims)} to verify")
            if claims:
                print("Claims to verify:")
                for claim in claims:
                    print(f"  - {claim}")
            
            # Run full validation
            try:
                validate_content_pieces(data, brand_name, cited_evidence)
                print("Full validation: PASS")
            except ValueError as e:
                print(f"Full validation: FAIL")
                print(f"Error: {e}")
    
    finally:
        session.close()

if __name__ == "__main__":
    replay_validators()
