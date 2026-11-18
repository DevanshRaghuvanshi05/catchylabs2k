"""
Pipeline orchestrator module for CatchyLab Agent 1 (Researcher).
Executes the full 7-stage market research pipeline.
"""

import os
import re
import json
import time
import uuid
import logging
import hashlib
import contextvars
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai

from config import GEMINI_API_KEY, SERPAPI_API_KEY
from search_engine import search_web_serpapi
from evidence_extractor import generate_queries, extract_evidence_from_doc, check_evidence_coverage
from database import perform_semantic_recall, save_research_run, init_research_run, current_research_run_id, current_stage, redact_secrets
from intelligence import build_research_context

logger = logging.getLogger("CatchyLabResearcher")


def normalize_text(text: str) -> str:
    """Normalizes text string for deduplication."""
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def filter_documents(all_retrieved: list[dict], top_n_per_query: int = 3) -> list[dict]:
    """Filters retrieved documents keeping top N results per query with unique URLs."""
    logger.info(f"=== STAGE 4: Filtering Documents (Top {top_n_per_query} per query) ===")
    seen_urls = set()
    filtered_docs = []
    query_counts = {}
    
    for doc in all_retrieved:
        url = doc.get("url")
        q = doc.get("query")
        if not url or url in seen_urls:
            continue
        
        current_count = query_counts.get(q, 0)
        if current_count < top_n_per_query:
            seen_urls.add(url)
            query_counts[q] = current_count + 1
            filtered_docs.append(doc)
            
    logger.info(f"Retained {len(filtered_docs)} documents out of {len(all_retrieved)} raw results.")
    return filtered_docs


def extract_all_evidence(
    client: genai.Client,
    documents: list[dict],
    brief: dict,
    max_concurrency: int = 1
) -> tuple[list[dict], dict]:
    """Iterates through documents and extracts structured evidence (sequential by default to respect API quotas)."""
    concurrency_label = "Sequential (concurrency=1)" if max_concurrency <= 1 else f"Parallel (max {max_concurrency} workers)"
    logger.info(f"=== STAGE 5: Extracting Evidence from {len(documents)} Documents ({concurrency_label}) ===")
    extracted_evidence = []
    
    stats = {
        "documents_processed": len(documents),
        "documents_successful_extraction": 0,
        "documents_skipped_no_claim": 0,
        "documents_failed_extraction": 0
    }
    
    if not documents:
        return [], stats

    def _process_doc(item):
        idx, doc = item
        title_snippet = doc.get("title", "No Title")[:50]
        doc_url = doc.get("url", "No URL")
        logger.info(f" Processing doc {idx}/{len(documents)}: {title_snippet} ({doc_url})")
        res = extract_evidence_from_doc(client, doc, brief)
        return idx, doc, res

    def _handle_result(idx, doc, extraction_res):
        status = extraction_res.get("status")
        if status == "success":
            evidence = extraction_res.get("data")
            extracted_evidence.append(evidence)
            stats["documents_successful_extraction"] += 1
            claim_text = evidence.get("claim", "")[:80] if isinstance(evidence, dict) else ""
            logger.info(f"  [+] Doc {idx}/{len(documents)} SUCCESS: Claim Extracted -> '{claim_text}...'")
        elif status == "no_claim":
            stats["documents_skipped_no_claim"] += 1
            reason = extraction_res.get("reason", "No claim")
            logger.info(f"  [-] Doc {idx}/{len(documents)} SKIPPED: {reason}")
        else:
            stats["documents_failed_extraction"] += 1
            reason = extraction_res.get("reason", "Unknown failure")
            raw_text = extraction_res.get("raw_response")
            raw_snippet = repr(raw_text[:150]) if raw_text else "None"
            logger.error(f"  [!] Doc {idx}/{len(documents)} FAILED EXTRACTION: Reason: {reason} | Raw Output: {raw_snippet}")

    if max_concurrency <= 1:
        for idx, doc in enumerate(documents, 1):
            try:
                _, _, extraction_res = _process_doc((idx, doc))
                _handle_result(idx, doc, extraction_res)
            except Exception as exc:
                stats["documents_failed_extraction"] += 1
                logger.error(f"  [!] Exception during doc processing: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = [executor.submit(contextvars.copy_context().run, _process_doc, (idx, doc)) for idx, doc in enumerate(documents, 1)]
            for future in as_completed(futures):
                try:
                    idx, doc, extraction_res = future.result()
                    _handle_result(idx, doc, extraction_res)
                except Exception as exc:
                    stats["documents_failed_extraction"] += 1
                    logger.error(f"  [!] Exception during parallel doc processing: {exc}")

    logger.info(
        f"Stage 5 Extraction Summary -> Total Processed: {stats['documents_processed']} | "
        f"Success: {stats['documents_successful_extraction']} | "
        f"Skipped (No Claim): {stats['documents_skipped_no_claim']} | "
        f"Failed: {stats['documents_failed_extraction']}"
    )
    return extracted_evidence, stats


def deduplicate_evidence(evidence_list: list[dict]) -> list[dict]:
    """Deduplicates evidence items based on claim text hash and URL."""
    logger.info(f"=== STAGE 6: Deduplicating Evidence ({len(evidence_list)} initial items) ===")
    unique_claims = {}
    
    for item in evidence_list:
        claim_text = item.get("claim", "")
        norm_text = normalize_text(claim_text)
        url = item.get("source", {}).get("url", "") if isinstance(item.get("source"), dict) else ""
        
        claim_hash = hashlib.md5(f"{url}:{norm_text[:100]}".encode("utf-8")).hexdigest()
        
        if claim_hash in unique_claims:
            existing_conf = unique_claims[claim_hash].get("confidence", 0.0)
            new_conf = item.get("confidence", 0.0)
            if new_conf > existing_conf:
                unique_claims[claim_hash] = item
        else:
            unique_claims[claim_hash] = item
            
    deduped = list(unique_claims.values())
    removed_count = len(evidence_list) - len(deduped)
    logger.info(f"Deduplication complete. Removed {removed_count} duplicates. Final count: {len(deduped)}")
    return deduped


def save_report(
    brief: dict,
    queries: list[str],
    total_raw_docs: int,
    evidence_items: list[dict],
    extraction_stats: dict = None,
    coverage_info: dict = None,
    output_dir: str = ".",
    research_run_id: str = None,
    search_errors: list = None,
    research_context: dict = None
) -> str:
    """Saves final structured evidence and metadata to a timestamped JSON file."""
    logger.info("=== STAGE 7: Writing Final Research Report ===")
    
    stats = extraction_stats or {}
    cov = coverage_info or {}
    brand_slug = re.sub(r"[^a-zA-Z0-9]", "_", brief.get("brand", "brand").lower())
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{brand_slug}_{timestamp_str}.json"
    filepath = os.path.join(output_dir, filename)
    
    is_success = len(evidence_items) > 0
    failure_reason = None
    if not is_success:
        if stats.get("documents_processed", 0) > 0:
            failure_reason = (
                f"Zero evidence extracted from {stats.get('documents_processed', 0)} processed documents "
                f"({stats.get('documents_failed_extraction', 0)} failed, "
                f"{stats.get('documents_skipped_no_claim', 0)} skipped with no claim)."
            )
        elif search_errors:
            failure_reason = redact_secrets(f"Search provider failure: {search_errors[0]}")
        else:
            failure_reason = "Search returned zero documents across all queries."

    # Compute coverage evaluation status: success, failed, or not_evaluated
    cov_eval_status = cov.get("coverage_evaluation_status")
    if not cov_eval_status:
        cov_res = cov.get("coverage_check_result")
        if isinstance(cov_res, dict):
            if cov_res.get("evaluated") is True or cov_res.get("status") == "success":
                cov_eval_status = "success"
            elif cov_res.get("evaluated") is False or cov_res.get("status") == "failed":
                cov_eval_status = "failed"
            else:
                cov_eval_status = "not_evaluated"
        else:
            cov_eval_status = "not_evaluated"
    
    research_context = research_context or build_research_context(
        brief, evidence_items, search_queries=queries
    )
    report_data = {
        "metadata": {
            "research_run_id": research_run_id or brief.get("research_run_id"),
            "brand": brief.get("brand"),
            "market": brief.get("market"),
            "audience": brief.get("audience"),
            "objective": brief.get("objective"),
            "language_preference": brief.get("language_preference", "auto"),
            "generated_at": datetime.now().isoformat(),
            "status": "success" if is_success else "failed",
            "error": failure_reason,
            "queries_used": queries,
            "total_raw_docs_retrieved": total_raw_docs,
            "total_evidence_count": len(evidence_items),
            "documents_processed": stats.get("documents_processed", 0),
            "documents_successful_extraction": stats.get("documents_successful_extraction", 0),
            "documents_skipped_no_claim": stats.get("documents_skipped_no_claim", 0),
            "documents_failed_extraction": stats.get("documents_failed_extraction", 0),
            "coverage_evaluation_status": cov_eval_status,
            "coverage_rounds": cov.get("coverage_rounds", 1),
            "coverage_gaps_identified": cov.get("coverage_gaps_identified", []),
            "additional_queries_used": cov.get("additional_queries_used", []),
            "reused_evidence_count": cov.get("reused_evidence_count", 0),
            "fresh_evidence_count": cov.get("fresh_evidence_count", 0)
        },
        "evidence": evidence_items,
        "research_context": research_context,
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Research Report written to: {os.path.abspath(filepath)}")
    return filepath


def run_research_pipeline(
    brief: dict,
    gemini_key: str = None,
    serpapi_key: str = None,
    max_documents: int = None,
    research_run_id: str = None
) -> dict:
    """Executes the full research pipeline given a brief dictionary."""
    pipeline_start_time = time.time()
    # Resolve API keys — prefer explicitly passed keys (from jobs.py credential resolution),
    # then fall through to credential_manager which enforces BYOK policy.
    if gemini_key:
        g_key = gemini_key
    else:
        from services.credential_manager import get_gemini_key as _get_gemini
        g_key = _get_gemini(user_id=brief.get('user_id'))

    if serpapi_key:
        s_key = serpapi_key
    else:
        from services.credential_manager import get_serpapi_key as _get_serpapi
        s_key = _get_serpapi(user_id=brief.get('user_id'))
    
    run_id = research_run_id or str(uuid.uuid4())
    init_research_run(run_id, brief)
    current_research_run_id.set(run_id)
    
    logger.info(f"Starting CatchyLab Agent 1 (Researcher) Pipeline [Run ID: {run_id}]")
    logger.info(f"Brief: Brand={brief.get('brand')}, Market={brief.get('market')}, Audience={brief.get('audience')}")
    
    client = genai.Client(api_key=g_key)
    
    # 1.5 Semantic Recall Stage (pgvector search across past evidence)
    current_stage.set("Stage 1.5")
    reused_evidence = perform_semantic_recall(client, brief)

    # 2. Query Generation (Round 1)
    current_stage.set("Stage 2")
    queries = generate_queries(client, brief)
    
    # 3. Search / Retrieval (Round 1 - Parallel)
    current_stage.set("Stage 3")
    logger.info(f"=== STAGE 3: Search & Retrieval across {len(queries)} Queries (Parallel) ===")
    all_docs = []
    search_errors = []
    with ThreadPoolExecutor(max_workers=min(5, len(queries))) as executor:
        future_to_query = {
            executor.submit(contextvars.copy_context().run, search_web_serpapi, q, api_key=s_key, max_results=4): q 
            for q in queries
        }
        for future in as_completed(future_to_query):
            q = future_to_query[future]
            try:
                docs = future.result()
                all_docs.extend(docs)
            except Exception as exc:
                search_errors.append(f"Query '{q}' error: {exc}")
                logger.error(f"Search query '{q}' generated error: {exc}")

    logger.info(f"Total raw docs retrieved across all queries: {len(all_docs)}")
    
    # 4. Relevance Filtering (Round 1)
    current_stage.set("Stage 4")
    filtered_docs = filter_documents(all_docs, top_n_per_query=3)
    
    if max_documents and len(filtered_docs) > max_documents:
        logger.info(f"Limiting filtered documents from {len(filtered_docs)} to {max_documents} for testing/quota management.")
        filtered_docs = filtered_docs[:max_documents]
    
    # 5. Evidence Extraction (Round 1 - Parallel)
    current_stage.set("Stage 5")
    raw_evidence_r1, extraction_stats_r1 = extract_all_evidence(client, filtered_docs, brief)
    
    # 6. Deduplication (Round 1)
    current_stage.set("Stage 6")
    deduped_r1 = deduplicate_evidence(raw_evidence_r1)
    
    # 5.5 Coverage Check Stage
    current_stage.set("Stage 5.5")
    coverage_res = check_evidence_coverage(client, brief, deduped_r1)
    
    coverage_rounds = 1
    gaps_identified = coverage_res.get("gaps", [])
    additional_queries = coverage_res.get("additional_queries", [])
    additional_queries_used = []
    
    final_evidence = deduped_r1
    total_raw_docs_count = len(all_docs)
    combined_queries = list(queries)
    combined_stats = dict(extraction_stats_r1)

    cov_evaluated = coverage_res.get("evaluated", coverage_res.get("status") == "success")
    if not cov_evaluated:
        logger.warning(
            f"=== STAGE 5.5: Coverage check evaluation failed: {coverage_res.get('error')}. "
            "Proceeding with existing evidence without triggering additional rounds."
        )
    elif not coverage_res.get("sufficient") and additional_queries:
        logger.info(f"=== STAGE 5.5: Coverage Insufficient! Triggering Round 2 with {len(additional_queries)} Additional Queries ===")
        coverage_rounds = 2
        additional_queries_used = additional_queries
        combined_queries.extend(additional_queries)
        
        # Round 2: Search -> Filter -> Extract
        current_stage.set("Stage 3")
        logger.info(f"=== ROUND 2: Search & Retrieval for {len(additional_queries)} Additional Queries (Parallel) ===")
        all_docs_r2 = []
        with ThreadPoolExecutor(max_workers=min(5, len(additional_queries))) as executor:
            future_to_query = {
                executor.submit(contextvars.copy_context().run, search_web_serpapi, q, api_key=s_key, max_results=4): q 
                for q in additional_queries
            }
            for future in as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    docs = future.result()
                    all_docs_r2.extend(docs)
                except Exception as exc:
                    search_errors.append(f"Round 2 query '{q}' error: {exc}")
                    logger.error(f"Round 2 search query '{q}' error: {exc}")

        total_raw_docs_count += len(all_docs_r2)
        current_stage.set("Stage 4")
        filtered_docs_r2 = filter_documents(all_docs_r2, top_n_per_query=3)
        
        if max_documents and len(filtered_docs_r2) > max_documents:
            filtered_docs_r2 = filtered_docs_r2[:max_documents]

        current_stage.set("Stage 5")
        raw_evidence_r2, extraction_stats_r2 = extract_all_evidence(client, filtered_docs_r2, brief)
        
        # Merge evidence & deduplicate combined set
        current_stage.set("Stage 6")
        merged_raw_evidence = deduped_r1 + raw_evidence_r2
        final_evidence = deduplicate_evidence(merged_raw_evidence)
        
        for k in combined_stats:
            combined_stats[k] += extraction_stats_r2.get(k, 0)
            
        current_stage.set("Stage 5.5")
        logger.info("=== STAGE 5.5 (Round 2 Check): Evaluating Coverage After Round 2 ===")
        coverage_res_r2 = check_evidence_coverage(client, brief, final_evidence)
        cov_r2_evaluated = coverage_res_r2.get("evaluated", coverage_res_r2.get("status") == "success")
        if not cov_r2_evaluated:
            logger.warning(
                f"Round 2 coverage evaluation failed: {coverage_res_r2.get('error')}. "
                "Proceeding with available evidence."
            )
        elif not coverage_res_r2.get("sufficient"):
            logger.info("Coverage still insufficient after round 2, proceeding with available evidence")
            if coverage_res_r2.get("gaps"):
                gaps_identified = coverage_res_r2.get("gaps")
        else:
            logger.info("Coverage is now sufficient after round 2. Proceeding to write final report.")
    else:
        logger.info("Coverage evaluation passed (sufficient). Proceeding straight to final report writing.")

    # Merge recalled past evidence with fresh evidence
    if reused_evidence:
        logger.info(f"Merging {len(reused_evidence)} recalled past evidence items with fresh evidence...")
        final_evidence = deduplicate_evidence(reused_evidence + final_evidence)

    reused_count = sum(1 for item in final_evidence if item.get("reused_from_run_id"))
    fresh_count = len(final_evidence) - reused_count

    logger.info(f"Final Evidence Breakdown -> Reused: {reused_count} | Fresh: {fresh_count} | Total: {len(final_evidence)}")

    coverage_info = {
        "coverage_rounds": coverage_rounds,
        "coverage_gaps_identified": gaps_identified,
        "additional_queries_used": additional_queries_used,
        "coverage_check_result": coverage_res,
        "coverage_evaluation_status": "success" if cov_evaluated else "failed",
        "reused_evidence_count": reused_count,
        "fresh_evidence_count": fresh_count
    }
    research_context = build_research_context(
        brief, final_evidence, search_queries=combined_queries
    )

    # 7. Output Persistence
    current_stage.set("Stage 7")
    report_file = save_report(
        brief,
        combined_queries,
        total_raw_docs_count,
        final_evidence,
        extraction_stats=combined_stats,
        coverage_info=coverage_info,
        research_run_id=run_id,
        search_errors=search_errors,
        research_context=research_context
    )
    
    total_elapsed = time.time() - pipeline_start_time

    if len(final_evidence) == 0:
        if combined_stats.get('documents_processed', 0) > 0:
            err_msg = (
                f"Research pipeline produced zero evidence items for brand '{brief.get('brand')}'. "
                f"Processed {combined_stats.get('documents_processed', 0)} documents "
                f"({combined_stats.get('documents_failed_extraction', 0)} failed extraction, "
                f"{combined_stats.get('documents_skipped_no_claim', 0)} skipped with no claim). "
                f"Diagnostic report preserved at: {report_file}"
            )
        elif search_errors:
            err_msg = (
                f"Research pipeline failed: Search provider error for brand '{brief.get('brand')}'. "
                f"{redact_secrets(search_errors[0])}. Diagnostic report preserved at: {report_file}"
            )
        else:
            err_msg = (
                f"Research pipeline produced zero evidence items for brand '{brief.get('brand')}'. "
                f"Search returned zero documents across all queries. "
                f"Diagnostic report preserved at: {report_file}"
            )
        logger.error(f"[PIPELINE FAILED] {err_msg}")
        raise ValueError(err_msg)

    logger.info(f"PIPELINE COMPLETE in {total_elapsed:.2f} seconds! Total evidence count: {len(final_evidence)}")
    
    return {
        "research_run_id": run_id,
        "report_file": report_file,
        "report_data": {
            "metadata": {
                "research_run_id": run_id,
                "brand": brief.get("brand"),
                "market": brief.get("market"),
                "audience": brief.get("audience"),
                "objective": brief.get("objective"),
                "generated_at": datetime.now().isoformat(),
                "status": "success",
                "error": None,
                "queries_used": combined_queries,
                "total_raw_docs_retrieved": total_raw_docs_count,
                "total_evidence_count": len(final_evidence),
                "documents_processed": combined_stats.get("documents_processed", 0),
                "documents_successful_extraction": combined_stats.get("documents_successful_extraction", 0),
                "documents_skipped_no_claim": combined_stats.get("documents_skipped_no_claim", 0),
                "documents_failed_extraction": combined_stats.get("documents_failed_extraction", 0),
                "coverage_evaluation_status": "success" if cov_evaluated else "failed",
                "coverage_rounds": coverage_rounds,
                "coverage_gaps_identified": gaps_identified,
                "additional_queries_used": additional_queries_used,
                "reused_evidence_count": reused_count,
                "fresh_evidence_count": fresh_count
            },
            "evidence": final_evidence,
            "research_context": research_context
        },
        "research_context": research_context,
        "evidence_count": len(final_evidence),
        "queries": combined_queries,
        "extraction_stats": combined_stats,
        "coverage_info": coverage_info,
        "elapsed_seconds": round(total_elapsed, 2)
    }
