import argparse
import contextlib
import io
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFERENCE_DIR = os.path.join(ROOT_DIR, "inference")
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT_DIR, ".env"))
except Exception:
    pass


JOBS: Dict[str, Dict[str, Any]] = {}
IDEMPOTENCY_KEYS: Dict[str, str] = {}
JOBS_LOCK = threading.Lock()
JOB_RUN_LOCK = threading.Lock()
SCHEMA_VERSION = "gtm_research.v1"


def _safe_bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _debug_ui_enabled() -> bool:
    return _safe_bool_env("RESEARCH_DEBUG_UI", "true")


def _service_info() -> Dict[str, Any]:
    return {
        "service": "gtm-research-api",
        "schema_version": SCHEMA_VERSION,
        "debug_ui_enabled": _debug_ui_enabled(),
        "production_endpoints": [
            "POST /api/research",
            "GET /api/research/{job_id}",
            "GET /api/research/{job_id}/result",
        ],
        "debug_endpoints": [
            "GET /",
            "GET /api/config",
            "GET /api/jobs",
            "GET /api/jobs/{job_id}",
            "POST /api/agent",
            "POST /api/tools",
            "POST /api/research-debug",
        ],
    }


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _redacted_ready() -> Dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "vllm").strip().lower()
    if provider == "azure":
        required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]
    elif provider == "openai":
        required = ["OPENAI_API_KEY", "OPENAI_MODEL"]
    else:
        required = ["VLLM_API_BASE"]
    missing = [name for name in required if not os.getenv(name)]
    return {
        "provider": provider,
        "ready": not missing or provider == "vllm",
        "missing": missing,
        "visit_backend": os.getenv("VISIT_BACKEND", "jina"),
        "max_calls": os.getenv("MAX_LLM_CALL_PER_RUN", "100"),
        "request_timeout": os.getenv("LLM_REQUEST_TIMEOUT", "600"),
    }


def _extract_tool_names(messages: List[Dict[str, Any]]) -> List[str]:
    names = []
    for message in messages:
        content = str(message.get("content", ""))
        for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.S):
            try:
                payload = json.loads(match)
            except json.JSONDecodeError:
                continue
            if payload.get("name"):
                names.append(payload["name"])
    return names


def _short_quote(text: Any, max_words: int = 22) -> str:
    if not isinstance(text, str):
        return ""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ""
    words = clean.split()
    if len(words) <= max_words:
        return clean
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def _tool_payloads_from_messages(messages: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    payloads = []
    for message in messages:
        content = str(message.get("content", ""))
        for block in re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", content, flags=re.S):
            chunks = re.split(r"\n\n(?=\[[^\]\n]+\]\n)", block.strip())
            for chunk in chunks:
                match = re.match(r"^\[([^\]\n]+)\]\n(.*)$", chunk.strip(), flags=re.S)
                if not match:
                    continue
                tool_name, raw_payload = match.groups()
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    continue
                payloads.append((tool_name, payload))
    return payloads


def _extract_sources(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = []
    seen = set()
    for tool_name, payload in _tool_payloads_from_messages(messages):
        if tool_name == "clinical_trials_search":
            for study in payload.get("studies", [])[:12]:
                nct_id = study.get("nct_id") or ""
                if not nct_id or ("trial", nct_id) in seen:
                    continue
                seen.add(("trial", nct_id))
                outcomes = study.get("primary_outcomes") or []
                first_outcome = outcomes[0] if outcomes else {}
                quote_text = (
                    first_outcome.get("description")
                    or first_outcome.get("measure")
                    or study.get("brief_summary")
                    or study.get("eligibility", {}).get("criteria")
                    or ""
                )
                sources.append({
                    "kind": "Clinical trial",
                    "source": payload.get("source", "ClinicalTrials.gov API v2"),
                    "id": nct_id,
                    "locator": "primary_outcomes[0].description" if first_outcome.get("description") else "brief_summary",
                    "title": study.get("brief_title") or study.get("official_title") or nct_id,
                    "url": study.get("url") or (f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""),
                    "status": study.get("overall_status"),
                    "conditions": study.get("conditions", []),
                    "quote": _short_quote(quote_text),
                })
        elif tool_name == "literature_search":
            for paper in payload.get("results", [])[:12]:
                source_id = paper.get("pmid") or paper.get("pmcid") or paper.get("doi") or paper.get("title") or ""
                if not source_id or ("paper", source_id) in seen:
                    continue
                seen.add(("paper", source_id))
                sources.append({
                    "kind": "Literature",
                    "source": payload.get("source", "Europe PMC REST API"),
                    "id": source_id,
                    "locator": "abstract",
                    "title": paper.get("title") or source_id,
                    "url": paper.get("url") or "",
                    "pmid": paper.get("pmid"),
                    "pmcid": paper.get("pmcid"),
                    "doi": paper.get("doi"),
                    "year": paper.get("year"),
                    "quote": _short_quote(paper.get("abstract", "")),
                })
        elif tool_name in {"extract_webpage", "visit"}:
            source_id = payload.get("url") or payload.get("title") or tool_name
            if ("web", source_id) in seen:
                continue
            seen.add(("web", source_id))
            sources.append({
                "kind": "Web page",
                "source": payload.get("source") or tool_name,
                "id": source_id,
                "locator": "text",
                "title": payload.get("title") or payload.get("description") or source_id,
                "url": payload.get("url") or "",
                "quote": _short_quote(payload.get("text") or payload.get("summary") or payload.get("description") or ""),
            })
        elif tool_name == "parse_pdf_grobid":
            source_id = payload.get("file") or payload.get("title") or tool_name
            if ("pdf", source_id) in seen:
                continue
            seen.add(("pdf", source_id))
            sources.append({
                "kind": "PDF",
                "source": payload.get("source", "GROBID"),
                "id": source_id,
                "locator": "abstract" if payload.get("abstract") else "body",
                "title": payload.get("title") or source_id,
                "url": payload.get("file") if str(payload.get("file", "")).startswith(("http://", "https://")) else "",
                "quote": _short_quote(payload.get("abstract") or payload.get("body") or ""),
            })
    for index, source in enumerate(sources, 1):
        source["source_id"] = f"S{index}"
        source["aliases"] = _source_aliases(source)
    return sources


def _source_aliases(source: Dict[str, Any]) -> List[str]:
    aliases = []
    for key in ("source_id", "id", "pmid", "pmcid", "doi", "url"):
        value = source.get(key)
        if value:
            aliases.append(str(value))
    if source.get("pmid"):
        aliases.extend([f"PMID:{source['pmid']}", f"PMID: {source['pmid']}", f"PMID {source['pmid']}"])
    if source.get("pmcid"):
        aliases.extend([f"PMCID:{source['pmcid']}", f"PMCID: {source['pmcid']}", f"PMCID {source['pmcid']}"])
    if source.get("doi"):
        aliases.extend([f"DOI:{source['doi']}", f"DOI: {source['doi']}", f"DOI {source['doi']}"])
    return sorted(set(alias for alias in aliases if alias), key=len, reverse=True)


def _plain_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", str(value or ""))
    value = re.sub(r"[*_`>#]+", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _claim_lines(prediction: str) -> List[str]:
    claims = []
    for raw_line in str(prediction or "").splitlines():
        line = _plain_text(raw_line)
        if len(line) < 25:
            continue
        if line.lower().strip(":") in {"sources", "evidence", "literature", "bottom line", "key trial"}:
            continue
        claims.append(line)
    if claims:
        return claims[:40]
    sentences = re.split(r"(?<=[.!?])\s+", _plain_text(prediction))
    return [sentence for sentence in sentences if len(sentence) >= 25][:40]


def _build_evidence_ledger(prediction: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    claims = []
    cited_source_ids = set()
    for claim in _claim_lines(prediction):
        matched = []
        normalized_claim = claim.lower()
        for source in sources:
            for alias in source.get("aliases", []):
                if alias and alias.lower() in normalized_claim:
                    matched.append(source["source_id"])
                    cited_source_ids.add(source["source_id"])
                    break
        explicit_ids = re.findall(r"\bS\d+\b", claim)
        for explicit_id in explicit_ids:
            if any(source.get("source_id") == explicit_id for source in sources):
                matched.append(explicit_id)
                cited_source_ids.add(explicit_id)
        matched = sorted(set(matched), key=lambda sid: int(sid[1:]) if sid[1:].isdigit() else 9999)
        if matched:
            claims.append({"claim": claim, "source_ids": matched})
    return {
        "sources": sources,
        "claims": claims,
        "cited_source_ids": sorted(cited_source_ids, key=lambda sid: int(sid[1:]) if sid[1:].isdigit() else 9999),
    }


def _risk_for_claim(claim: str, has_sources: bool) -> Tuple[str, str]:
    text = claim.lower()
    if not has_sources:
        return "unsupported", "do_not_use"
    restricted_terms = [
        "cure", "curative", "improves survival", "reduces mortality", "safe and effective",
        "proven efficacy", "best", "superior", "guaranteed", "transformative",
    ]
    comparative_terms = ["better than", "superior to", "outperforms", "leading", "only platform"]
    if any(term in text for term in restricted_terms):
        return "restricted", "medical_legal_review"
    if any(term in text for term in comparative_terms):
        return "review", "marketing_review"
    if any(term in text for term in ["patient outcome", "clinical benefit", "efficacy", "treatment effect"]):
        return "review", "marketing_review"
    return "safe", "sales_internal"


def _confidence(source_count: int, claim_count: int, unsupported_count: int) -> str:
    if source_count >= 8 and claim_count >= 4 and unsupported_count == 0:
        return "high"
    if source_count >= 3 and claim_count >= 2:
        return "medium"
    return "low"


def _source_type(kind: str) -> str:
    normalized = str(kind or "").strip().lower().replace(" ", "_")
    return normalized or "source"


def _canonical_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for source in sources:
        output.append({
            "source_id": source.get("source_id"),
            "type": _source_type(source.get("kind")),
            "source_name": source.get("source"),
            "identifier": source.get("id"),
            "title": source.get("title"),
            "url": source.get("url"),
            "locator": source.get("locator"),
            "quote": source.get("quote"),
            "status": source.get("status"),
            "conditions": source.get("conditions", []),
            "pmid": source.get("pmid"),
            "pmcid": source.get("pmcid"),
            "doi": source.get("doi"),
            "year": source.get("year"),
            "retrieved_at": _utc_now(),
        })
    return output


def _canonical_claims(prediction: str, evidence_ledger: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    supported_claim_text = {claim["claim"] for claim in evidence_ledger.get("claims", [])}
    claims = []
    for index, claim in enumerate(evidence_ledger.get("claims", []), 1):
        risk, usage = _risk_for_claim(claim["claim"], True)
        claims.append({
            "claim_id": f"C{index}",
            "claim": claim["claim"],
            "source_ids": claim.get("source_ids", []),
            "risk_level": risk,
            "usage": usage,
        })

    unsupported = []
    for claim in _claim_lines(prediction):
        if claim in supported_claim_text:
            continue
        risk, usage = _risk_for_claim(claim, False)
        unsupported.append({
            "claim": claim,
            "risk_level": risk,
            "usage": usage,
            "reason": "No matching source identifier was found in the generated claim.",
        })
    return claims, unsupported[:20]


def _summary_from_prediction(prediction: str, max_chars: int = 1200) -> str:
    text = _plain_text(prediction)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _section_from_prediction(prediction: str, headings: List[str], max_chars: int = 900) -> str:
    lines = str(prediction or "").splitlines()
    active = False
    collected = []
    heading_patterns = [heading.lower() for heading in headings]
    for line in lines:
        normalized = _plain_text(line).lower().strip(":")
        is_heading = len(normalized) <= 80 and any(pattern in normalized for pattern in heading_patterns)
        if is_heading:
            active = True
            continue
        if active and re.match(r"^\s*(#{1,6}\s+|\*\*)?[A-Z][A-Za-z /&-]{2,60}(:|\*\*)?\s*$", line):
            break
        if active:
            text = _plain_text(line)
            if text:
                collected.append(text)
    output = " ".join(collected).strip()
    if len(output) > max_chars:
        output = output[:max_chars].rstrip() + "..."
    return output


def _recommended_steps_from_prediction(prediction: str) -> List[str]:
    section = _section_from_prediction(prediction, ["recommended", "next steps", "actions"], max_chars=1200)
    if not section:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", section)
    return [piece for piece in pieces if len(piece) >= 15][:8]


def _entities_from_sources(payload: Dict[str, Any], canonical_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    companies = []
    if payload.get("company"):
        companies.append({
            "name": payload.get("company"),
            "domain": payload.get("domain") or "",
            "aliases": payload.get("company_aliases", []),
        })
    trials = []
    products = []
    therapeutic_areas = []
    if payload.get("therapeutic_area"):
        therapeutic_areas.append(payload.get("therapeutic_area"))
    for source in canonical_sources:
        if source.get("type") == "clinical_trial":
            trials.append({
                "nct_id": source.get("identifier"),
                "title": source.get("title"),
                "status": source.get("status"),
                "conditions": source.get("conditions", []),
                "url": source.get("url"),
            })
            for condition in source.get("conditions", []) or []:
                if condition and condition not in therapeutic_areas:
                    therapeutic_areas.append(condition)
    return {
        "companies": companies,
        "therapeutic_areas": therapeutic_areas[:20],
        "trials": trials,
        "products": products,
        "sources": [{"source_id": source.get("source_id"), "identifier": source.get("identifier")} for source in canonical_sources],
    }


def _mode_instruction(mode: str, payload: Dict[str, Any]) -> str:
    mode = (mode or "content_brief").strip().lower()
    company = payload.get("company", "")
    therapeutic_area = payload.get("therapeutic_area", "")
    domain = payload.get("domain", "")
    if mode == "account_intel":
        return (
            f"Mode: account_intel. Research {company or 'the target account'} {f'({domain})' if domain else ''}. "
            "Identify commercial fit, active or relevant studies, precision medicine/recruitment signals, likely Sano angle, "
            "recommended actions, and risks."
        )
    if mode == "trial_scan":
        return (
            f"Mode: trial_scan. Scan trials and literature for {therapeutic_area or payload.get('query', 'the requested area')}. "
            "Prioritize NCT IDs, sponsor/status/phase, recruitment complexity, endpoints, eligibility, geography, and Sano-relevant needs."
        )
    if mode == "case_study_builder":
        return (
            "Mode: case_study_builder. Build a source-backed case-study evidence pack: problem, audience pain, proof points, "
            "candidate story arc, quotes, missing evidence, and claims needing review."
        )
    if mode == "gong_call_followup":
        return (
            "Mode: gong_call_followup. Use the call context to produce follow-up intelligence, pain points, account context, "
            "evidence-backed talking points, risks, and recommended next steps."
        )
    if mode == "competitor_scan":
        return (
            "Mode: competitor_scan. Identify competitor positioning, active claims, evidence quality, target audiences, "
            "differentiation gaps, and Sano-relevant positioning opportunities."
        )
    if mode == "disease_area_landscape":
        return (
            f"Mode: disease_area_landscape. Build a source-backed landscape for {therapeutic_area or payload.get('query', 'the disease area')}: "
            "trial activity, patient/recruitment challenge, market education angles, advocacy context, and commercial opportunities."
        )
    if mode == "quote_bank":
        return (
            "Mode: quote_bank. Prioritize high-quality short quotes with source metadata, grouped by theme and claim risk."
        )
    return (
        "Mode: content_brief. Produce a source-backed commercial/marketing research brief with evidence-backed claims, "
        "content angles, recommended next steps, and claims that need review."
    )


def _research_question(payload: Dict[str, Any]) -> str:
    pieces = [
        _mode_instruction(str(payload.get("mode", "content_brief")), payload),
        f"User request: {payload.get('query') or payload.get('question') or payload.get('brief') or ''}",
    ]
    if payload.get("call_context"):
        pieces.append(f"Gong/call context:\n{payload['call_context']}")
    requested_outputs = payload.get("requested_outputs") or []
    if requested_outputs:
        pieces.append(f"Requested output sections: {', '.join(str(item) for item in requested_outputs)}")
    pieces.append(
        "Return canonical GTM intelligence: executive summary, commercial angle, why now, recommended next steps, "
        "evidence-backed claims, risks, weak/unsupported claims, and source-backed marketing angles. "
        "Use concrete source identifiers in citations."
    )
    return "\n\n".join(piece for piece in pieces if piece)


def _canonical_research_result(job_id: str, payload: Dict[str, Any], agent_result: Dict[str, Any]) -> Dict[str, Any]:
    prediction = agent_result.get("prediction", "")
    evidence_ledger = agent_result.get("evidence_ledger", {})
    sources = _canonical_sources(evidence_ledger.get("sources") or agent_result.get("sources", []))
    claims, unsupported = _canonical_claims(prediction, evidence_ledger)
    risk_levels = sorted(set(claim["risk_level"] for claim in claims) | set(item["risk_level"] for item in unsupported))
    confidence = _confidence(len(sources), len(claims), len(unsupported))
    fit_score = min(100, 35 + len(sources) * 5 + len(claims) * 3 - len(unsupported) * 4)
    fit_score = max(0, fit_score)
    created_at = _utc_now()
    recommended_steps = _recommended_steps_from_prediction(prediction) or [
        "Review cited claims before commercial use.",
        "Route restricted or review claims through medical/legal approval.",
        "Use the evidence ledger to create destination-specific HubSpot, Gong, or Slack outputs in n8n.",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "mode": payload.get("mode", "content_brief"),
        "review_status": "draft",
        "confidence": confidence,
        "fit_score": fit_score,
        "brief": {
            "executive_summary": _summary_from_prediction(prediction),
            "commercial_angle": _section_from_prediction(prediction, ["commercial angle", "gtm angle", "positioning", "opportunity"]),
            "why_now": _section_from_prediction(prediction, ["why now", "urgency", "market timing"]),
            "recommended_next_steps": recommended_steps,
        },
        "entities": _entities_from_sources(payload, sources),
        "claims": claims,
        "sources": sources,
        "risks": [{"risk_level": level, "description": f"Claims labelled {level} require the corresponding usage handling."} for level in risk_levels],
        "unsupported_or_weak_claims": unsupported,
        "requires_medical_review": any(claim["risk_level"] in {"review", "restricted"} for claim in claims),
        "requires_legal_review": any(claim["risk_level"] in {"restricted"} for claim in claims),
        "recommended_channel": "sales_internal" if confidence != "low" else "research_review",
        "audit": {
            "created_at": created_at,
            "retrieved_at": created_at,
            "raw_transcript_path": agent_result.get("output_path"),
            "tool_names": agent_result.get("tool_names", []),
            "source_count": len(sources),
            "claim_count": len(claims),
            "unsupported_claim_count": len(unsupported),
        },
    }


def _json_tool_payload(raw: str, tool_name: str) -> Dict[str, Any]:
    if not raw or raw.lstrip().startswith(f"[{tool_name}]"):
        raise RuntimeError(raw or f"{tool_name} returned empty output")
    return json.loads(raw)


def run_tool_check(query: str, max_results: int) -> Dict[str, Any]:
    from tool_clinical import ClinicalTrialsSearch, LiteratureSearch

    trials = _json_tool_payload(
        ClinicalTrialsSearch().call({"query": query, "max_results": max_results}),
        "clinical_trials_search",
    )
    literature = _json_tool_payload(
        LiteratureSearch().call({"query": query, "max_results": max_results}),
        "literature_search",
    )
    return {
        "clinical_trials": {
            "source": trials.get("source"),
            "returned": trials.get("returned"),
            "studies": trials.get("studies", []),
        },
        "literature": {
            "source": literature.get("source"),
            "returned": literature.get("returned"),
            "results": literature.get("results", []),
        },
    }


def _depth_instruction(search_depth: str, source_target: int) -> str:
    depth = (search_depth or "standard").strip().lower()
    if depth == "quick":
        return (
            "Research depth: quick scan. Use 1-2 focused query variants and retrieve about "
            f"{source_target} high-signal sources. Prefer official structured sources first. "
            f"When a tool has max_results, use up to {max(2, min(source_target, 10))}."
        )
    if depth == "deep":
        return (
            "Research depth: deep evidence pack. Use multiple query variants across official, "
            "literature, regulatory, company, and reputable web sources where relevant. Retrieve "
            f"about {source_target} sources before synthesizing, unless the available evidence is sparse. "
            f"Build a broad evidence ledger before drafting the answer. When a tool has max_results, "
            f"use up to {max(10, min(source_target, 75))}."
        )
    return (
        "Research depth: standard brief. Use 2-4 query variants and retrieve about "
        f"{source_target} credible sources. Prefer official structured sources and biomedical literature, "
        "then use reputable web pages to fill marketing context. "
        f"When a tool has max_results, use up to {max(5, min(source_target, 25))}."
    )


def run_agent_query(
    question: str,
    max_calls: int,
    max_tries: int,
    timeout: float,
    max_output_tokens: int,
    search_depth: str,
    source_target: int,
    progress_callback=None,
) -> Dict[str, Any]:
    os.environ["MAX_LLM_CALL_PER_RUN"] = str(max_calls)
    os.environ["LLM_MAX_TRIES"] = str(max_tries)
    os.environ["LLM_REQUEST_TIMEOUT"] = str(timeout)
    os.environ["LLM_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)

    from react_agent import MultiTurnReactAgent

    model = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("MODEL_PATH")
        or "ui"
    )
    agent = MultiTurnReactAgent(
        llm={
            "model": model,
            "generate_cfg": {
                "max_input_tokens": 320000,
                "max_retries": max_tries,
                "temperature": 0,
                "top_p": 1,
                "presence_penalty": 0,
            },
            "model_type": "qwen_dashscope",
        },
        function_list=[
            "search",
            "visit",
            "google_scholar",
            "PythonInterpreter",
            "clinical_trials_search",
            "literature_search",
            "extract_webpage",
            "parse_pdf_grobid",
        ],
    )
    guided_question = (
        f"{_depth_instruction(search_depth, source_target)}\n\n"
        "Do not answer from memory. You must call source-gathering tools before producing the final answer. "
        "For clinical, biomedical, pharmaceutical, trial, or disease topics, use clinical_trials_search and/or "
        "literature_search. For marketing context, competitor pages, product pages, or case-study discovery, "
        "use search plus visit or extract_webpage when web credentials/backends are available.\n\n"
        "For each important claim, include citation markers using concrete source identifiers such as "
        "[NCT01234567], [PMID: 12345678], [PMCID: PMC123456], or [DOI: 10.xxxx/xxxx]. "
        "If you create [S1]-style source IDs, define them in a Sources section. Include URLs/identifiers "
        "and short direct quotes under 25 words.\n\n"
        f"User request:\n{question}"
    )
    result = agent._run(
        {"item": {"question": guided_question, "answer": ""}, "planning_port": 6001},
        model,
        progress_callback=progress_callback,
        require_source_tools=True,
    )
    output_dir = os.path.join(ROOT_DIR, "outputs", "ui")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"ui_agent_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    sources = _extract_sources(result.get("messages", []))
    evidence_ledger = _build_evidence_ledger(result.get("prediction", ""), sources)
    return {
        "termination": result.get("termination"),
        "prediction": result.get("prediction"),
        "tool_names": _extract_tool_names(result.get("messages", [])),
        "sources": sources,
        "evidence_ledger": evidence_ledger,
        "message_count": len(result.get("messages", [])),
        "output_path": output_path,
        "raw": result,
    }


def run_research_query(payload: Dict[str, Any], job_id: str, progress_callback=None) -> Dict[str, Any]:
    question = _research_question(payload)
    agent_result = run_agent_query(
        question=question,
        max_calls=int(payload.get("max_calls", payload.get("agent_max_calls", 10))),
        max_tries=int(payload.get("max_tries", 2)),
        timeout=float(payload.get("timeout", 180)),
        max_output_tokens=int(payload.get("max_output_tokens", 4096)),
        search_depth=str(payload.get("search_depth", "standard")),
        source_target=int(payload.get("target_sources", payload.get("source_target", 12))),
        progress_callback=progress_callback,
    )
    canonical = _canonical_research_result(job_id, payload, agent_result)
    output_dir = os.path.join(ROOT_DIR, "outputs", "research")
    os.makedirs(output_dir, exist_ok=True)
    canonical_path = os.path.join(output_dir, f"research_{job_id}.json")
    review_base_url = os.getenv("RESEARCH_REVIEW_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
    canonical["audit"]["canonical_output_path"] = canonical_path
    canonical["audit"]["review_url"] = f"{review_base_url}/api/research/{job_id}"
    with open(canonical_path, "w", encoding="utf-8") as handle:
        json.dump(canonical, handle, ensure_ascii=False, indent=2)
    return {
        "canonical": canonical,
        "agent_result": agent_result,
    }


class LiveJobWriter(io.TextIOBase):
    def __init__(self, job_id: str, buffer: io.StringIO):
        self.job_id = job_id
        self.buffer = buffer

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        if not isinstance(value, str):
            value = str(value)
        self.buffer.write(value)
        with JOBS_LOCK:
            job = JOBS.get(self.job_id)
            if job is not None:
                job["logs"] = self.buffer.getvalue()[-20000:]
                job["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        return len(value)

    def flush(self) -> None:
        return None


def _record_event(job_id: str, event: Dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **event,
    }
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.setdefault("events", []).append(payload)
        job["events"] = job["events"][-200:]
        job["updated_at"] = payload["timestamp"]


def _start_job(kind: str, payload: Dict[str, Any]) -> str:
    idempotency_key = str(payload.get("idempotency_key", "")).strip()
    if idempotency_key:
        with JOBS_LOCK:
            existing_job_id = IDEMPOTENCY_KEYS.get(idempotency_key)
            if existing_job_id and existing_job_id in JOBS:
                return existing_job_id

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        if idempotency_key:
            IDEMPOTENCY_KEYS[idempotency_key] = job_id
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "idempotency_key": idempotency_key or None,
            "status": "queued",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "logs": "",
            "events": [],
            "result": None,
            "error": None,
        }

    def worker() -> None:
        buffer = io.StringIO()
        writer = LiveJobWriter(job_id, buffer)
        try:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "running"
                JOBS[job_id]["updated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            _record_event(job_id, {"event": "job_started", "kind": kind})
            start = time.time()
            with JOB_RUN_LOCK:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    if kind == "research":
                        result = run_research_query(
                            payload=payload,
                            job_id=job_id,
                            progress_callback=lambda event: _record_event(job_id, event),
                        )
                    elif kind == "agent":
                        result = run_agent_query(
                            question=payload["question"],
                            max_calls=int(payload.get("max_calls", 8)),
                            max_tries=int(payload.get("max_tries", 2)),
                            timeout=float(payload.get("timeout", 120)),
                            max_output_tokens=int(payload.get("max_output_tokens", 4096)),
                            search_depth=str(payload.get("search_depth", "standard")),
                            source_target=int(payload.get("source_target", 12)),
                            progress_callback=lambda event: _record_event(job_id, event),
                        )
                    elif kind == "tools":
                        _record_event(job_id, {"event": "tool_check_started", "query": payload["query"]})
                        result = run_tool_check(
                            query=payload["query"],
                            max_results=int(payload.get("max_results", 5)),
                        )
                        _record_event(job_id, {"event": "tool_check_finished"})
                    else:
                        raise RuntimeError(f"Unknown job kind: {kind}")
            result["elapsed_seconds"] = round(time.time() - start, 2)
            with JOBS_LOCK:
                JOBS[job_id].update({
                    "status": "completed",
                    "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "logs": buffer.getvalue()[-20000:],
                    "result": result,
                })
            _record_event(job_id, {"event": "job_finished", "elapsed_seconds": result["elapsed_seconds"]})
        except Exception as exc:
            with JOBS_LOCK:
                JOBS[job_id].update({
                    "status": "failed",
                    "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "logs": buffer.getvalue()[-20000:],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
            _record_event(job_id, {"event": "job_failed", "error": f"{type(exc).__name__}: {exc}"})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return job_id


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Clinical Research Agent</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #eef2f6;
      --line: #d8dee7;
      --text: #17202a;
      --muted: #5b6675;
      --accent: #156f73;
      --accent-2: #8a5a00;
      --danger: #a33b35;
      --ok: #267346;
      --focus: #295fbd;
      --mono: Consolas, "SFMono-Regular", Menlo, monospace;
      --sans: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background: var(--bg);
      min-height: 100vh;
    }
    .app {
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      background: var(--surface);
      border-right: 1px solid var(--line);
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    main {
      min-width: 0;
      padding: 20px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 16px;
    }
    h1 {
      font-size: 20px;
      line-height: 1.2;
      margin: 0 0 4px;
      font-weight: 650;
      letter-spacing: 0;
    }
    h2 {
      font-size: 14px;
      line-height: 1.2;
      margin: 0;
      font-weight: 650;
      letter-spacing: 0;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 650;
      color: var(--muted);
      margin-bottom: 6px;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 9px 10px;
      font: inherit;
      font-size: 14px;
      outline: none;
    }
    textarea {
      min-height: 170px;
      resize: vertical;
      line-height: 1.45;
    }
    textarea:focus, input:focus, select:focus {
      border-color: var(--focus);
      box-shadow: 0 0 0 3px rgba(41, 95, 189, .12);
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .controls {
      display: grid;
      gap: 12px;
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    button {
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 9px 11px;
      font: inherit;
      font-size: 14px;
      font-weight: 650;
      cursor: pointer;
      min-height: 38px;
    }
    button.secondary {
      background: #fff;
      color: var(--accent);
    }
    button:disabled {
      opacity: .55;
      cursor: not-allowed;
    }
    .status-grid {
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .status-grid div {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--surface-2);
      padding-bottom: 6px;
    }
    .status-grid strong { color: var(--text); font-weight: 650; text-align: right; }
    .toolbar {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
      background: #fff;
    }
    .pill.ok { color: var(--ok); border-color: rgba(38,115,70,.35); }
    .pill.fail { color: var(--danger); border-color: rgba(163,59,53,.35); }
    .results {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
    }
    section {
      min-width: 0;
      min-height: 0;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .section-head {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }
    .section-body {
      padding: 14px;
      min-height: 0;
      overflow: auto;
    }
    [hidden] {
      display: none !important;
    }
    .answer {
      white-space: pre-wrap;
      line-height: 1.55;
      font-size: 14px;
    }
    .answer h3 {
      margin: 14px 0 7px;
      font-size: 15px;
      line-height: 1.25;
    }
    .answer p { margin: 0 0 10px; }
    .answer ul {
      margin: 0 0 10px 18px;
      padding: 0;
    }
    .answer li { margin: 4px 0; }
    .answer strong { font-weight: 700; }
    .empty {
      color: var(--muted);
      font-size: 14px;
    }
    .list {
      display: grid;
      gap: 10px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    .item-title {
      font-size: 13px;
      font-weight: 650;
      line-height: 1.35;
      margin-bottom: 6px;
    }
    .item-meta {
      font-size: 12px;
      line-height: 1.4;
      color: var(--muted);
    }
    .quote {
      margin-top: 8px;
      padding-left: 10px;
      border-left: 3px solid var(--accent);
      color: #2e3a46;
      font-size: 13px;
      line-height: 1.45;
    }
    .source-link {
      color: var(--focus);
      text-decoration: none;
      word-break: break-word;
    }
    .source-link:hover { text-decoration: underline; }
    .claim-list {
      display: grid;
      gap: 8px;
      margin-bottom: 12px;
    }
    .claim {
      padding: 8px;
      border: 1px solid var(--surface-2);
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.45;
      background: #fbfcfd;
    }
    .citation {
      display: inline-flex;
      align-items: center;
      min-height: 20px;
      padding: 1px 6px;
      margin: 0 1px;
      border-radius: 999px;
      background: #e6f2f3;
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
      font-size: 12px;
    }
    .source-highlight {
      outline: 3px solid rgba(41, 95, 189, .18);
      border-color: var(--focus);
    }
    .event-list {
      display: grid;
      gap: 8px;
    }
    .event {
      border-left: 3px solid var(--line);
      padding: 4px 0 4px 9px;
      font-size: 13px;
      line-height: 1.4;
    }
    .event strong { font-weight: 650; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
      color: #25313f;
    }
    .tabs {
      display: inline-grid;
      grid-auto-flow: column;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #fff;
    }
    .tab {
      border: 0;
      border-radius: 0;
      min-height: 30px;
      padding: 5px 10px;
      background: #fff;
      color: var(--muted);
      font-size: 13px;
    }
    .tab.active {
      background: var(--surface-2);
      color: var(--text);
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .results { grid-template-columns: 1fr; }
      textarea { min-height: 140px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div>
        <h1>Clinical Research Agent</h1>
        <div class="status-grid" id="configStatus"></div>
      </div>
      <div class="controls">
        <div>
          <label for="question">Research Question</label>
          <textarea id="question">Use official clinical and biomedical sources to find current or recent gene therapy evidence for spinal muscular atrophy. Summarize NCT IDs, recruitment status, intervention or study type, endpoint patterns, and cite PMIDs/DOIs where available.</textarea>
        </div>
        <div class="row">
          <div>
            <label for="maxCalls">Max Calls</label>
            <input id="maxCalls" type="number" min="1" max="30" value="8">
          </div>
          <div>
            <label for="timeout">Timeout</label>
            <input id="timeout" type="number" min="30" max="900" value="120">
          </div>
        </div>
        <div class="row">
          <div>
            <label for="searchDepth">Search Depth</label>
            <select id="searchDepth">
              <option value="standard" selected>Standard brief</option>
              <option value="quick">Quick scan</option>
              <option value="deep">Deep evidence pack</option>
            </select>
          </div>
          <div>
            <label for="sourceTarget">Target Sources</label>
            <input id="sourceTarget" type="number" min="2" max="75" value="12">
          </div>
        </div>
        <div class="button-row">
          <button id="runAgent">Run Agent</button>
          <button id="runN8n" class="secondary">Run n8n Output</button>
          <button id="runTools" class="secondary">Check Tools</button>
        </div>
      </div>
      <div class="controls">
        <div>
          <label for="toolQuery">Tool Query</label>
          <input id="toolQuery" value="gene therapy rare disease">
        </div>
        <div class="row">
          <div>
            <label for="maxResults">Max Results</label>
            <input id="maxResults" type="number" min="1" max="25" value="5">
          </div>
          <div>
            <label for="maxTokens">Max Tokens</label>
            <input id="maxTokens" type="number" min="512" max="20000" value="4096">
          </div>
        </div>
      </div>
    </aside>
    <main>
      <div class="toolbar">
        <div>
          <span class="pill" id="jobState">idle</span>
          <span class="pill" id="jobMeta">no job</span>
        </div>
        <div class="tabs">
          <button class="tab active" data-view="answer">Answer</button>
          <button class="tab" data-view="sources">Sources</button>
          <button class="tab" data-view="n8n">n8n JSON</button>
          <button class="tab" data-view="progress">Progress</button>
          <button class="tab" data-view="logs">Debug Logs</button>
          <button class="tab" data-view="raw">Debug Raw</button>
        </div>
      </div>
      <div class="results">
        <section>
          <div class="section-head">
            <h2 id="mainTitle">Answer</h2>
            <span class="pill" id="toolsUsed">sources: 0</span>
          </div>
          <div class="section-body">
            <div id="answerView" class="answer empty">Final answer will appear here.</div>
            <div id="sourcesView" class="list" hidden></div>
            <pre id="n8nView" hidden>Final n8n JSON will appear here.</pre>
            <div id="progressView" class="event-list" hidden></div>
            <pre id="logsView" hidden></pre>
            <pre id="rawView" hidden></pre>
          </div>
        </section>
        <section>
          <div class="section-head">
            <h2>Recent Jobs</h2>
            <button class="tab" id="refreshJobs">Refresh</button>
          </div>
          <div class="section-body">
            <div class="list" id="jobsList"></div>
          </div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const state = { activeJob: null, activeView: 'answer', pollTimer: null, lastResult: null, outputViewOnComplete: null, renderedJobId: null };
    const $ = (id) => document.getElementById(id);

    function setBusy(isBusy) {
      $('runAgent').disabled = isBusy;
      $('runN8n').disabled = isBusy;
      $('runTools').disabled = isBusy;
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...options
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      return data;
    }

    async function loadConfig() {
      const config = await api('/api/config');
      $('configStatus').innerHTML = `
        <div><span>Provider</span><strong>${config.provider}</strong></div>
        <div><span>LLM</span><strong>${config.ready ? 'ready' : 'missing'}</strong></div>
        <div><span>Visit</span><strong>${config.visit_backend}</strong></div>
      `;
    }

    function setView(view) {
      state.activeView = view;
      for (const button of document.querySelectorAll('.tab[data-view]')) {
        button.classList.toggle('active', button.dataset.view === view);
      }
      $('answerView').hidden = view !== 'answer';
      $('sourcesView').hidden = view !== 'sources';
      $('n8nView').hidden = view !== 'n8n';
      $('progressView').hidden = view !== 'progress';
      $('logsView').hidden = view !== 'logs';
      $('rawView').hidden = view !== 'raw';
      const titles = {
        answer: 'Answer',
        sources: 'Sources',
        n8n: 'n8n JSON',
        progress: 'Progress',
        logs: 'Debug Logs',
        raw: 'Debug Raw'
      };
      $('mainTitle').textContent = titles[view] || view[0].toUpperCase() + view.slice(1);
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[char]));
    }

    function inlineMarkdown(value) {
      let html = escapeHtml(value);
      html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      return html;
    }

    function sourceByAlias(ledger, token) {
      if (!ledger || !ledger.sources) return null;
      const normalized = String(token || '').toLowerCase().replace(/\s+/g, '').replace(/[:]/g, '');
      for (const source of ledger.sources) {
        const aliases = [
          ...(source.aliases || []),
          source.id,
          source.identifier,
          source.pmid ? `PMID:${source.pmid}` : '',
          source.pmid ? `PMID ${source.pmid}` : '',
          source.pmcid ? `PMCID:${source.pmcid}` : '',
          source.pmcid ? `PMCID ${source.pmcid}` : '',
          source.doi ? `DOI:${source.doi}` : ''
        ].filter(Boolean);
        for (const alias of aliases) {
          const aliasNorm = String(alias).toLowerCase().replace(/\s+/g, '').replace(/[:]/g, '');
          if (aliasNorm === normalized) return source;
        }
      }
      return null;
    }

    function linkCitations(html, ledger) {
      if (!ledger || !ledger.sources) return html;
      html = html.replace(/\[(S\d+)\]/g, (match, sid) => {
        const source = ledger.sources.find(item => item.source_id === sid);
        return source ? `<a class="citation" href="#source-${sid}" data-source-id="${sid}">[${sid}]</a>` : match;
      });
      html = html.replace(/\bNCT\d{8}\b/g, (token) => {
        const source = sourceByAlias(ledger, token);
        return source ? `<a class="citation" href="#source-${source.source_id}" data-source-id="${source.source_id}">${token}</a>` : token;
      });
      html = html.replace(/\bPMCID:?\s*PMC\d+\b/gi, (token) => {
        const source = sourceByAlias(ledger, token);
        return source ? `<a class="citation" href="#source-${source.source_id}" data-source-id="${source.source_id}">${token}</a>` : token;
      });
      html = html.replace(/\bPMID:?\s*\d+\b/gi, (token) => {
        const source = sourceByAlias(ledger, token);
        return source ? `<a class="citation" href="#source-${source.source_id}" data-source-id="${source.source_id}">${token}</a>` : token;
      });
      return html;
    }

    function renderMarkdown(value, ledger) {
      const lines = String(value || '').replace(/\r\n/g, '\n').trim().split('\n');
      let html = '';
      let inList = false;
      const closeList = () => {
        if (inList) {
          html += '</ul>';
          inList = false;
        }
      };
      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
          closeList();
          continue;
        }
        const heading = line.match(/^#{1,6}\s+(.*)$/);
        if (heading) {
          closeList();
          html += `<h3>${linkCitations(inlineMarkdown(heading[1]), ledger)}</h3>`;
          continue;
        }
        const bullet = line.match(/^[-*]\s+(.*)$/);
        if (bullet) {
          if (!inList) {
            html += '<ul>';
            inList = true;
          }
          html += `<li>${linkCitations(inlineMarkdown(bullet[1]), ledger)}</li>`;
          continue;
        }
        closeList();
        html += `<p>${linkCitations(inlineMarkdown(line), ledger)}</p>`;
      }
      closeList();
      return html || '<p>No answer returned.</p>';
    }

    function sourceCardsFromToolResult(payload) {
      const sources = [];
      const clinical = payload?.clinical_trials?.studies || [];
      const literature = payload?.literature?.results || [];
      for (const study of clinical.slice(0, 8)) {
        const outcome = (study.primary_outcomes || [])[0] || {};
        sources.push({
          kind: 'Clinical trial',
          source: 'ClinicalTrials.gov API v2',
          id: study.nct_id,
          title: study.brief_title,
          url: study.url,
          status: study.overall_status,
          conditions: study.conditions || [],
          quote: outcome.description || outcome.measure || study.brief_summary || ''
        });
      }
      for (const paper of literature.slice(0, 8)) {
        sources.push({
          kind: 'Literature',
          source: 'Europe PMC REST API',
          id: paper.pmid || paper.pmcid || paper.doi,
          title: paper.title,
          url: paper.url,
          pmid: paper.pmid,
          pmcid: paper.pmcid,
          doi: paper.doi,
          year: paper.year,
          quote: paper.abstract || ''
        });
      }
      return sources;
    }

    function truncateQuote(value) {
      const words = String(value || '').replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
      return words.slice(0, 28).join(' ') + (words.length > 28 ? '...' : '');
    }

    function renderSources(result) {
      const sourcesView = $('sourcesView');
      sourcesView.innerHTML = '';
      const payload = result?.result || result || {};
      const ledger = payload.evidence_ledger || {};
      const claimsList = ledger.claims || payload.claims || [];
      const sources = ledger.sources || payload.sources || sourceCardsFromToolResult(payload);
      sources.forEach((source, index) => {
        if (!source.source_id) source.source_id = `S${index + 1}`;
      });
      if (claimsList.length) {
        const claims = document.createElement('div');
        claims.className = 'claim-list';
        claims.innerHTML = claimsList.slice(0, 20).map(claim => {
          const refs = (claim.source_ids || []).map(sid => `<a class="citation" href="#source-${sid}" data-source-id="${sid}">[${sid}]</a>`).join(' ');
          return `<div class="claim">${escapeHtml(claim.claim)} ${refs}</div>`;
        }).join('');
        sourcesView.appendChild(claims);
      }
      for (const source of sources) {
        const item = document.createElement('div');
        item.className = 'item';
        item.id = `source-${source.source_id || source.id || source.identifier || ''}`;
        const meta = [
          source.source_id ? `Ledger ID: ${source.source_id}` : '',
          source.source || source.source_name,
          source.id || source.identifier ? `ID: ${source.id || source.identifier}` : '',
          source.locator ? `Locator: ${source.locator}` : '',
          source.status ? `Status: ${source.status}` : '',
          source.year ? `Year: ${source.year}` : '',
          source.pmid ? `PMID: ${source.pmid}` : '',
          source.pmcid ? `PMCID: ${source.pmcid}` : '',
          source.doi ? `DOI: ${source.doi}` : '',
          source.conditions?.length ? `Conditions: ${source.conditions.join(', ')}` : ''
        ].filter(Boolean).map(escapeHtml).join('<br>');
        const quote = truncateQuote(source.quote);
        item.innerHTML = `
          <div class="item-title">${escapeHtml(source.kind || source.type || 'Source')} - ${escapeHtml(source.title || source.id || source.identifier || '')}</div>
          <div class="item-meta">${meta}</div>
          ${source.url ? `<div class="item-meta"><a class="source-link" href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.url)}</a></div>` : ''}
          ${quote ? `<div class="quote">“${escapeHtml(quote)}”</div>` : ''}
        `;
        sourcesView.appendChild(item);
      }
      if (!sourcesView.childElementCount) {
        sourcesView.innerHTML = '<div class="empty">No sources for this result.</div>';
      }
      return sources.length;
    }

    function eventLabel(event) {
      const name = event.event || 'event';
      if (name === 'round_started') return `Round ${event.round} started`;
      if (name === 'model_response') return `Model responded: ${event.tool_calls || 0} tool call(s), ${event.chars || 0} chars`;
      if (name === 'tool_started') return `Calling ${event.tool}`;
      if (name === 'tool_finished') return `${event.tool} returned ${event.chars || 0} chars`;
      if (name === 'agent_finished') return `Agent finished: ${event.termination}`;
      if (name === 'job_finished') return `Job finished in ${event.elapsed_seconds}s`;
      return name.replace(/_/g, ' ');
    }

    function renderProgress(job) {
      const view = $('progressView');
      const events = job.events || [];
      if (!events.length) {
        view.innerHTML = '<div class="empty">No progress events yet.</div>';
        return;
      }
      view.innerHTML = events.slice(-80).map(event => `
        <div class="event">
          <strong>${escapeHtml(eventLabel(event))}</strong>
          <div class="item-meta">${escapeHtml(event.timestamp || '')}</div>
          ${event.arguments ? `<pre>${escapeHtml(JSON.stringify(event.arguments, null, 2))}</pre>` : ''}
        </div>
      `).join('');
      view.scrollTop = view.scrollHeight;
    }

    function canonicalFromJob(job) {
      return job?.result?.canonical || null;
    }

    function renderCanonicalBrief(canonical) {
      if (!canonical) return 'No n8n output returned.';
      const brief = canonical.brief || {};
      const lines = [];
      if (brief.executive_summary) lines.push(`### Executive summary\n${brief.executive_summary}`);
      if (brief.commercial_angle) lines.push(`### Commercial angle\n${brief.commercial_angle}`);
      if (brief.why_now) lines.push(`### Why now\n${brief.why_now}`);
      if ((brief.recommended_next_steps || []).length) {
        lines.push(`### Recommended next steps\n${brief.recommended_next_steps.map(item => `- ${item}`).join('\n')}`);
      }
      if (!lines.length) {
        lines.push(`### Result\nSchema: ${canonical.schema_version || 'unknown'}\nConfidence: ${canonical.confidence || 'unknown'}\nSources: ${(canonical.sources || []).length}\nClaims: ${(canonical.claims || []).length}`);
      }
      return lines.join('\n\n');
    }

    function resetFinalOutputPanels(kind) {
      $('answerView').className = 'answer empty';
      $('answerView').textContent = 'Final answer will appear here when the job completes.';
      $('sourcesView').innerHTML = '<div class="empty">Final sources will appear here when the job completes.</div>';
      $('n8nView').textContent = kind === 'research'
        ? 'Final n8n JSON will appear here when the research job completes.'
        : 'This job does not produce n8n JSON. Use "Run n8n Output" for the API payload.';
      $('logsView').textContent = '';
      $('rawView').textContent = '';
      $('toolsUsed').textContent = 'sources: 0';
      renderProgress({ events: [] });
    }

    function renderJob(job) {
      if (state.renderedJobId !== job.id) {
        state.renderedJobId = job.id;
        resetFinalOutputPanels(job.kind);
      }
      const statusClass = job.status === 'completed' ? 'ok' : (job.status === 'failed' ? 'fail' : '');
      $('jobState').textContent = job.status;
      $('jobState').className = `pill ${statusClass}`;
      $('jobMeta').textContent = `${job.kind} ${job.id.slice(0, 8)}`;
      renderProgress(job);
      if (job.status === 'queued' || job.status === 'running') {
        return;
      }
      $('logsView').textContent = job.logs || '';
      $('rawView').textContent = JSON.stringify(job, null, 2);
      const canonical = canonicalFromJob(job);
      if (job.result) {
        $('n8nView').textContent = canonical
          ? JSON.stringify(canonical, null, 2)
          : 'This job does not produce n8n JSON. Use "Run n8n Output" for the API payload.';
        state.lastResult = job.result;
        const displayResult = canonical || job.result;
        const sourceCount = renderSources(displayResult);
        const toolNames = canonical?.audit?.tool_names || job.result.tool_names || [];
        const tools = toolNames.join(', ') || 'none';
        $('toolsUsed').textContent = `tools: ${tools}; sources: ${sourceCount}`;
        if (canonical) {
          $('answerView').classList.remove('empty');
          $('answerView').innerHTML = renderMarkdown(renderCanonicalBrief(canonical), { sources: canonical.sources || [], claims: canonical.claims || [] });
        } else if (job.kind === 'agent') {
          $('answerView').classList.remove('empty');
          $('answerView').innerHTML = renderMarkdown(job.result.prediction || 'No answer returned.', job.result.evidence_ledger);
        } else {
          $('answerView').classList.remove('empty');
          $('answerView').innerHTML = renderMarkdown('Tool check completed. Open the Sources tab to inspect ClinicalTrials.gov and Europe PMC records.');
        }
      }
      if (job.error) {
        $('answerView').classList.add('empty');
        $('answerView').textContent = job.error;
        $('n8nView').textContent = JSON.stringify({ error: job.error, job_id: job.id, status: job.status }, null, 2);
      }
    }

    async function pollJob(jobId) {
      const job = await api(`/api/jobs/${jobId}`);
      renderJob(job);
      await loadJobs();
      if (job.status === 'queued' || job.status === 'running') {
        state.pollTimer = setTimeout(() => pollJob(jobId), 1200);
      } else {
        setBusy(false);
        if (state.outputViewOnComplete) {
          setView(state.outputViewOnComplete);
          state.outputViewOnComplete = null;
        }
      }
    }

    async function startAgent() {
      clearTimeout(state.pollTimer);
      setBusy(true);
      state.outputViewOnComplete = 'answer';
      resetFinalOutputPanels('agent');
      setView('progress');
      const data = await api('/api/agent', {
        method: 'POST',
        body: JSON.stringify({
          question: $('question').value,
          max_calls: Number($('maxCalls').value),
          max_tries: 2,
          timeout: Number($('timeout').value),
          max_output_tokens: Number($('maxTokens').value),
          search_depth: $('searchDepth').value,
          source_target: Number($('sourceTarget').value)
        })
      });
      state.activeJob = data.job_id;
      pollJob(data.job_id);
    }

    async function startN8nResearch() {
      clearTimeout(state.pollTimer);
      setBusy(true);
      state.outputViewOnComplete = 'n8n';
      resetFinalOutputPanels('research');
      setView('progress');
      const data = await api('/api/research-debug', {
        method: 'POST',
        body: JSON.stringify({
          mode: 'content_brief',
          query: $('question').value,
          search_depth: $('searchDepth').value,
          target_sources: Number($('sourceTarget').value),
          max_calls: Number($('maxCalls').value),
          max_tries: 2,
          timeout: Number($('timeout').value),
          max_output_tokens: Number($('maxTokens').value)
        })
      });
      state.activeJob = data.job_id;
      pollJob(data.job_id);
    }

    async function startTools() {
      clearTimeout(state.pollTimer);
      setBusy(true);
      state.outputViewOnComplete = 'sources';
      resetFinalOutputPanels('tools');
      setView('progress');
      const data = await api('/api/tools', {
        method: 'POST',
        body: JSON.stringify({
          query: $('toolQuery').value,
          max_results: Number($('maxResults').value)
        })
      });
      state.activeJob = data.job_id;
      pollJob(data.job_id);
    }

    async function loadJobs() {
      const data = await api('/api/jobs');
      const list = $('jobsList');
      list.innerHTML = '';
      for (const job of data.jobs) {
        const item = document.createElement('button');
        item.className = 'secondary';
        item.style.textAlign = 'left';
        item.innerHTML = `${job.status} · ${job.kind}<br><span style="font-weight:400;color:var(--muted)">${job.created_at}</span>`;
        item.onclick = () => { state.activeJob = job.id; pollJob(job.id); };
        list.appendChild(item);
      }
      if (!list.childElementCount) list.innerHTML = '<div class="empty">No jobs yet.</div>';
    }

    document.querySelectorAll('.tab[data-view]').forEach(button => {
      button.addEventListener('click', () => setView(button.dataset.view));
    });
    document.addEventListener('click', (event) => {
      const link = event.target.closest('a[data-source-id]');
      if (!link) return;
      event.preventDefault();
      const sourceId = link.dataset.sourceId;
      setView('sources');
      requestAnimationFrame(() => {
        document.querySelectorAll('.source-highlight').forEach(item => item.classList.remove('source-highlight'));
        const target = document.getElementById(`source-${sourceId}`);
        if (target) {
          target.classList.add('source-highlight');
          target.scrollIntoView({ block: 'center', behavior: 'smooth' });
        }
      });
    });
    $('runAgent').addEventListener('click', () => startAgent().catch(err => { setBusy(false); alert(err.message); }));
    $('runN8n').addEventListener('click', () => startN8nResearch().catch(err => { setBusy(false); alert(err.message); }));
    $('runTools').addEventListener('click', () => startTools().catch(err => { setBusy(false); alert(err.message); }));
    $('refreshJobs').addEventListener('click', () => loadJobs());
    $('searchDepth').addEventListener('change', () => {
      const depth = $('searchDepth').value;
      if (depth === 'quick') {
        $('sourceTarget').value = 6;
        $('maxCalls').value = 5;
      } else if (depth === 'deep') {
        $('sourceTarget').value = 30;
        $('maxCalls').value = 14;
      } else {
        $('sourceTarget').value = 12;
        $('maxCalls').value = 8;
      }
    });

    loadConfig().catch(() => {});
    loadJobs().catch(() => {});
  </script>
</body>
</html>
"""


class UIHandler(BaseHTTPRequestHandler):
    server_version = "ClinicalResearchUI/0.1"

    def _json_safe(self, value: Any, seen=None) -> Any:
        if seen is None:
            seen = set()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        value_id = id(value)
        if value_id in seen:
            return "[circular]"
        if isinstance(value, dict):
            seen.add(value_id)
            output = {str(key): self._json_safe(item, seen) for key, item in value.items()}
            seen.remove(value_id)
            return output
        if isinstance(value, (list, tuple, set)):
            seen.add(value_id)
            output = [self._json_safe(item, seen) for item in value]
            seen.remove(value_id)
            return output
        return str(value)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(self._json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _send_html(self) -> None:
        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        if _safe_bool_env("UI_DEBUG"):
            super().log_message(format, *args)

    def _authorized(self) -> bool:
        expected = os.getenv("RESEARCH_API_KEY", "").strip()
        if not expected:
            return True
        auth = self.headers.get("Authorization", "")
        api_key = self.headers.get("X-Research-API-Key", "")
        return auth == f"Bearer {expected}" or api_key == expected

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._send_json({"error": "unauthorized"}, status=401)
        return False

    def _require_debug_ui(self) -> bool:
        if _debug_ui_enabled():
            return True
        self._send_json({
            "error": "debug UI is disabled",
            "message": "Use POST /api/research for n8n and automation workflows.",
            "service": _service_info(),
        }, status=403)
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            if not _debug_ui_enabled():
                self._send_json(_service_info())
                return
            self._send_html()
            return
        if parsed.path == "/api/config":
            if not self._require_debug_ui():
                return
            self._send_json(_redacted_ready())
            return
        if parsed.path == "/api/jobs":
            if not self._require_debug_ui():
                return
            with JOBS_LOCK:
                jobs = [
                    {key: job[key] for key in ("id", "kind", "status", "created_at", "updated_at")}
                    for job in sorted(JOBS.values(), key=lambda value: value["created_at"], reverse=True)
                ]
            self._send_json({"jobs": jobs[:30]})
            return
        match = re.match(r"^/api/jobs/([a-f0-9]+)$", parsed.path)
        if match:
            if not self._require_debug_ui():
                return
            job_id = match.group(1)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                self._send_json({"error": "job not found"}, status=404)
                return
            self._send_json(job)
            return
        match = re.match(r"^/api/research/([a-f0-9]+)(/result)?$", parsed.path)
        if match:
            if not self._require_auth():
                return
            job_id = match.group(1)
            result_only = bool(match.group(2))
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job or job.get("kind") != "research":
                self._send_json({"error": "research job not found"}, status=404)
                return
            if result_only:
                if job.get("status") != "completed":
                    self._send_json({
                        "job_id": job_id,
                        "status": job.get("status"),
                        "error": job.get("error"),
                    }, status=202 if job.get("status") in {"queued", "running"} else 500)
                    return
                self._send_json(job.get("result", {}).get("canonical"))
                return
            self._send_json({
                "job_id": job_id,
                "status": job.get("status"),
                "kind": job.get("kind"),
                "created_at": job.get("created_at"),
                "updated_at": job.get("updated_at"),
                "idempotency_key": job.get("idempotency_key"),
                "events": job.get("events", []),
                "error": job.get("error"),
                "result_url": f"/api/research/{job_id}/result" if job.get("status") == "completed" else None,
            })
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path in {"/api/agent", "/api/tools", "/api/research-debug"} and not self._require_debug_ui():
                return
            payload = self._read_json()
            if parsed.path == "/api/agent":
                question = str(payload.get("question", "")).strip()
                if not question:
                    self._send_json({"error": "question is required"}, status=400)
                    return
                job_id = _start_job("agent", payload)
                self._send_json({"job_id": job_id})
                return
            if parsed.path == "/api/tools":
                query = str(payload.get("query", "")).strip()
                if not query:
                    self._send_json({"error": "query is required"}, status=400)
                    return
                job_id = _start_job("tools", payload)
                self._send_json({"job_id": job_id})
                return
            if parsed.path == "/api/research-debug":
                if not any(payload.get(key) for key in ("query", "question", "brief", "company", "therapeutic_area", "call_context")):
                    self._send_json({"error": "one of query, question, brief, company, therapeutic_area, or call_context is required"}, status=400)
                    return
                job_id = _start_job("research", payload)
                with JOBS_LOCK:
                    job = JOBS.get(job_id, {})
                self._send_json({
                    "job_id": job_id,
                    "status": job.get("status", "queued"),
                    "debug_status_url": f"/api/jobs/{job_id}",
                    "n8n_result_url": f"/api/research/{job_id}/result",
                    "idempotency_key": job.get("idempotency_key"),
                    "note": "Debug UI wrapper. The completed job.result.canonical is the same payload returned by /api/research/{job_id}/result.",
                }, status=202)
                return
            if parsed.path == "/api/research":
                if not self._require_auth():
                    return
                if not any(payload.get(key) for key in ("query", "question", "brief", "company", "therapeutic_area", "call_context")):
                    self._send_json({"error": "one of query, question, brief, company, therapeutic_area, or call_context is required"}, status=400)
                    return
                job_id = _start_job("research", payload)
                with JOBS_LOCK:
                    job = JOBS.get(job_id, {})
                self._send_json({
                    "job_id": job_id,
                    "status": job.get("status", "queued"),
                    "result_url": f"/api/research/{job_id}/result",
                    "status_url": f"/api/research/{job_id}",
                    "idempotency_key": job.get("idempotency_key"),
                }, status=202)
                return
            self._send_json({"error": "not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the GTM Research API with optional local debug UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), UIHandler)
    ui_state = "enabled" if _debug_ui_enabled() else "disabled"
    print(f"GTM Research API: http://{args.host}:{args.port} (debug UI {ui_state})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
