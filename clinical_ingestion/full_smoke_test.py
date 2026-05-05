import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class SmokeFailure(AssertionError):
    pass


def _add_inference_path() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    inference_dir = os.path.abspath(os.path.join(current_dir, "..", "inference"))
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))
    load_dotenv(os.path.join(root_dir, ".env"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _parse_tool_json(tool_name: str, raw: str) -> Dict[str, Any]:
    _require(isinstance(raw, str) and raw.strip(), f"{tool_name} returned empty output")
    _require(not raw.lstrip().startswith(f"[{tool_name}]"), raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"{tool_name} returned invalid JSON: {exc}") from exc


def _tool_names_from_messages(messages: List[Dict[str, Any]]) -> List[str]:
    names = []
    for message in messages:
        content = message.get("content", "")
        for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.S):
            try:
                payload = json.loads(match)
            except json.JSONDecodeError:
                continue
            if payload.get("name"):
                names.append(payload["name"])
    return names


def _run_check(name: str, func) -> Tuple[str, Any]:
    print(f"\n=== {name} ===")
    try:
        result = func()
    except SmokeFailure as exc:
        print(f"FAIL: {exc}")
        return "fail", str(exc)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return "fail", f"{type(exc).__name__}: {exc}"
    print("PASS")
    return "pass", result


def run_tool_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    from tool_clinical import ClinicalTrialsSearch, ExtractWebpage, LiteratureSearch
    from tool_visit import Visit

    trials_raw = ClinicalTrialsSearch().call({
        "query": args.query,
        "max_results": args.max_results,
    })
    trials_payload = _parse_tool_json("clinical_trials_search", trials_raw)
    studies = trials_payload.get("studies") or []
    _require(studies, "clinical_trials_search returned no studies")

    first_study = studies[0]
    _require(re.match(r"^NCT\d{8}$", str(first_study.get("nct_id", ""))) is not None, "first study has no valid NCT ID")
    _require(bool(first_study.get("overall_status")), "first study has no overall_status")
    _require("eligibility" in first_study, "first study missing eligibility block")
    _require("primary_outcomes" in first_study, "first study missing primary_outcomes")

    literature_raw = LiteratureSearch().call({
        "query": args.query,
        "max_results": args.max_results,
    })
    literature_payload = _parse_tool_json("literature_search", literature_raw)
    papers = literature_payload.get("results") or []
    _require(papers, "literature_search returned no papers")

    first_paper = papers[0]
    _require(bool(first_paper.get("title")), "first paper has no title")
    _require(bool(first_paper.get("pmid") or first_paper.get("pmcid") or first_paper.get("doi")), "first paper has no PMID, PMCID, or DOI")

    evidence = {
        "clinical_trials": {
            "source": trials_payload.get("source"),
            "returned": trials_payload.get("returned"),
            "first_nct_id": first_study.get("nct_id"),
            "first_status": first_study.get("overall_status"),
            "first_url": first_study.get("url"),
        },
        "literature": {
            "source": literature_payload.get("source"),
            "returned": literature_payload.get("returned"),
            "first_title": first_paper.get("title"),
            "first_pmid": first_paper.get("pmid"),
            "first_pmcid": first_paper.get("pmcid"),
            "first_doi": first_paper.get("doi"),
        },
    }

    if not args.skip_visit:
        article_terms = [
            term.lower().strip(".,:;()[]")
            for term in str(first_paper.get("title", "")).split()
            if len(term.strip(".,:;()[]")) > 4
        ][:6]
        visit_candidates = []
        if first_paper.get("url"):
            visit_candidates.append({"url": first_paper["url"], "terms": article_terms})
        if first_paper.get("pmcid"):
            visit_candidates.append({
                "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{first_paper['pmcid']}/",
                "terms": article_terms,
            })
        if first_paper.get("pmid"):
            visit_candidates.append({
                "url": f"https://europepmc.org/article/MED/{first_paper['pmid']}",
                "terms": article_terms,
            })
        visit_candidates.extend([
            {
                "url": "https://www.fda.gov/vaccines-blood-biologics/cellular-gene-therapy-products/approved-cellular-and-gene-therapy-products",
                "terms": ["gene", "therapy"],
            },
            {
                "url": "https://medlineplus.gov/genetics/understanding/therapy/genetherapy/",
                "terms": ["gene", "therapy"],
            },
        ])
        if first_study.get("url"):
            visit_candidates.append({"url": first_study["url"], "terms": ["clinical", "trial"]})
        _require(bool(visit_candidates), "no URL available for visit smoke")

        previous_backend = os.getenv("VISIT_BACKEND")
        if args.visit_backend:
            os.environ["VISIT_BACKEND"] = args.visit_backend
        try:
            visit_url = ""
            visit_result = ""
            visit_error = "no visit candidate succeeded"
            for candidate in visit_candidates:
                candidate_url = candidate["url"]
                candidate_result = Visit().call({
                    "url": candidate_url,
                    "goal": "Extract the article title, citation identifiers, abstract, and relevance to rare-disease gene therapy.",
                })
                if "Evidence in page:" not in candidate_result or "Summary:" not in candidate_result:
                    visit_error = candidate_result[:500]
                    continue
                expected_terms = candidate.get("terms") or []
                if expected_terms and not any(term in candidate_result.lower() for term in expected_terms):
                    visit_error = "visit output did not contain expected article title terms"
                    continue
                visit_url = candidate_url
                visit_result = candidate_result
                break
        finally:
            if previous_backend is None:
                os.environ.pop("VISIT_BACKEND", None)
            else:
                os.environ["VISIT_BACKEND"] = previous_backend
        _require(bool(visit_url), visit_error)
        evidence["visit"] = {
            "backend": args.visit_backend or os.getenv("VISIT_BACKEND", "jina"),
            "url": visit_url,
            "chars": len(visit_result),
        }

        extract_payload = None
        extract_url = ""
        extract_error = "no extract_webpage candidate succeeded"
        for candidate in visit_candidates:
            candidate_url = candidate["url"]
            extract_raw = ExtractWebpage().call({
                "url": candidate_url,
                "goal": "Extract clean article evidence for smoke validation.",
            })
            try:
                candidate_payload = _parse_tool_json("extract_webpage", extract_raw)
            except SmokeFailure as exc:
                extract_error = str(exc)
                continue
            if candidate_payload.get("source") != "trafilatura":
                extract_error = "extract_webpage did not use trafilatura"
                continue
            if len(candidate_payload.get("text", "")) < 500:
                extract_error = "extract_webpage returned too little extracted text"
                continue
            extract_payload = candidate_payload
            extract_url = candidate_url
            break
        _require(extract_payload is not None, extract_error)
        evidence["extract_webpage"] = {
            "source": extract_payload.get("source"),
            "url": extract_url,
            "title": extract_payload.get("title"),
            "chars": len(extract_payload.get("text", "")),
        }

    print(json.dumps(evidence, ensure_ascii=False, indent=2)[:4000])
    return evidence


def _agent_ready() -> Tuple[bool, str]:
    provider = os.getenv("LLM_PROVIDER", "vllm").strip().lower()
    if provider == "azure":
        missing = [
            name for name in (
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_DEPLOYMENT",
            )
            if not os.getenv(name)
        ]
        return not missing, f"missing {', '.join(missing)}" if missing else "ready"
    if provider == "openai":
        missing = [name for name in ("OPENAI_API_KEY", "OPENAI_MODEL") if not os.getenv(name)]
        return not missing, f"missing {', '.join(missing)}" if missing else "ready"
    return True, "vllm/openai-compatible backend configured"


def run_agent_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    ready, reason = _agent_ready()
    if not ready:
        if args.require_agent:
            raise SmokeFailure(f"agent smoke required but backend is not ready: {reason}")
        print(f"SKIP: agent backend is not ready: {reason}")
        return {"skipped": True, "reason": reason}

    os.environ.setdefault("MAX_LLM_CALL_PER_RUN", str(args.agent_max_calls))
    os.environ.setdefault("LLM_MAX_TRIES", str(args.agent_max_tries))
    os.environ.setdefault("LLM_REQUEST_TIMEOUT", str(args.agent_timeout))
    os.environ.setdefault("LLM_MAX_OUTPUT_TOKENS", str(args.agent_max_output_tokens))

    from react_agent import MultiTurnReactAgent

    model = args.model or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_MODEL") or os.getenv("MODEL_PATH", "smoke")
    question = (
        f'Use clinical_trials_search with query "{args.query}" and max_results 2. '
        f'Use literature_search with query "{args.query}" and max_results 2. '
        "Return the final answer inside <answer> tags with exactly these headings: "
        "Evidence, Key trial, Literature, Bottom line. Include at least one NCT ID, trial status, "
        "disease focus, primary outcome or endpoint pattern, and one PMID, PMCID, or DOI if available. "
        "Keep it under 180 words."
    )
    item = {"question": question, "answer": ""}
    agent = MultiTurnReactAgent(
        llm={
            "model": model,
            "generate_cfg": {
                "max_input_tokens": 320000,
                "max_retries": args.agent_max_tries,
                "temperature": 0,
                "top_p": 1,
                "presence_penalty": 0,
            },
            "model_type": "qwen_dashscope",
        },
        function_list=[
            "clinical_trials_search",
            "literature_search",
            "visit",
            "extract_webpage",
            "parse_pdf_grobid",
        ],
    )

    result = agent._run({"item": item, "planning_port": 6001}, model)
    messages = result.get("messages") or []
    transcript = "\n\n".join(str(message.get("content", "")) for message in messages)
    tool_names = _tool_names_from_messages(messages)
    prediction = str(result.get("prediction", ""))

    _require(result.get("termination") == "answer", f"agent did not terminate with answer: {result.get('termination')}")
    _require("clinical_trials_search" in tool_names, "agent did not call clinical_trials_search")
    _require("literature_search" in tool_names, "agent did not call literature_search")
    _require("ClinicalTrials.gov API v2" in transcript, "transcript missing ClinicalTrials.gov evidence")
    _require("Europe PMC REST API" in transcript, "transcript missing Europe PMC evidence")
    _require(re.search(r"NCT\d{8}", prediction) is not None, "final answer missing NCT ID")
    _require(any(token in prediction.lower() for token in ("status", "recruit", "completed", "active")), "final answer missing trial status")
    _require(any(token in prediction.lower() for token in ("endpoint", "outcome", "primary")), "final answer missing endpoint/outcome language")
    _require(re.search(r"\b(PMID|PMCID|DOI)\b", prediction, flags=re.I) is not None, "final answer missing PMID, PMCID, or DOI")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"agent_full_smoke_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    summary = {
        "termination": result.get("termination"),
        "tool_names": tool_names,
        "prediction": prediction,
        "output_path": output_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full clinical evidence and agent smoke tests.")
    parser.add_argument("--query", default="gene therapy rare disease")
    parser.add_argument("--max-results", type=int, default=2)
    parser.add_argument("--visit-backend", default="trafilatura", choices=["trafilatura", "jina", "firecrawl"])
    parser.add_argument("--skip-visit", action="store_true")
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--require-agent", action="store_true")
    parser.add_argument("--agent-max-calls", type=int, default=6)
    parser.add_argument("--agent-max-tries", type=int, default=2)
    parser.add_argument("--agent-timeout", type=float, default=120)
    parser.add_argument("--agent-max-output-tokens", type=int, default=4096)
    parser.add_argument("--model", default="")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "smoke"))
    args = parser.parse_args()

    _load_dotenv()
    _add_inference_path()

    checks = []
    if not args.skip_tools:
        checks.append(("tool evidence smoke", lambda: run_tool_smoke(args)))
    if not args.skip_agent:
        checks.append(("agent end-to-end smoke", lambda: run_agent_smoke(args)))

    results = {}
    failed = False
    for name, func in checks:
        status, result = _run_check(name, func)
        results[name] = {"status": status, "result": result}
        failed = failed or status == "fail"

    print("\n=== summary ===")
    print(json.dumps(results, ensure_ascii=False, indent=2)[:6000])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
