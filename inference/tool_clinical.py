import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool


REQUEST_TIMEOUT = int(os.getenv("CLINICAL_TOOL_TIMEOUT", "60"))
MAX_TEXT_CHARS = int(os.getenv("CLINICAL_TOOL_MAX_TEXT_CHARS", "50000"))


def _parse_params(params: Union[str, dict]) -> Dict[str, Any]:
    if isinstance(params, dict):
        return params
    if isinstance(params, str):
        return json.loads(params)
    raise ValueError("Tool input must be a JSON object.")


def _truncate(value: Any, limit: int = MAX_TEXT_CHARS) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


def _compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_text_list(items: Any, key: Optional[str] = None, limit: int = 8) -> List[str]:
    values = []
    for item in _as_list(items):
        if isinstance(item, dict) and key:
            candidate = item.get(key)
        else:
            candidate = item
        if candidate:
            values.append(str(candidate))
    return values[:limit]


@register_tool("clinical_trials_search", allow_overwrite=True)
class ClinicalTrialsSearch(BaseTool):
    name = "clinical_trials_search"
    description = "Search ClinicalTrials.gov using its public API and return normalized trial records."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Free-text clinical trial search query."},
            "condition": {"type": "string", "description": "Optional condition/disease filter."},
            "intervention": {"type": "string", "description": "Optional drug, device, gene, or behavioral intervention filter."},
            "status": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional overall statuses, e.g. RECRUITING or COMPLETED.",
            },
            "max_results": {"type": "integer", "description": "Maximum studies to return, up to 100."},
        },
        "required": ["query"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = _parse_params(params)
            query = params["query"]
            max_results = min(int(params.get("max_results", 10)), 100)
        except Exception as e:
            return f"[clinical_trials_search] Invalid request: {e}"

        api_params = {
            "format": "json",
            "pageSize": max_results,
            "query.term": query,
        }
        if params.get("condition"):
            api_params["query.cond"] = params["condition"]
        if params.get("intervention"):
            api_params["query.intr"] = params["intervention"]
        if params.get("status"):
            api_params["filter.overallStatus"] = ",".join(str(s) for s in _as_list(params["status"]))

        try:
            response = requests.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params=api_params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            return f"[clinical_trials_search] Request failed: {e}"

        studies = []
        for study in payload.get("studies", []):
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            status_mod = protocol.get("statusModule", {})
            desc = protocol.get("descriptionModule", {})
            cond = protocol.get("conditionsModule", {})
            design = protocol.get("designModule", {})
            arms = protocol.get("armsInterventionsModule", {})
            outcomes = protocol.get("outcomesModule", {})
            eligibility = protocol.get("eligibilityModule", {})
            locations = protocol.get("contactsLocationsModule", {})

            interventions = []
            for intervention in arms.get("interventions", []) or []:
                interventions.append({
                    "type": intervention.get("type"),
                    "name": intervention.get("name"),
                    "description": _truncate(intervention.get("description", ""), 1200),
                })

            primary_outcomes = []
            for outcome in outcomes.get("primaryOutcomes", []) or []:
                primary_outcomes.append({
                    "measure": outcome.get("measure"),
                    "time_frame": outcome.get("timeFrame"),
                    "description": _truncate(outcome.get("description", ""), 1200),
                })

            location_names = []
            for location in locations.get("locations", []) or []:
                pieces = [
                    location.get("facility"),
                    location.get("city"),
                    location.get("state"),
                    location.get("country"),
                ]
                location_names.append(", ".join(piece for piece in pieces if piece))

            studies.append({
                "nct_id": ident.get("nctId"),
                "brief_title": ident.get("briefTitle"),
                "official_title": ident.get("officialTitle"),
                "overall_status": status_mod.get("overallStatus"),
                "start_date": status_mod.get("startDateStruct", {}).get("date"),
                "completion_date": status_mod.get("completionDateStruct", {}).get("date"),
                "conditions": _extract_text_list(cond.get("conditions"), limit=12),
                "phases": _extract_text_list(design.get("phases"), limit=6),
                "enrollment": design.get("enrollmentInfo", {}),
                "interventions": interventions[:8],
                "primary_outcomes": primary_outcomes[:8],
                "eligibility": {
                    "sex": eligibility.get("sex"),
                    "minimum_age": eligibility.get("minimumAge"),
                    "maximum_age": eligibility.get("maximumAge"),
                    "healthy_volunteers": eligibility.get("healthyVolunteers"),
                    "criteria": _truncate(eligibility.get("eligibilityCriteria", ""), 5000),
                },
                "brief_summary": _truncate(desc.get("briefSummary", ""), 4000),
                "detailed_description": _truncate(desc.get("detailedDescription", ""), 6000),
                "locations": location_names[:10],
                "url": f"https://clinicaltrials.gov/study/{ident.get('nctId')}" if ident.get("nctId") else None,
            })

        return _compact_json({
            "source": "ClinicalTrials.gov API v2",
            "query": query,
            "returned": len(studies),
            "studies": studies,
        })


@register_tool("literature_search", allow_overwrite=True)
class LiteratureSearch(BaseTool):
    name = "literature_search"
    description = "Search biomedical literature through Europe PMC and return citation metadata and abstracts."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Biomedical literature search query."},
            "max_results": {"type": "integer", "description": "Maximum papers to return, up to 100."},
        },
        "required": ["query"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = _parse_params(params)
            query = params["query"]
            max_results = min(int(params.get("max_results", 10)), 100)
        except Exception as e:
            return f"[literature_search] Invalid request: {e}"

        try:
            response = requests.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": max_results,
                    "resultType": "core",
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            return f"[literature_search] Request failed: {e}"

        results = []
        for item in payload.get("resultList", {}).get("result", []) or []:
            results.append({
                "title": item.get("title"),
                "authors": item.get("authorString"),
                "journal": item.get("journalTitle"),
                "year": item.get("pubYear"),
                "doi": item.get("doi"),
                "pmid": item.get("pmid"),
                "pmcid": item.get("pmcid"),
                "source": item.get("source"),
                "is_open_access": item.get("isOpenAccess"),
                "abstract": _truncate(item.get("abstractText", ""), 6000),
                "url": item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url")
                if item.get("fullTextUrlList", {}).get("fullTextUrl") else None,
            })

        return _compact_json({
            "source": "Europe PMC REST API",
            "query": query,
            "returned": len(results),
            "results": results,
        })


def _load_trafilatura():
    try:
        import trafilatura
        return trafilatura
    except Exception:
        repo_path = os.getenv("TRAFILATURA_REPO_PATH")
        if not repo_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_path = os.path.abspath(os.path.join(current_dir, "..", "..", "trafilatura"))
        if os.path.exists(repo_path):
            sys.path.insert(0, repo_path)
            import trafilatura
            return trafilatura
    raise ImportError("Install trafilatura or set TRAFILATURA_REPO_PATH.")


@register_tool("extract_webpage", allow_overwrite=True)
class ExtractWebpage(BaseTool):
    name = "extract_webpage"
    description = "Fetch a webpage and extract clean article/case-study text with Trafilatura."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Webpage URL to fetch and extract."},
            "goal": {"type": "string", "description": "Specific information goal for extraction."},
        },
        "required": ["url", "goal"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = _parse_params(params)
            url = params["url"]
            goal = params.get("goal", "")
        except Exception as e:
            return f"[extract_webpage] Invalid request: {e}"

        try:
            trafilatura = _load_trafilatura()
        except Exception as e:
            return f"[extract_webpage] Trafilatura unavailable: {e}"

        try:
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": os.getenv("CRAWLER_USER_AGENT", "DeepResearchClinicalBot/0.1")},
            )
            response.raise_for_status()
        except Exception as e:
            return f"[extract_webpage] Fetch failed for {url}: {e}"

        try:
            raw = trafilatura.extract(
                response.text,
                url=url,
                output_format="json",
                include_comments=False,
                include_links=True,
                include_tables=True,
                favor_precision=True,
            )
            if raw:
                extracted = json.loads(raw)
            else:
                text = trafilatura.extract(
                    response.text,
                    url=url,
                    include_comments=False,
                    include_links=True,
                    include_tables=True,
                    favor_precision=True,
                )
                extracted = {"text": text or ""}
        except Exception as e:
            return f"[extract_webpage] Extraction failed for {url}: {e}"

        return _compact_json({
            "source": "trafilatura",
            "url": url,
            "goal": goal,
            "title": extracted.get("title"),
            "author": extracted.get("author"),
            "date": extracted.get("date"),
            "hostname": extracted.get("hostname"),
            "description": extracted.get("description"),
            "text": _truncate(extracted.get("text", ""), MAX_TEXT_CHARS),
        })


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find_first_text(root: ET.Element, tag_name: str) -> str:
    for elem in root.iter():
        if _strip_namespace(elem.tag) == tag_name and elem.text:
            return " ".join(elem.itertext()).strip()
    return ""


def _find_section_text(root: ET.Element, section_name: str, max_paragraphs: int = 80) -> List[str]:
    output = []
    for elem in root.iter():
        if _strip_namespace(elem.tag) != section_name:
            continue
        for child in elem.iter():
            if _strip_namespace(child.tag) == "p":
                text = " ".join(child.itertext()).strip()
                text = re.sub(r"\s+", " ", text)
                if text:
                    output.append(text)
                if len(output) >= max_paragraphs:
                    return output
    return output


def _resolve_file_path(path: str) -> Optional[str]:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = [
        path,
        os.path.abspath(path),
        os.path.join(os.getenv("FILE_CORPUS_PATH", "./eval_data/file_corpus"), path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


@register_tool("parse_pdf_grobid", allow_overwrite=True)
class GrobidPDFParser(BaseTool):
    name = "parse_pdf_grobid"
    description = "Parse clinical or scholarly PDFs with a self-hosted GROBID service."
    parameters = {
        "type": "object",
        "properties": {
            "file": {"type": "string", "description": "Local PDF filename/path or a direct PDF URL."},
            "include_body": {"type": "boolean", "description": "Whether to include extracted body paragraphs."},
        },
        "required": ["file"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = _parse_params(params)
            file_ref = params["file"]
            include_body = bool(params.get("include_body", True))
        except Exception as e:
            return f"[parse_pdf_grobid] Invalid request: {e}"

        grobid_base = os.getenv("GROBID_BASE_URL", "http://localhost:8070").rstrip("/")
        endpoint = f"{grobid_base}/api/processFulltextDocument"

        cleanup_path = None
        file_path = None
        try:
            if file_ref.startswith(("http://", "https://")):
                response = requests.get(file_ref, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                fd, cleanup_path = tempfile.mkstemp(suffix=".pdf")
                with os.fdopen(fd, "wb") as handle:
                    handle.write(response.content)
                file_path = cleanup_path
            else:
                file_path = _resolve_file_path(file_ref)
                if not file_path:
                    return f"[parse_pdf_grobid] File not found: {file_ref}"

            with open(file_path, "rb") as handle:
                response = requests.post(
                    endpoint,
                    files={"input": (os.path.basename(file_path), handle, "application/pdf")},
                    data={
                        "consolidateHeader": "1",
                        "consolidateCitations": "1",
                        "includeRawCitations": "1",
                    },
                    timeout=max(REQUEST_TIMEOUT, 120),
                )
            response.raise_for_status()
            tei_xml = response.text
        except Exception as e:
            return f"[parse_pdf_grobid] GROBID request failed: {e}"
        finally:
            if cleanup_path and os.path.exists(cleanup_path):
                try:
                    os.remove(cleanup_path)
                except OSError:
                    pass

        try:
            root = ET.fromstring(tei_xml)
            body_paragraphs = _find_section_text(root, "body") if include_body else []
            payload = {
                "source": "GROBID",
                "file": file_ref,
                "title": _find_first_text(root, "title"),
                "abstract": _truncate("\n".join(_find_section_text(root, "abstract", max_paragraphs=20)), 12000),
                "body": _truncate("\n\n".join(body_paragraphs), MAX_TEXT_CHARS) if include_body else "",
                "tei_xml_chars": len(tei_xml),
            }
            return _compact_json(payload)
        except Exception as e:
            return _compact_json({
                "source": "GROBID",
                "file": file_ref,
                "parse_warning": str(e),
                "tei_xml": _truncate(tei_xml, MAX_TEXT_CHARS),
            })
