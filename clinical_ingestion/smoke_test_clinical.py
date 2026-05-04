import argparse
import json
import os
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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


def _print_result(name: str, result: Any) -> None:
    print(f"\n=== {name} ===")
    if isinstance(result, str):
        print(result[:5000])
        if len(result) > 5000:
            print(f"\n...[truncated {len(result) - 5000} chars]")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2)[:5000])


def smoke_azure() -> None:
    from openai import AzureOpenAI

    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        _print_result("azure", f"Skipped: missing {', '.join(missing)}")
        return

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        timeout=60.0,
    )
    kwargs = {
        "model": os.environ["AZURE_OPENAI_DEPLOYMENT"],
        "messages": [{"role": "user", "content": "Reply with exactly: azure-ok"}],
    }
    token_param = os.getenv("LLM_MAX_TOKEN_PARAM", "max_tokens")
    kwargs[token_param] = int(os.getenv("SMOKE_MAX_OUTPUT_TOKENS", "32"))
    if os.getenv("SMOKE_SET_TEMPERATURE", "false").lower() in {"1", "true", "yes"}:
        kwargs["temperature"] = 0

    response = client.chat.completions.create(**kwargs)
    _print_result("azure", response.choices[0].message.content)


def smoke_trials(query: str) -> None:
    from tool_clinical import ClinicalTrialsSearch

    result = ClinicalTrialsSearch().call({"query": query, "max_results": 3})
    _print_result("clinical_trials_search", result)


def smoke_literature(query: str) -> None:
    from tool_clinical import LiteratureSearch

    result = LiteratureSearch().call({"query": query, "max_results": 3})
    _print_result("literature_search", result)


def smoke_extract(url: str) -> None:
    from tool_clinical import ExtractWebpage

    result = ExtractWebpage().call({"url": url, "goal": "Smoke-test webpage extraction."})
    _print_result("extract_webpage", result)


def smoke_grobid(pdf: str) -> None:
    from tool_clinical import GrobidPDFParser

    result = GrobidPDFParser().call({"file": pdf, "include_body": False})
    _print_result("parse_pdf_grobid", result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Azure and clinical ingestion tools.")
    parser.add_argument("--skip-azure", action="store_true")
    parser.add_argument("--skip-apis", action="store_true")
    parser.add_argument("--query", default="gene therapy rare disease")
    parser.add_argument("--url", default="")
    parser.add_argument("--pdf", default="")
    args = parser.parse_args()

    _load_dotenv()
    _add_inference_path()

    if not args.skip_azure:
        smoke_azure()
    if not args.skip_apis:
        smoke_trials(args.query)
        smoke_literature(args.query)
    if args.url:
        smoke_extract(args.url)
    if args.pdf:
        smoke_grobid(args.pdf)


if __name__ == "__main__":
    main()
