# Clinical Research Agent Setup

This repo now has a clinical research path layered on top of the original ReAct inference loop.

## Model Backend

Use Azure OpenAI by setting:

```env
LLM_PROVIDER=azure
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
```

When `LLM_PROVIDER=azure`, you can run `inference/run_multi_react.py` directly. The original `inference/run_react_infer.sh` still starts local vLLM servers and is only needed for `LLM_PROVIDER=vllm`.

## Web Extraction

The existing `visit` tool can use one of three backends:

```env
VISIT_BACKEND=trafilatura
TRAFILATURA_REPO_PATH=C:\Users\danie\source\repos\DawBatteriesNotIncluded\trafilatura
```

Other values:

- `jina`: original Jina Reader behavior.
- `firecrawl`: self-hosted Firecrawl via `FIRECRAWL_BASE_URL`, default `http://localhost:3002`.
- `trafilatura`: local or installed Trafilatura extraction.

There is also a direct `extract_webpage` tool for one-off clean page extraction.

## Clinical Sources

The agent has additional tools:

- `clinical_trials_search`: ClinicalTrials.gov API v2.
- `literature_search`: Europe PMC REST API.
- `parse_pdf_grobid`: self-hosted GROBID PDF parsing.
- `extract_webpage`: Trafilatura HTML extraction.

For biomedical work, use official APIs before general web crawling wherever possible.

## GROBID

Run a GROBID service and point the agent at it:

```env
GROBID_BASE_URL=http://localhost:8070
FILE_CORPUS_PATH=./eval_data/file_corpus
```

The sibling `grobid` repo includes Docker and service documentation. The parser calls:

```text
POST /api/processFulltextDocument
```

## Controlled Site Crawling

Use Scrapy outside the LLM loop for approved domains:

```bash
python clinical_ingestion/crawl_sites.py ^
  --start-url https://example.com/case-studies ^
  --allowed-domain example.com ^
  --include-pattern "/case-stud" ^
  --output outputs/example_case_studies.jsonl ^
  --max-pages 100 ^
  --depth 2
```

This writes JSONL with URL, fetch timestamp, raw HTML hash, content type, and Trafilatura extraction. Keep raw source archives or hashes for auditability.

## Minimal Azure Run

Example:

```bash
cd inference
python -u run_multi_react.py ^
  --dataset eval_data/example.jsonl ^
  --output ../outputs ^
  --model azure ^
  --max_workers 2 ^
  --roll_out_count 1
```

Use a dataset JSONL with:

```json
{"question": "Find active gene therapy trials for X and summarize eligibility and endpoints.", "answer": ""}
```
