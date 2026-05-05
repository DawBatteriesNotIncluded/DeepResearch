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
LLM_MAX_TOKEN_PARAM=max_completion_tokens
LLM_OMIT_SAMPLING_PARAMS=true
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

## Full Smoke Test

Run the tool-level and agent-level clinical smoke test from the repo root:

```bash
python clinical_ingestion/full_smoke_test.py
```

The smoke test validates:

- ClinicalTrials.gov API output has NCT IDs, status, eligibility, outcomes, and URLs.
- Europe PMC output has citation identifiers such as PMID, PMCID, or DOI.
- The configured `visit` path returns `Evidence in page` and `Summary` sections.
- The agent calls the clinical tools and returns a final `<answer>` with structured evidence.

For a faster tool-only check:

```bash
python clinical_ingestion/full_smoke_test.py --skip-agent
```

## Local Debug UI

The browser UI is intended only for local debugging and evidence review. n8n and other automation should use the Research API endpoints below instead of `/api/agent` or `/api/tools`.

Start the local debug UI from the repo root:

```bash
python clinical_ingestion/ui_server.py
```

Then open:

```text
http://127.0.0.1:8765
```

The debug UI uses the same `.env` backend settings as the command-line agent. Use **Check Tools** for a quick ClinicalTrials.gov/Europe PMC check, **Run Agent** for the direct LLM-backed workflow, or **Run n8n Output** to run the Research API workflow and inspect the exact canonical JSON that n8n receives. Agent transcripts are saved under `outputs/ui/`.

The **n8n JSON** tab shows `job.result.canonical`, which is the same payload returned by `GET /api/research/{job_id}/result`. The **Sources** tab shows source cards extracted from tool responses or canonical Research API output, including source type, identifier, URL, metadata, and a short quote from the retrieved text when available. The **Answer** tab renders Markdown headings and bold text for readability.

The **Progress** tab streams live job events while the agent is running, including model rounds, tool calls, and tool completion events. The **Answer** and **n8n JSON** tabs are final-output views and populate only when a job completes. UI agent runs require at least one source-gathering tool call before a final answer is accepted.

The UI also builds an evidence ledger from tool responses. Ledger entries have stable IDs such as `S1`, a source identifier such as an NCT ID or PMID, a locator such as `primary_outcomes[0].description` or `abstract`, a short quote, and the source URL. Answer citations using NCT IDs, PMIDs, PMCIDs, DOIs, or `S1`-style markers become clickable links into the **Sources** tab.

Use **Search Depth** to control breadth:

- **Quick scan**: a small number of high-signal sources.
- **Standard brief**: a moderate source set for a normal research brief.
- **Deep evidence pack**: broader discovery with more query variants and more target sources.

The **Target Sources** and **Max Calls** controls can be adjusted manually when a marketing brief needs either a fast scan or a deeper source pack.

For a deployed n8n-facing service, disable the browser UI and ad-hoc debug endpoints:

```env
RESEARCH_DEBUG_UI=false
```

With debug UI disabled, `/` returns service metadata, `/api/agent`, `/api/tools`, `/api/research-debug`, and `/api/jobs` return a disabled message, and `/api/research` remains available.

## Research API for n8n

The production contract for n8n is:

```text
POST /api/research
GET /api/research/{job_id}
GET /api/research/{job_id}/result
```

Example request:

```json
{
  "mode": "account_intel",
  "company": "Example Bio",
  "domain": "example.com",
  "therapeutic_area": "rare disease gene therapy",
  "query": "Find source-backed GTM proof points for rare disease gene therapy",
  "search_depth": "standard",
  "target_sources": 12,
  "max_calls": 10,
  "idempotency_key": "hubspot-company-123-2026-05-05"
}
```

The result endpoint returns canonical JSON with:

- `schema_version`, `job_id`, `mode`, `review_status`, `confidence`, and `fit_score`.
- `brief` with executive summary, commercial angle, why-now, and recommended next steps.
- `entities` for companies, therapeutic areas, trials, products, and source references.
- `claims` with `claim_id`, `source_ids`, `risk_level`, and `usage`.
- `sources` with `source_id`, identifier, URL, locator, quote, and retrieval metadata.
- `unsupported_or_weak_claims`, `risks`, review flags, and audit paths.

Optional API-key auth can be enabled with:

```env
RESEARCH_API_KEY=your-shared-secret
RESEARCH_REVIEW_BASE_URL=http://127.0.0.1:8765
RESEARCH_DEBUG_UI=false
```

When `RESEARCH_API_KEY` is set, pass either:

```text
Authorization: Bearer your-shared-secret
```

or:

```text
X-Research-API-Key: your-shared-secret
```
