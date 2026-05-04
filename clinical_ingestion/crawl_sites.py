import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Iterable, List
from urllib.parse import urlparse


def _add_local_repo(env_var: str, sibling_name: str) -> None:
    repo_path = os.getenv(env_var)
    if not repo_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_path = os.path.abspath(os.path.join(current_dir, "..", "..", sibling_name))
    if os.path.exists(repo_path) and repo_path not in sys.path:
        sys.path.insert(0, repo_path)


_add_local_repo("SCRAPY_REPO_PATH", "scrapy")
_add_local_repo("TRAFILATURA_REPO_PATH", "trafilatura")

import scrapy
import trafilatura
from scrapy.crawler import CrawlerProcess


def _compile_patterns(patterns: Iterable[str]) -> List[re.Pattern]:
    return [re.compile(pattern) for pattern in patterns if pattern]


class ClinicalPageSpider(scrapy.Spider):
    name = "clinical_page_spider"

    def __init__(
        self,
        start_urls: List[str],
        allowed_domains: List[str],
        include_patterns: List[str],
        exclude_patterns: List[str],
        include_raw_html: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls
        self.allowed_domains = allowed_domains
        self.include_patterns = _compile_patterns(include_patterns)
        self.exclude_patterns = _compile_patterns(exclude_patterns)
        self.include_raw_html = include_raw_html

    def parse(self, response):
        html = response.text
        extracted = trafilatura.extract(
            html,
            url=response.url,
            output_format="json",
            include_comments=False,
            include_links=True,
            include_tables=True,
            favor_precision=True,
        )

        item = {
            "url": response.url,
            "status": response.status,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "content_type": response.headers.get("Content-Type", b"").decode("utf-8", errors="ignore"),
            "raw_html_sha256": hashlib.sha256(html.encode("utf-8", errors="ignore")).hexdigest(),
            "extractor": "trafilatura",
        }
        if extracted:
            try:
                item["extracted"] = json.loads(extracted)
            except json.JSONDecodeError:
                item["extracted"] = {"text": extracted}
        else:
            item["extracted"] = {}
        if self.include_raw_html:
            item["raw_html"] = html
        yield item

        for href in response.css("a::attr(href)").getall():
            next_url = response.urljoin(href)
            if self._should_follow(next_url):
                yield scrapy.Request(next_url, callback=self.parse)

    def _should_follow(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if self.allowed_domains and not any(
            parsed.netloc == domain or parsed.netloc.endswith("." + domain)
            for domain in self.allowed_domains
        ):
            return False
        if self.include_patterns and not any(pattern.search(url) for pattern in self.include_patterns):
            return False
        if self.exclude_patterns and any(pattern.search(url) for pattern in self.exclude_patterns):
            return False
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl approved clinical/case-study web sources into JSONL.")
    parser.add_argument("--start-url", action="append", required=True, help="Seed URL. Repeat for multiple seeds.")
    parser.add_argument("--allowed-domain", action="append", default=[], help="Allowed domain/netloc, e.g. example.com.")
    parser.add_argument("--include-pattern", action="append", default=[], help="Regex URL allow-list. Repeatable.")
    parser.add_argument("--exclude-pattern", action="append", default=[], help="Regex URL deny-list. Repeatable.")
    parser.add_argument("--output", required=True, help="JSONL output path.")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--download-delay", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--include-raw-html", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    process = CrawlerProcess(settings={
        "ROBOTSTXT_OBEY": True,
        "USER_AGENT": os.getenv("CRAWLER_USER_AGENT", "DeepResearchClinicalBot/0.1"),
        "DOWNLOAD_DELAY": args.download_delay,
        "CONCURRENT_REQUESTS_PER_DOMAIN": args.concurrency,
        "CLOSESPIDER_PAGECOUNT": args.max_pages,
        "DEPTH_LIMIT": args.depth,
        "FEEDS": {
            args.output: {
                "format": "jsonlines",
                "encoding": "utf8",
                "overwrite": True,
            }
        },
        "LOG_LEVEL": os.getenv("SCRAPY_LOG_LEVEL", "INFO"),
    })
    process.crawl(
        ClinicalPageSpider,
        start_urls=args.start_url,
        allowed_domains=args.allowed_domain,
        include_patterns=args.include_pattern,
        exclude_patterns=args.exclude_pattern,
        include_raw_html=args.include_raw_html,
    )
    process.start()


if __name__ == "__main__":
    main()
