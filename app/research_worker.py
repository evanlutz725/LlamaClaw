from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.clients import BraveSearchClient


async def run_search(api_key: str, query: str, count: int, scrape_count: int, max_chars: int) -> dict:
    client = BraveSearchClient(api_key=api_key)
    results = await client.search(query, count=count)
    results = await client.enrich_with_page_content(results, max_fetch=scrape_count, max_chars=max_chars)
    return {
        "worker_type": "search",
        "target": query,
        "results": [item.model_dump() for item in results],
        "summary": client.format_results(results),
    }


async def run_crawl(api_key: str, url: str, max_pages: int, max_chars: int) -> dict:
    client = BraveSearchClient(api_key=api_key)
    pages = await client.crawl_site(url, max_pages=max_pages, max_chars=max_chars)
    return {
        "worker_type": "crawl",
        "target": url,
        "pages": [item.model_dump() for item in pages],
        "summary": client.format_site_pages(pages),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["search", "crawl"])
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query")
    parser.add_argument("--url")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--scrape-count", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=1800)
    args = parser.parse_args()

    if args.mode == "search":
        payload = asyncio.run(run_search(args.api_key, args.query or "", args.count, args.scrape_count, args.max_chars))
    else:
        payload = asyncio.run(run_crawl(args.api_key, args.url or "", args.max_pages, args.max_chars))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
