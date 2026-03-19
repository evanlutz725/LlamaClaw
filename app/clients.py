from __future__ import annotations

import asyncio
from collections import deque
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.models import SearchResult, SitePage


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self._model,
            "stream": False,
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message", {})
        return message.get("content", "").strip()


class BraveSearchClient:
    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        if not self._api_key:
            return []

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": count}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
        web = data.get("web", {}).get("results", [])
        return [
            SearchResult(
                title=item.get("title", "").strip(),
                url=item.get("url", "").strip(),
                snippet=item.get("description", "").strip(),
            )
            for item in web
            if item.get("url")
        ]

    async def enrich_with_page_content(self, results: list[SearchResult], max_fetch: int = 3, max_chars: int = 1800) -> list[SearchResult]:
        if not results:
            return results

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            tasks = [self._fetch_page_excerpt(client, result.url, max_chars) for result in results[:max_fetch]]
            excerpts = await asyncio.gather(*tasks, return_exceptions=False)

        enriched: list[SearchResult] = []
        for index, result in enumerate(results):
            page_excerpt = excerpts[index] if index < len(excerpts) else None
            enriched.append(result.model_copy(update={"page_excerpt": page_excerpt}))
        return enriched

    async def crawl_site(self, start_url: str, max_pages: int = 8, max_chars: int = 1800) -> list[SitePage]:
        parsed_start = urlparse(start_url)
        if not parsed_start.scheme or not parsed_start.netloc:
            return []

        allowed_host = parsed_start.netloc
        seen: set[str] = set()
        queue: deque[str] = deque([start_url])
        pages: list[SitePage] = []

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            while queue and len(pages) < max_pages:
                current_url = queue.popleft()
                normalized_url = self._normalize_url(current_url)
                if normalized_url in seen:
                    continue
                seen.add(normalized_url)

                parsed_current = urlparse(current_url)
                if parsed_current.netloc != allowed_host:
                    continue

                try:
                    response = await client.get(current_url)
                    response.raise_for_status()
                except httpx.HTTPError:
                    continue

                if "text/html" not in response.headers.get("content-type", ""):
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else None
                excerpt = self._extract_text(soup, max_chars)
                if excerpt:
                    pages.append(SitePage(url=str(response.url), title=title, excerpt=excerpt))

                for link in self._extract_same_domain_links(soup, str(response.url), allowed_host):
                    normalized_link = self._normalize_url(link)
                    if normalized_link not in seen:
                        queue.append(link)

        return pages

    async def _fetch_page_excerpt(self, client: httpx.AsyncClient, url: str, max_chars: int) -> str | None:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        text = self._extract_text(soup, max_chars)
        return text[:max_chars] if text else None

    @staticmethod
    def _extract_text(soup: BeautifulSoup, max_chars: int) -> str:
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(chunk.strip() for chunk in soup.get_text(separator=" ").split() if chunk.strip())
        return text[:max_chars] if text else ""

    @staticmethod
    def _extract_same_domain_links(soup: BeautifulSoup, base_url: str, allowed_host: str) -> list[str]:
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href", "").strip()
            if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme in {"http", "https"} and parsed.netloc == allowed_host:
                links.append(absolute)
        return links

    @staticmethod
    def _normalize_url(url: str) -> str:
        parsed = urlparse(url)
        normalized_path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc}{normalized_path}"

    @staticmethod
    def format_results(results: list[SearchResult]) -> str:
        if not results:
            return "No Brave Search results were available."

        lines: list[str] = []
        for index, item in enumerate(results, start=1):
            lines.append(f"{index}. {item.title}")
            lines.append(f"   URL: {item.url}")
            lines.append(f"   Snippet: {item.snippet}")
            if item.page_excerpt:
                lines.append(f"   Page excerpt: {item.page_excerpt}")
        return "\n".join(lines)

    @staticmethod
    def format_site_pages(pages: list[SitePage]) -> str:
        if not pages:
            return "No crawled site pages were available."

        lines: list[str] = []
        for index, page in enumerate(pages, start=1):
            lines.append(f"{index}. {page.title or 'Untitled page'}")
            lines.append(f"   URL: {page.url}")
            lines.append(f"   Excerpt: {page.excerpt}")
        return "\n".join(lines)
