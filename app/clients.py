from __future__ import annotations

import httpx

from app.models import SearchResult


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

    @staticmethod
    def format_results(results: list[SearchResult]) -> str:
        if not results:
            return "No Brave Search results were available."

        lines: list[str] = []
        for index, item in enumerate(results, start=1):
            lines.append(f"{index}. {item.title}")
            lines.append(f"   URL: {item.url}")
            lines.append(f"   Snippet: {item.snippet}")
        return "\n".join(lines)
