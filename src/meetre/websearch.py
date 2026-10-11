"""Opt-in web search for the chat window (DuckDuckGo HTML, no API key).

This is the ONLY part of meetre that talks to the internet with user content:
it sends the typed question as a search query (never audio or the transcript)
and only when "Web search" is ticked in the chat window.
"""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from typing import List, NamedTuple

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko)"
_TIMEOUT = 8


class Hit(NamedTuple):
    title: str
    url: str
    snippet: str


def _get(url: str, data: bytes = None) -> str:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:  # noqa: S310 — https only
        return r.read(2_000_000).decode(r.headers.get_content_charset() or "utf-8", "replace")


def _text(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _real_url(href: str) -> str:
    # DuckDuckGo wraps results as //duckduckgo.com/l/?uddg=<encoded target>
    q = urllib.parse.urlparse(html.unescape(href)).query
    target = urllib.parse.parse_qs(q).get("uddg")
    return target[0] if target else html.unescape(href)


def search(query: str, limit: int = 5) -> List[Hit]:
    """Top web results for ``query`` (title, url, snippet)."""
    page = _get("https://html.duckduckgo.com/html/",
                urllib.parse.urlencode({"q": query[:300]}).encode())
    links = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    hits = []
    for i, (href, title) in enumerate(links):
        url = _real_url(href)
        if not url.startswith("http") or "duckduckgo.com/y.js" in url:
            continue  # ads
        hits.append(Hit(_text(title), url, _text(snippets[i]) if i < len(snippets) else ""))
        if len(hits) >= limit:
            break
    return hits


def page_text(url: str, limit: int = 3000) -> str:
    """Readable text of a page (scripts/styles/markup stripped), truncated."""
    raw = _get(url)
    raw = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", raw)
    text = _text(raw)
    return re.sub(r"\s+", " ", text)[:limit]


def context_for(query: str, limit: int = 5, read_top: int = 2) -> tuple:
    """(formatted context for the LLM, hits). Reads the top pages for depth."""
    hits = search(query, limit)
    blocks = []
    for i, h in enumerate(hits, 1):
        body = h.snippet
        if i <= read_top:
            try:
                body = page_text(h.url) or body
            except Exception:  # noqa: BLE001 — a slow/blocked page just keeps its snippet
                pass
        blocks.append(f"[{i}] {h.title}\n{h.url}\n{body}")
    return "\n\n".join(blocks), hits
