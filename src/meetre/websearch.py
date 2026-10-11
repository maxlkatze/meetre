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

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
       "(KHTML, like Gecko) Version/18.5 Safari/605.1.15")
_HEADERS = {"User-Agent": _UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,de;q=0.8"}
_TIMEOUT = 8


class Hit(NamedTuple):
    title: str
    url: str
    snippet: str


def _get(url: str, data: bytes = None) -> str:
    req = urllib.request.Request(url, data=data, headers=_HEADERS)
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


# Bot-protection pages (Cloudflare & co.) that need a real browser engine.
_CHALLENGE = re.compile(r"Just a moment|Client Challenge|Enable JavaScript|"
                        r"Checking your browser|cf-browser-verification|captcha", re.I)


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", raw)
    return re.sub(r"\s+", " ", _text(raw)).strip()


def page_text(url: str, limit: int = 3000) -> str:
    """Readable text of a page, truncated.

    Plain HTTP first; when the site blocks it (403/429/503, a bot challenge)
    or renders its content with JavaScript, load it in an invisible WebKit
    view — Safari's engine, so it passes the checks a real browser passes.
    """
    import urllib.error

    try:
        raw = _get(url)
        text = _html_to_text(raw)
        if not _CHALLENGE.search(text[:2000]) and len(text) > 400:
            return text[:limit]
    except urllib.error.HTTPError as e:
        if e.code not in (401, 403, 429, 503):
            raise
    except urllib.error.URLError:
        raise
    text = _webkit_text(url)
    if not text:
        raise RuntimeError("the site blocked automated access (also in WebKit)")
    return re.sub(r"[ \t]+", " ", text).strip()[:limit]


_JS_TEXT = "document.title + '\\n' + (document.body ? document.body.innerText : '')"
_wk_ready = False


def _wk_classes():
    global _wk_ready
    import objc

    if not _wk_ready:
        objc.loadBundle("WebKit", {}, bundle_path="/System/Library/Frameworks/WebKit.framework")
        objc.registerMetaDataForSelector(
            b"WKWebView", b"evaluateJavaScript:completionHandler:",
            {"arguments": {3: {"callable": {"retval": {"type": b"v"}, "arguments": {
                0: {"type": b"^v"}, 1: {"type": b"@"}, 2: {"type": b"@"}}}}}})
        _wk_ready = True
    return objc.lookUpClass("WKWebView"), objc.lookUpClass("WKWebViewConfiguration")


class _WebKitPage:
    """One off-screen page load. All methods run on the main thread."""

    def __init__(self, url: str):
        from AppKit import NSMakeRect
        from Foundation import NSURL, NSURLRequest

        WKWebView, WKConf = _wk_classes()
        conf = WKConf.alloc().init()
        try:  # don't let pages play audio/video while they're scraped
            conf.setMediaTypesRequiringUserActionForPlayback_(0xFFFFFFFF)
        except Exception:  # noqa: BLE001
            pass
        self.view = WKWebView.alloc().initWithFrame_configuration_(NSMakeRect(0, 0, 1280, 900), conf)
        self.view.setCustomUserAgent_(_UA)
        self.view.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(url)))
        self.text = None   # latest innerText snapshot
        self.prev = None
        self.stable = 0

    def poll(self) -> None:
        if self.view is None or self.view.isLoading():
            return

        def got(result, error):
            t = str(result or "")
            if t and not _CHALLENGE.search(t[:2000]) and len(t) > 200:
                self.stable = self.stable + 1 if t == self.prev else 0
                self.prev = t
            self.text = t

        self.view.evaluateJavaScript_completionHandler_(_JS_TEXT, got)

    @property
    def done(self) -> bool:
        return self.stable >= 1  # same real content twice in a row

    def close(self) -> None:
        if self.view is not None:
            self.view.stopLoading()
            self.view = None


def _webkit_text(url: str, timeout: float = 25) -> str:
    """innerText of ``url`` rendered by WebKit ('' if it never got through)."""
    import threading
    import time

    try:
        _wk_classes()
    except Exception:  # noqa: BLE001 — no WebKit (shouldn't happen on macOS)
        return ""
    end = time.monotonic() + timeout

    if threading.current_thread() is threading.main_thread():
        # CLI / tests: drive the run loop ourselves.
        from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop

        page = _WebKitPage(url)
        try:
            while time.monotonic() < end and not page.done:
                NSRunLoop.currentRunLoop().runMode_beforeDate_(
                    NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.4))
                page.poll()
            return page.prev or ""
        finally:
            page.close()

    # Worker thread inside the menu-bar app: WebKit lives on the main thread.
    from PyObjCTools import AppHelper

    holder, started = {}, threading.Event()

    def start():
        try:
            holder["page"] = _WebKitPage(url)
        finally:
            started.set()

    AppHelper.callAfter(start)
    if not started.wait(10) or "page" not in holder:
        return ""
    page = holder["page"]
    try:
        while time.monotonic() < end and not page.done:
            time.sleep(0.5)
            AppHelper.callAfter(page.poll)
        return page.prev or ""
    finally:
        AppHelper.callAfter(page.close)


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
