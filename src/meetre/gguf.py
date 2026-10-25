"""The local LLM: GGUF models run by llama.cpp's ``llama-server``.

Summaries, titles, speaker names and chat all run here (MLX is only used for
speech recognition now). Measured on real meetings (2026-10-25, K2 Horizon
7B): GGUF Q8 summaries were as fast as MLX 4-bit or faster, titles 2-3×
faster (the transcript stays in the server's prompt cache), and only
llama-server can force structured output (``json_schema``) — MLX name
detection answered in shapes the code had to throw away, and once looped to
the token cap (95 s).

- **Models** live in ``~/models`` (``Config.gguf_folder``), one folder per
  family, like LocalCompanion: catalog models are downloaded there (pinned
  commit, SHA-256 checked), and any other ``.gguf`` in the folder is usable
  too. A file meetre downloaded is recorded in ``.meetre-downloads.json`` —
  only those can be deleted from the app, never the user's own.
- **llama-server**: on PATH or a usual build spot, else one pinned, tested
  official macOS arm64 build is downloaded once into ``~/models/.llama.cpp``
  (SHA-256 checked).
- **Server**: one at a time on a free 127.0.0.1 port, tied to our process
  (it dies with us), stopped after IDLE_SECONDS. Flags from LocalCompanion's
  measurements: flash attention, few context checkpoints and a small host
  prompt cache (the defaults grew llama-server to 12.6 GB).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tarfile
import threading
import time
import urllib.request
from collections import namedtuple
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

PREFIX = "gguf:"
IDLE_SECONDS = 120
# 16k: K2 Q8 needs 11.4 GB instead of 13.6 GB at 32k, same summary (25 min
# meeting); the longest prompts are a 24k-char chunk / 40k-char chat
# transcript (~11k tokens). An 8-bit KV cache saved 1 GB more but changed
# the summary and wrote slower — not used.
CONTEXT = 16384
_SEARCH = ["~/llama.cpp/build/bin/llama-server", "/opt/homebrew/bin/llama-server",
           "/usr/local/bin/llama-server"]
# llama.cpp publishes a build almost every hour; meetre downloads one tested
# build (b11887: same speed as b11481 on a 25-min summary, 34.8 tok/s) so
# every install runs the same code. Bump after a speed + quality check.
LLAMA_BUILD = "b11887"
LLAMA_SHA256 = "13d5a906ab066f57a937a24e345eb541cb423b249c2254ed83cb2a24bba3f502"
LLAMA_SIZE = 16287809
MANIFEST = ".meetre-downloads.json"

# A downloadable model: HF repo + file, pinned commit, folder under ~/models.
GGUFSpec = namedtuple("GGUFSpec", "repo file revision folder")
CATALOG = {
    "k2-horizon-7b": GGUFSpec("IFM/K2-Horizon-7B-GGUF", "K2-Horizon-7B-Q8_0.gguf",
                              "7a75dbda05e3d440351543131ff3f7fa5687e879", "K2-Horizon-7B"),
    "gemma4-12b": GGUFSpec("unsloth/gemma-4-12B-it-qat-GGUF", "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
                           "980b060c40a8539ac159e0501a3e0f66a6365af3", "gemma-4-qat"),
    "gemma4-e4b": GGUFSpec("unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
                           "8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265", "gemma-4-qat"),
    "qwen3.5-9b": GGUFSpec("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-UD-Q5_K_XL.gguf",
                           "3885219b6810b007914f3a7950a8d1b469d598a5", "qwen3.5"),
    "qwen3.5-4b": GGUFSpec("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-UD-Q4_K_XL.gguf",
                           "e87f176479d0855a907a41277aca2f8ee7a09523", "qwen3.5"),
}

ProgressCB = Callable[[float, int, int], None]


def is_gguf(model: Optional[str]) -> bool:
    return bool(model) and model.startswith(PREFIX)


def models_dir() -> Path:
    try:
        from .config import Config

        return Path(Config.load().gguf_folder or "~/models").expanduser()
    except Exception:  # noqa: BLE001
        return Path("~/models").expanduser()


# ---------------------------------------------------------------------------
# Finding and downloading models
# ---------------------------------------------------------------------------

def scan(folder: Optional[str] = None) -> List[Dict[str, object]]:
    """GGUF models under ``folder`` (recursive). Vision projectors
    (``mmproj``), MTP draft heads and later parts of split files are skipped."""
    root = Path(folder).expanduser() if folder else models_dir()
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.rglob("*.gguf")):
        name = p.name.lower()
        if "mmproj" in name or name.startswith("mtp-") or any(
                part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if "-of-" in p.stem and not p.stem.split("-of-")[0].endswith("00001"):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        out.append({"id": PREFIX + str(p), "label": p.stem, "size_gb": round(size / 1e9, 1),
                    "folder": p.parent.name, "path": p})
    return out


def path_for(alias: str) -> Path:
    """Where a catalog model is (any file of that name under the models
    folder — the user may already have it), else where it will go."""
    spec = CATALOG[alias]
    root = models_dir()
    if root.is_dir():
        for p in root.rglob(spec.file):
            if not any(part.startswith(".") for part in p.relative_to(root).parts):
                return p
    return root / spec.folder / spec.file


def _manifest() -> Dict[str, dict]:
    try:
        return json.loads((models_dir() / MANIFEST).read_text())
    except (OSError, ValueError):
        return {}


def _save_manifest(data: Dict[str, dict]) -> None:
    p = models_dir() / MANIFEST
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)


def downloaded() -> List[Dict[str, object]]:
    """Files meetre downloaded that are still there: ``{alias, path, size}``."""
    out = []
    for path, info in _manifest().items():
        p = Path(path)
        if p.is_file():
            out.append({"alias": info.get("alias"), "path": p, "size": p.stat().st_size})
    return out


def delete(path: str) -> int:
    """Delete a file meetre downloaded (never the user's own). Freed bytes."""
    data = _manifest()
    if path not in data:
        raise RuntimeError("Only models meetre downloaded can be deleted here.")
    p = Path(path)
    freed = p.stat().st_size if p.is_file() else 0
    p.unlink(missing_ok=True)
    data.pop(path)
    _save_manifest(data)
    try:
        p.parent.rmdir()  # only if now empty
    except OSError:
        pass
    return freed


def _hf_sha256(spec: GGUFSpec) -> Tuple[Optional[str], Optional[int]]:
    """The LFS SHA-256 and size of a catalog file, from the Hub API."""
    url = f"https://huggingface.co/api/models/{spec.repo}/tree/{spec.revision}"
    with urllib.request.urlopen(url, timeout=30) as r:
        for f in json.loads(r.read()):
            if f.get("path") == spec.file:
                lfs = f.get("lfs") or {}
                return lfs.get("oid"), lfs.get("size") or f.get("size")
    return None, None


def _download(url: str, dest: Path, size: Optional[int], sha256: Optional[str],
              on_progress: Optional[ProgressCB]) -> None:
    """Resumable download to ``dest`` (via ``.part``), checked by size/SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:  # server ignored the range: start over
            have = 0
        total = size or (have + int(r.headers.get("Content-Length") or 0))
        with open(part, "ab" if have else "wb") as f:
            done, last = have, 0.0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if on_progress and time.monotonic() - last > 0.25:
                    last = time.monotonic()
                    on_progress(done / total if total else 0.0, done, total)
    if size and part.stat().st_size != size:
        raise RuntimeError(f"Download incomplete: {dest.name}")
    if sha256:
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {dest.name} — download removed.")
    part.replace(dest)
    if on_progress:
        on_progress(1.0, size or 0, size or 0)


def ensure(alias: str, on_progress: Optional[ProgressCB] = None) -> Path:
    """The local file of catalog model ``alias``, downloaded if missing."""
    path = path_for(alias)
    if path.is_file():
        return path
    spec = CATALOG[alias]
    sha, size = _hf_sha256(spec)
    url = f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{spec.file}"
    _download(url, path, size, sha, on_progress)
    data = _manifest()
    data[str(path)] = {"alias": alias, "repo": spec.repo, "revision": spec.revision}
    _save_manifest(data)
    return path


# ---------------------------------------------------------------------------
# llama-server
# ---------------------------------------------------------------------------

def server_binary() -> Optional[str]:
    """An installed llama-server (PATH is empty under launchd, so the usual
    spots are checked too), else the one meetre downloaded, else None."""
    found = shutil.which("llama-server")
    if found:
        return found
    for p in _SEARCH:
        p = os.path.expanduser(p)
        if os.access(p, os.X_OK):
            return p
    ours = sorted((models_dir() / ".llama.cpp").glob("llama-b*/llama-server"))
    return str(ours[-1]) if ours else None


def ensure_server(on_progress: Optional[ProgressCB] = None) -> str:
    """llama-server, downloading the pinned official macOS arm64 build on
    first use (SHA-256 checked)."""
    found = server_binary()
    if found:
        return found
    name = f"llama-{LLAMA_BUILD}-bin-macos-arm64.tar.gz"
    base = models_dir() / ".llama.cpp"
    tgz = base / name
    _download(f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/{name}",
              tgz, LLAMA_SIZE, LLAMA_SHA256, on_progress)
    with tarfile.open(tgz) as t:
        t.extractall(base, filter="data")
    tgz.unlink(missing_ok=True)
    found = server_binary()
    if not found:
        raise RuntimeError("llama.cpp download didn't contain llama-server.")
    return found


def _tie_to_us(proc) -> None:
    """llama-server (~8 GB) dies with the engine, even if the engine is
    killed hard: a tiny shell loop waits for our pid to go, then stops it
    (only if WE went — never a reused pid). From LocalCompanion."""
    me = os.getpid()
    subprocess.Popen(["/bin/sh", "-c",
                      f"while kill -0 {me} 2>/dev/null && kill -0 {proc.pid} 2>/dev/null; "
                      f"do sleep 2; done; kill -0 {me} 2>/dev/null || kill {proc.pid} 2>/dev/null"],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


TIMINGS: List[dict] = []  # the last requests' prompt/cache/speed numbers (perf logs)


class Server:
    """One running llama-server for one GGUF file."""

    def __init__(self, path: str):
        self.path = path
        self.port = _free_port()
        self.no_system = False     # template has no system role: fold it in
        self.first_system = False  # template takes a system message only first
        binary = ensure_server()
        args = [binary, "-m", path, "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(CONTEXT), "-ngl", "999", "-fa", "on", "--parallel", "1",
                "--jinja", "--no-webui",
                # LocalCompanion measured the defaults (32 context checkpoints,
                # 8 GB host prompt cache) growing llama-server to 12.6 GB.
                "--ctx-checkpoints", "2", "--cache-ram", "512"]
        # A .jinja next to the model replaces its own template (as in
        # LocalCompanion: fixes for templates that always think, etc.).
        tmpl = sorted(Path(path).parent.glob("*.jinja"))
        if tmpl:
            args += ["--chat-template-file", str(tmpl[0])]
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     text=True)
        _tie_to_us(self.proc)
        self._log: List[str] = []
        threading.Thread(target=self._drain, daemon=True).start()
        self.last_used = time.monotonic()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited: " + " ".join(self._log[-5:]))
            try:
                if self._get("/health").get("status") == "ok":
                    return
            except Exception:  # noqa: BLE001 — still loading
                pass
            time.sleep(0.2)
        self.stop()
        raise RuntimeError("llama-server didn't come up within 3 minutes")

    def _drain(self):
        for line in self.proc.stderr:
            self._log = (self._log + [line.strip()])[-50:]

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self._url(path), timeout=2) as r:
            return json.loads(r.read())

    def post(self, path: str, body: dict, stream: bool = False):
        self.last_used = time.monotonic()
        req = urllib.request.Request(self._url(path), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            raise ServerError(e.read().decode("utf-8", "replace")) from None
        if not stream:
            with r:
                return json.loads(r.read())
        return r

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def fit_roles(self, messages: List[dict]) -> List[dict]:
        if self.no_system:
            return _fold_system(messages)
        if self.first_system:
            return messages[:1] + [{**m, "role": "user"} if m["role"] == "system" else m
                                   for m in messages[1:]]
        return messages


class ServerError(RuntimeError):
    pass


def _fold_system(messages: List[dict]) -> List[dict]:
    """For templates without a system role: system text goes into the next
    user message; consecutive same roles are merged (LocalCompanion)."""
    out, pending = [], []
    for m in messages:
        if m["role"] == "system":
            pending.append(m["content"])
            continue
        content = m["content"]
        if m["role"] == "user" and pending:
            content = "\n\n".join(pending) + "\n\n" + content
            pending = []
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n" + content}
        else:
            out.append({"role": m["role"], "content": content})
    return out


_lock = threading.Lock()
_current: Dict[str, Server] = {}


def load(model: str) -> Server:
    """The running server for ``gguf:<path>`` (started if needed)."""
    path = model[len(PREFIX):]
    if not Path(path).is_file():
        raise RuntimeError(f"GGUF model not found: {path}")
    with _lock:
        srv = _current.get("server")
        if srv is None or srv.path != path or srv.proc.poll() is not None:
            if srv is not None:
                srv.stop()
            srv = Server(path)
            _current["server"] = srv
            threading.Thread(target=_idle_watch, args=(srv,), daemon=True).start()
    return srv


def _idle_watch(srv: Server) -> None:
    while srv.proc.poll() is None:
        time.sleep(5)
        if time.monotonic() - srv.last_used > IDLE_SECONDS:
            with _lock:
                if _current.get("server") is srv:
                    _current.pop("server")
                srv.stop()


def release() -> None:
    with _lock:
        srv = _current.pop("server", None)
    if srv is not None:
        srv.stop()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_THINK_END = re.compile(r"</[^>]*think[^>]*>")


def chat(server: Server, messages: List[dict], max_tokens: int, *,
         template_kwargs: Optional[dict] = None, json_schema: Optional[dict] = None,
         repetition_penalty: Optional[float] = None,
         sampling: Optional[dict] = None) -> Iterator[Tuple[str, str]]:
    """Stream a chat reply as ``("think", text)`` / ``("text", text)`` pieces.

    The server applies the model's own chat template and separates the
    reasoning. Greedy by default (summaries: sampling dropped sections, see
    CLAUDE.md). Template quirks are handled as in LocalCompanion: no system
    role → folded in; system only first → later ones become user turns; a
    malformed think-closing tag (K2) → the rest of the "reasoning" is the
    answer."""
    body = {"messages": server.fit_roles(messages), "max_tokens": max_tokens,
            "stream": True, "cache_prompt": True, "temperature": 0,
            "chat_template_kwargs": template_kwargs or {}}
    if repetition_penalty:
        body.update(repeat_penalty=repetition_penalty, repeat_last_n=256)
    if json_schema is not None:
        body["json_schema"] = json_schema
    body.update(sampling or {})
    try:
        r = server.post("/v1/chat/completions", body, stream=True)
    except ServerError as e:
        msg = str(e)
        if re.search(r"roles (are supported|must alternate)|[Ss]ystem role not supported", msg) \
                and not server.no_system:
            server.no_system = True
            yield from chat(server, messages, max_tokens, template_kwargs=template_kwargs,
                            json_schema=json_schema, repetition_penalty=repetition_penalty,
                            sampling=sampling)
            return
        if "System message must be at the beginning" in msg and not server.first_system:
            server.first_system = True
            yield from chat(server, messages, max_tokens, template_kwargs=template_kwargs,
                            json_schema=json_schema, repetition_penalty=repetition_penalty,
                            sampling=sampling)
            return
        raise
    thinking, leaked = "", False
    with r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("timings"):
                TIMINGS[:] = (TIMINGS + [chunk["timings"]])[-20:]
            for c in chunk.get("choices") or []:
                delta = c.get("delta") or {}
                server.last_used = time.monotonic()
                if delta.get("content"):
                    yield "text", delta["content"]
                elif delta.get("reasoning_content"):
                    piece = delta["reasoning_content"]
                    if leaked:
                        yield "text", piece
                        continue
                    thinking += piece
                    m = _THINK_END.search(thinking)
                    if m:
                        leaked = True
                        if thinking[m.end():]:
                            yield "text", thinking[m.end():]
                    else:
                        yield "think", piece


class Tokenizer:
    """Chat templating for raw-prompt use (chat with tools parses the model's
    own tool-call markup from the raw text, so it renders the prompt itself)."""

    def __init__(self, server: Server):
        self.server = server

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False,
                            **kwargs) -> str:
        body = {"messages": self.server.fit_roles(messages)}
        extra = {k: v for k, v in kwargs.items() if k != "tools"}
        if extra:
            body["chat_template_kwargs"] = extra
        if kwargs.get("tools"):
            body["tools"] = kwargs["tools"]
        return self.server.post("/apply-template", body)["prompt"]


def complete(server: Server, prompt: str, max_tokens: int) -> Iterator[str]:
    """Text pieces of a greedy completion of a raw ``prompt``."""
    body = {"prompt": prompt, "n_predict": max_tokens, "stream": True,
            "temperature": 0, "cache_prompt": True}
    with server.post("/completion", body, stream=True) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:])
            piece = data.get("content") or ""
            if piece:
                server.last_used = time.monotonic()
                yield piece
            if data.get("stop"):
                break
