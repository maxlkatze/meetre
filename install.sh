#!/usr/bin/env bash
#
# meetre installer for macOS — no admin required.
#
#   * uses your existing python3 / git if present (>= 3.9),
#   * otherwise downloads a LOCAL, relocatable Python into .runtime/ (no sudo),
#   * creates a venv and installs meetre + the menu-bar app,
#   * registers a login startup item (LaunchAgent) and launches the menu bar.
#
# Run from the project folder:   bash install.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUNTIME="$ROOT/.runtime"

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
info() { printf "\033[36m›\033[0m %s\n" "$1"; }
ok()   { printf "\033[32m✓\033[0m %s\n" "$1"; }
warn() { printf "\033[33m!\033[0m %s\n" "$1"; }
die()  { printf "\033[31m✗ %s\033[0m\n" "$1"; exit 1; }

case "$(uname -m)" in
  arm64)  ARCH="aarch64" ;;
  x86_64) ARCH="x86_64" ;;
  *) die "Unsupported architecture: $(uname -m)" ;;
esac

# --- 1. Python (>= 3.9) -----------------------------------------------------
find_python() {
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,9) else 1)' 2>/dev/null; then
        command -v "$c"; return 0
      fi
    fi
  done
  return 1
}

install_local_python() {
  info "No suitable python3 found — downloading a local Python (no admin)…"
  local api url tarball
  api="https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
  url="$(curl -fsSL "$api" \
        | grep browser_download_url \
        | grep "${ARCH}-apple-darwin-install_only.tar.gz" \
        | grep -v '\.sha256' | head -1 | cut -d'"' -f4)"
  [ -n "$url" ] || die "Could not find a python-build-standalone release asset."
  mkdir -p "$RUNTIME"
  tarball="$RUNTIME/python.tar.gz"
  info "Fetching $(basename "$url")"
  curl -fSL "$url" -o "$tarball"
  tar -xzf "$tarball" -C "$RUNTIME"   # extracts to $RUNTIME/python
  rm -f "$tarball"
  [ -x "$RUNTIME/python/bin/python3" ] || die "Local Python install failed."
  echo "$RUNTIME/python/bin/python3"
}

if PY="$(find_python)"; then
  ok "Python: $PY ($("$PY" --version 2>&1))"
else
  PY="$(install_local_python)"
  ok "Local Python: $PY ($("$PY" --version 2>&1))"
fi

# --- 2. Git (only checked; needed for auto-update) --------------------------
if git --version >/dev/null 2>&1; then
  ok "Git: $(command -v git) ($(git --version))"
else
  warn "Git not found. Auto-update needs git."
  info "Triggering the Xcode Command Line Tools installer (no admin password)…"
  xcode-select --install 2>/dev/null || true
  warn "Finish the popup, then re-run install.sh. Continuing without git for now."
fi

# --- 3. Virtual environment + dependencies ---------------------------------
info "Creating virtual environment (.venv)…"
"$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip wheel
info "Installing meetre + menu-bar app (this downloads MLX models on first use)…"
.venv/bin/pip install -q -e ".[menubar]"
ok "meetre installed"
info "Optional: speaker detection →  .venv/bin/pip install -e '.[persons]'"

# --- 4. Add meetre to your shell PATH --------------------------------------
mkdir -p "$HOME/.local/bin"
ln -sf "$ROOT/.venv/bin/meetre" "$HOME/.local/bin/meetre"
ok "Linked 'meetre' into ~/.local/bin"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) : ;;  # already on PATH
  *)
    RC="$HOME/.zshrc"; [ "${SHELL##*/}" = "bash" ] && RC="$HOME/.bashrc"
    if ! grep -qs '.local/bin' "$RC" 2>/dev/null; then
      printf '\n# meetre\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$RC"
      info "Added ~/.local/bin to PATH in $RC (open a new terminal to pick it up)."
    fi
    ;;
esac

# --- 5. HuggingFace token (optional, for speaker detection) ----------------
echo
info "Speaker detection (who-said-what) needs a free HuggingFace token."
info "Accept the terms at huggingface.co/pyannote/speaker-diarization-3.1 first."
printf "Paste your HuggingFace token (or press Enter to skip): "
read -r HF_TOKEN || true
if [ -n "${HF_TOKEN:-}" ]; then
  .venv/bin/meetre config hf_token "$HF_TOKEN" >/dev/null && ok "HuggingFace token saved"
else
  info "Skipped — add later with:  meetre config hf_token <token>"
fi

# --- 6. Startup item + launch the menu bar ---------------------------------
info "Registering login startup item and launching the menu bar…"
.venv/bin/python -c "from meetre import autostart; autostart.enable()"
ok "Menu-bar app started and set to launch at login (✦ icon, top-right)."

# --- 7. Done ---------------------------------------------------------------
echo
bold "meetre is ready."
echo "  • Menu bar:  the ✦ icon (top-right). Click → ● Record…"
echo "  • CLI:       .venv/bin/meetre        (add an alias if you like)"
echo "  • Update:    .venv/bin/meetre update   (or the menu's 'Check for updates')"
if ! git remote >/dev/null 2>&1 || [ -z "$(git remote 2>/dev/null)" ]; then
  echo
  warn "No git remote is configured, so auto-update can't pull yet."
  echo "  Add one once you've pushed meetre somewhere, e.g.:"
  echo "    git remote add origin <your-repo-url>"
fi
