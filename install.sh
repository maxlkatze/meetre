#!/usr/bin/env bash
#
# meetre installer for macOS — no admin required.
#
#   * uses your existing python3 / git if present (>= 3.11),
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

# Log helpers write to stderr so they never pollute command substitution
# (e.g. PY="$(install_local_python)" must capture ONLY the python path).
bold() { printf "\033[1m%s\033[0m\n" "$1" >&2; }
info() { printf "\033[36m›\033[0m %s\n" "$1" >&2; }
ok()   { printf "\033[32m✓\033[0m %s\n" "$1" >&2; }
warn() { printf "\033[33m!\033[0m %s\n" "$1" >&2; }
die()  { printf "\033[31m✗ %s\033[0m\n" "$1" >&2; exit 1; }

case "$(uname -m)" in
  arm64)  ARCH="aarch64" ;;
  x86_64) ARCH="x86_64" ;;
  *) die "Unsupported architecture: $(uname -m)" ;;
esac

# --- 0. Refuse TCC-protected install locations -----------------------------
# macOS guards ~/Downloads, ~/Desktop and ~/Documents (TCC). The login item we
# register is launched by launchd, which does NOT inherit Terminal's access to
# those folders — so it can't even read the venv's pyvenv.cfg and crashes at
# startup (the menu-bar icon just silently disappears). Bail out early with a
# fix rather than install something that will vanish on next login.
case "$ROOT/" in
  "$HOME/Downloads/"*|"$HOME/Desktop/"*|"$HOME/Documents/"*)
    SAFE="$HOME/$(basename "$ROOT")"
    die "meetre is in a macOS-protected folder and would crash at login:
    $ROOT

  ~/Downloads, ~/Desktop and ~/Documents block background apps (login items)
  from reading their files, so the menu bar would just disappear.

  Move it out and re-run, e.g.:
    mv \"$ROOT\" \"$SAFE\"
    cd \"$SAFE\" && bash install.sh"
    ;;
esac

# --- 1. Python (>= 3.11) ----------------------------------------------------
# meetre needs Python >= 3.11 (mlx-lm 0.31.3+ and the current model lineup).
# If no recent enough python3 is on PATH we fetch a local, relocatable CPython.
PY_SERIES="3.12"   # version of the auto-downloaded local Python

find_python() {
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,11) else 1)' 2>/dev/null; then
        command -v "$c"; return 0
      fi
    fi
  done
  return 1
}

install_local_python() {
  info "No suitable python3 (>= 3.11) found — downloading a local Python ${PY_SERIES} (no admin)…"
  local api url tarball
  api="https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
  # Pick the newest CPython ${PY_SERIES}.x install_only build for this arch.
  url="$(curl -fsSL "$api" \
        | grep browser_download_url \
        | grep "cpython-${PY_SERIES}\." \
        | grep "${ARCH}-apple-darwin-install_only.tar.gz" \
        | grep -v '\.sha256' | head -1 | cut -d'"' -f4)"
  # Fall back to whatever install_only build is newest if the series isn't found.
  [ -n "$url" ] || url="$(curl -fsSL "$api" \
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

# --- 2. Git (existing, else install a LOCAL git without admin) -------------
REPO_URL="https://github.com/maxlkatze/meetre.git"
GIT=""
if git --version >/dev/null 2>&1; then
  GIT="$(command -v git)"
  ok "Git: $GIT ($(git --version))"
else
  info "No git found — installing a local git without admin (via micromamba)…"
  case "$ARCH" in aarch64) MM_ARCH="osx-arm64" ;; x86_64) MM_ARCH="osx-64" ;; esac
  export MAMBA_ROOT_PREFIX="$RUNTIME/mamba"
  mkdir -p "$RUNTIME"
  if curl -fsSL "https://micro.mamba.pm/api/micromamba/${MM_ARCH}/latest" \
       | tar -xj -C "$RUNTIME" bin/micromamba 2>/dev/null \
     && "$RUNTIME/bin/micromamba" create -y -q -p "$RUNTIME/conda" -c conda-forge git >/dev/null 2>&1 \
     && [ -x "$RUNTIME/conda/bin/git" ]; then
    GIT="$RUNTIME/conda/bin/git"
    export PATH="$RUNTIME/conda/bin:$PATH"
    ok "Local git installed: $GIT ($("$GIT" --version))"
  else
    warn "Could not install git automatically — auto-update will be disabled."
  fi
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

# --- 6. Make this a git checkout (so auto-update works) --------------------
# Tarball/curl installs aren't git repos; turn the folder into one tracking main.
if [ -n "$GIT" ] && [ ! -d ".git" ]; then
  info "Linking this install to $REPO_URL for auto-update…"
  if "$GIT" init -q \
     && "$GIT" remote add origin "$REPO_URL" \
     && "$GIT" fetch -q --depth 1 origin main \
     && "$GIT" reset -q --hard FETCH_HEAD \
     && "$GIT" branch -M main \
     && "$GIT" branch --set-upstream-to=origin/main main >/dev/null 2>&1; then
    ok "Auto-update enabled (origin/main)"
  else
    warn "Could not link to the remote; auto-update disabled (install still works)."
  fi
fi

# --- 7. Startup item + launch the menu bar ---------------------------------
info "Registering login startup item and launching the menu bar…"
.venv/bin/python -c "from meetre import autostart; autostart.enable()"
ok "Menu-bar app started and set to launch at login (✦ icon, top-right)."

# --- 8. Done ---------------------------------------------------------------
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
