#!/usr/bin/env bash
#
# meetre installer for macOS — no admin required.
#
#   * uses your existing python3 / git if present (>= 3.11),
#   * otherwise downloads a LOCAL, relocatable Python into .runtime/ (no sudo),
#   * creates a venv and installs the meetre engine,
#   * builds the native app into ~/Applications/Meetre.app and opens it —
#     its onboarding takes it from there (options, login item, models).
#
# Run from the project folder:   bash install.sh [--persons]
#   --persons  also install speaker detection (pyannote).
#
# Never interactive. Progress goes to stdout as "::meetre-step <n>/<total>
# <label>" lines (Meetre.app's onboarding shows them); human-readable logs
# go to stderr. MEETRE_NO_LAUNCH=1 (set by the app when it runs this
# itself) skips building/opening the app.
#
set -euo pipefail

PERSONS=0
for arg in "$@"; do
  case "$arg" in
    --persons) PERSONS=1 ;;
  esac
done
STEPS=7
step() { printf "::meetre-step %s/%s %s\n" "$1" "$STEPS" "$2"; }

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

step 1 "Preparing Python"
if PY="$(find_python)"; then
  ok "Python: $PY ($("$PY" --version 2>&1))"
else
  PY="$(install_local_python)"
  ok "Local Python: $PY ($("$PY" --version 2>&1))"
fi

# --- 2. Git (existing, else install a LOCAL git without admin) -------------
REPO_URL="https://github.com/maxlkatze/meetre.git"
step 2 "Getting update tools"
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
# Re-running install.sh is the common "update" path, so this MUST be safe to run
# over an existing install: reuse a working .venv (keeping its downloaded models
# and state) instead of recreating it. Only build a fresh venv when none exists
# or the current one is broken — and then move the old one aside to .venv.old
# rather than clobbering it in place.
step 3 "Creating a home for meetre"
if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import sys" >/dev/null 2>&1; then
  ok "Reusing existing virtual environment (.venv)"
else
  if [ -e ".venv" ]; then
    warn "Existing .venv is broken — moving it aside to .venv.old"
    rm -rf .venv.old
    mv .venv .venv.old
  fi
  info "Creating virtual environment (.venv)…"
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install -q --upgrade pip wheel
step 4 "Installing the engine"
info "Installing meetre + menu-bar app (this downloads MLX models on first use)…"
EXTRAS="menubar"; [ "$PERSONS" = 1 ] && EXTRAS="menubar,persons"
.venv/bin/pip install -q -e ".[$EXTRAS]"
ok "meetre installed"
[ "$PERSONS" = 1 ] || info "Optional: speaker detection →  .venv/bin/pip install -e '.[persons]'"

# --- 4. Add meetre to your shell PATH --------------------------------------
step 5 "Linking the meetre command"
mkdir -p "$HOME/.local/bin"
ln -sf "$ROOT/.venv/bin/meetre" "$HOME/.local/bin/meetre"
ok "Linked 'meetre' into ~/.local/bin"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) : ;;  # already on PATH
  *)
    RC="$HOME/.zshrc"; [ "${SHELL##*/}" = "bash" ] && RC="$HOME/.bashrc"
    if ! grep -qsF 'export PATH="$HOME/.local/bin:$PATH"' "$RC" 2>/dev/null; then
      printf '\n# meetre\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$RC"
      info "Added ~/.local/bin to PATH in $RC (open a new terminal to pick it up)."
    fi
    ;;
esac

# --- 5. HuggingFace token (optional, for speaker detection) ----------------
# Asked in the app's onboarding; the app passes it through when it runs us.
if [ -n "${MEETRE_HF_TOKEN:-}" ]; then
  .venv/bin/meetre config hf_token "$MEETRE_HF_TOKEN" >/dev/null && ok "HuggingFace token saved"
fi

# --- 6. Make this a git checkout (so auto-update works) --------------------
# Tarball/curl installs aren't git repos; turn the folder into one tracking main.
step 6 "Turning on auto-update"
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

# --- 7. The app ------------------------------------------------------------
# Never quit meetre under a running recording (the engine keeps this file
# while recording/processing; a stale one from a dead engine is ignored).
BUSY="$HOME/.cache/meetre/busy"
if [ -f "$BUSY" ] && kill -0 "$(cut -d' ' -f1 "$BUSY")" 2>/dev/null; then
  die "meetre is busy ($(cut -d' ' -f2- "$BUSY")). Stop the recording and wait until it's processed, then run this again."
fi

# Retire the legacy Python menu bar: the native app replaces it (its
# onboarding registers Meetre.app as the login item).
LEGACY_AGENT="$HOME/Library/LaunchAgents/com.agendaro.meetre.menubar.plist"
if [ -f "$LEGACY_AGENT" ] && ! grep -q "Meetre.app" "$LEGACY_AGENT"; then
  launchctl unload "$LEGACY_AGENT" >/dev/null 2>&1 || true
  rm -f "$LEGACY_AGENT"
  ok "Removed the old menu-bar login item"
fi
pkill -f "meetre.menubar" >/dev/null 2>&1 || true

if [ "${MEETRE_NO_LAUNCH:-0}" = 1 ]; then
  ok "Engine ready."
  printf "::meetre-done\n"
  exit 0
fi

step 7 "Building the app"
# /usr/bin/swift always exists as a shim; xcrun tells us if it's real.
xcrun --find swift >/dev/null 2>&1 || die "Building Meetre.app needs Apple's Swift tools.
  Install them (free, a few minutes):   xcode-select --install
  Then run this installer again."
bash "$ROOT/mac/build.sh" >&2 || die "Building Meetre.app failed (see above)."
APP_DIR="$HOME/Applications"
mkdir -p "$APP_DIR"
# Unchanged app → leave the installed one (and the running app) alone. Each
# ad-hoc-signed build is a new identity to macOS, so replacing it needlessly
# drops the Screen Recording / Microphone permissions it was granted.
NEW="$ROOT/mac/build/Meetre.app/Contents"
OLD="$APP_DIR/Meetre.app/Contents"
if [ -d "$OLD" ] && cmp -s "$NEW/MacOS/Meetre" "$OLD/MacOS/Meetre" \
   && cmp -s "$NEW/Info.plist" "$OLD/Info.plist"; then
  "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister" -u "$ROOT/mac/build/Meetre.app" >/dev/null 2>&1 || true
  rm -rf "$ROOT/mac/build/Meetre.app"
  ok "Meetre.app is up to date"
  printf "::meetre-done\n"
  pgrep -f "/Meetre.app/Contents/MacOS/Meetre" >/dev/null || open "$APP_DIR/Meetre.app"
  exit 0
fi
# Quit a running copy first: `open` on the same bundle id would otherwise just
# bring the old build to the front instead of launching the new one.
osascript -e 'tell application id "net.cubedpixels.meetre.app" to quit' >/dev/null 2>&1 || true
for _ in 1 2 3 4 5 6 7 8 9 10; do
  pgrep -f "/Meetre.app/Contents/MacOS/Meetre" >/dev/null || break
  sleep 0.5
done
pkill -f "/Meetre.app/Contents/MacOS/Meetre" >/dev/null 2>&1 || true
rm -rf "$APP_DIR/Meetre.app"
cp -R "$ROOT/mac/build/Meetre.app" "$APP_DIR/Meetre.app"
ok "Installed $APP_DIR/Meetre.app"

# Exactly one meetre in Launchpad/Spotlight: unregister and delete the legacy
# Python wrapper (meetre.app, regenerated only by the old menu bar) and the
# build output, then register the installed copy.
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
for old in "$ROOT/meetre.app" "$ROOT/mac/build/Meetre.app"; do
  if [ -d "$old" ]; then
    "$LSREGISTER" -u "$old" >/dev/null 2>&1 || true
    rm -rf "$old"
  fi
done
"$LSREGISTER" -f "$APP_DIR/Meetre.app" >/dev/null 2>&1 || true
ok "Meetre is in Launchpad"

printf "::meetre-done\n"
bold "meetre is ready — opening the app."
open "$APP_DIR/Meetre.app"
