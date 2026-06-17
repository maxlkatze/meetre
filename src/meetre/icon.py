"""Menu-bar / notification icon for the macOS app.

Renders an SF Symbol once into a small PNG in the config dir so it can be used
both as the status-bar image (a real image icon, template-tinted to match the
menu bar) and as the icon shown on notifications. No image asset ships with the
package and no extra dependency is needed — it is drawn on the fly via AppKit.

Everything is best-effort: if AppKit/SF Symbols are unavailable, the callers
fall back to the plain Unicode glyph they used before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import CONFIG_DIR

# The status-bar symbol; "sparkles" mirrors the ✦ glyph the app used before.
_SYMBOL = "sparkles"
_ICON_PATH = CONFIG_DIR / "icon.png"
_SIZE = 36  # points; AppKit renders at the bar's natural resolution

_cached: Optional[str] = None
_tried = False


def _render(path: Path) -> bool:
    """Draw the SF Symbol into a PNG at ``path``. Returns True on success."""
    try:
        from AppKit import (
            NSBitmapImageRep,
            NSImage,
            NSMakeRect,
            NSMakeSize,
        )

        img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(_SYMBOL, None)
        if img is None:
            return False
        img.setSize_(NSMakeSize(_SIZE, _SIZE))
        img.lockFocus()
        rep = NSBitmapImageRep.alloc().initWithFocusedViewRect_(
            NSMakeRect(0, 0, _SIZE, _SIZE)
        )
        img.unlockFocus()
        if rep is None:
            return False
        # 4 == NSBitmapImageFileType.PNG (NSPNGFileType)
        data = rep.representationUsingType_properties_(4, None)
        if data is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(data.writeToFile_atomically_(str(path), True))
    except Exception:  # noqa: BLE001
        return False


def icon_path() -> Optional[str]:
    """Path to the app icon PNG, rendering it once if needed; None if it can't
    be produced on this machine."""
    global _cached, _tried
    if _cached is not None:
        return _cached
    if _ICON_PATH.exists():
        _cached = str(_ICON_PATH)
        return _cached
    if _tried:
        return None
    _tried = True
    if _render(_ICON_PATH):
        _cached = str(_ICON_PATH)
        return _cached
    return None
