"""Rich-based presentation helpers shared by the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

BANNER = r"""[bold cyan]
  __  __ ___ ___ _____ ___ ___
 |  \/  | __| __|_   _| _ \ __|
 | |\/| | _|| _|  | | |   / _|
 |_|  |_|___|___| |_| |_|_\___|
[/bold cyan][dim]  macOS meeting recorder + transcripts[/dim]"""


def banner() -> None:
    console.print(BANNER)


def status_panel(cfg, backend: Optional[str]) -> None:
    if cfg.person_detection:
        from .transcriber import diarization_ready

        ready, reason = diarization_ready(cfg.hf_token)
        persons = "[green]on[/green]" if ready else f"[yellow]on — not ready: {reason}[/yellow]"
    else:
        persons = "[dim]off[/dim]"
    system = "[green]on[/green]" if cfg.capture_system else "[dim]off[/dim]"
    if cfg.num_speakers:
        spk = f"exactly {cfg.num_speakers}"
    elif cfg.min_speakers or cfg.max_speakers:
        spk = f"estimate {cfg.min_speakers or '?'}–{cfg.max_speakers or '?'}"
    else:
        spk = "auto-estimate"
    body = (
        f"[bold]Model[/bold]            {cfg.model}\n"
        f"[bold]Backend[/bold]          {backend or '[red]none installed[/red]'}\n"
        f"[bold]Language[/bold]         {cfg.language or 'auto-detect'}\n"
        f"[bold]System audio[/bold]     {system}\n"
        f"[bold]Person detection[/bold] {persons}\n"
        f"[bold]Speakers[/bold]         {spk}\n"
        f"[bold]Transcripts[/bold]      {cfg.transcripts_path}"
    )
    console.print(Panel(body, title="meetre", border_style="cyan", expand=False))


def list_transcripts(out_dir: Path) -> List[Path]:
    if not out_dir.exists():
        return []
    return sorted(out_dir.glob("*.md"), reverse=True)


def transcripts_table(out_dir: Path) -> None:
    files = list_transcripts(out_dir)
    if not files:
        console.print(f"[yellow]No transcripts yet in[/yellow] {out_dir}")
        return
    table = Table(title=f"Transcripts in {out_dir}", border_style="cyan")
    table.add_column("#", style="dim", justify="right")
    table.add_column("File", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Modified", style="dim")
    import datetime as _dt

    for i, f in enumerate(files, 1):
        st = f.stat()
        size = f"{st.st_size / 1024:.1f} KB"
        mtime = _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        table.add_row(str(i), f.name, size, mtime)
    console.print(table)


def devices_table(devices: List[dict]) -> None:
    table = Table(title="Audio input devices", border_style="cyan")
    table.add_column("Index", justify="right", style="bold")
    table.add_column("Name")
    table.add_column("Ch", justify="right")
    table.add_column("Use", style="green")
    for d in devices:
        use = "↩ system audio" if d["loopback"] else "🎤 mic"
        table.add_row(str(d["index"]), d["name"], str(d["channels"]), use)
    console.print(table)
    console.print(
        "[dim]No '↩ system audio' device? Install BlackHole to capture other "
        "participants:[/dim] [bold]brew install blackhole-2ch[/bold]"
    )


def info(msg: str) -> None:
    console.print(f"[cyan]›[/cyan] {msg}")


def ok(msg: str) -> None:
    console.print(f"[green]✓[/green] {msg}")


def warn(msg: str) -> None:
    console.print(f"[yellow]![/yellow] {msg}")


def error(msg: str) -> None:
    console.print(f"[red]✗[/red] {msg}")
