"""DocLAMAR command line — same engine as the desktop app.

    python cli.py index  "C:/Users/me/Documents/papers"
    python cli.py ask    "C:/Users/me/Documents/papers" "What loss function was used?"
    python cli.py chat   "C:/Users/me/Documents/papers"        # folder conversation
    python cli.py chat   "C:/Users/me/Documents/papers/x.pdf"  # one document
    python cli.py status "C:/Users/me/Documents/papers"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


def _engine():
    from doclamar.engine import Engine

    return Engine()


def _sync(engine, folder: str) -> None:
    from doclamar.indexer import IndexJob

    job = IndexJob(folder=os.path.abspath(folder))
    worker = threading.Thread(target=engine.indexer.sync_folder, args=(folder, job), daemon=True)
    worker.start()
    with console.status("Scanning…") as status:
        while worker.is_alive():
            if job.state == "indexing":
                name = os.path.basename(job.current_file or "")
                detail = f" ({job.detail})" if job.detail else ""
                status.update(f"Indexing {job.processed}/{job.to_index}: {name}{detail}")
            time.sleep(0.2)
    _print_status(engine, folder)


def _print_status(engine, folder: str) -> None:
    s = engine.store.folder_summary(folder)
    console.print(f"[green]{s['files_indexed']}[/green] of {s['files_total']} files indexed, "
                  f"{s['chunks']} chunks (last sync: {s['synced_at'] or 'never'})")
    for issue in s["issues"]:
        console.print(f"  [yellow]{issue['status']}[/yellow] {issue['path']} — {issue['error'] or ''}")


def _show(result: dict) -> None:
    console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))
    cites = [c for c in result["citations"] if c["cited"]] or result["citations"]
    if cites:
        table = Table(show_header=True, header_style="dim")
        for col in ("#", "File", "Where", "Snippet"):
            table.add_column(col)
        for c in cites:
            where = " · ".join(x for x in (c["pages"], c["section"]) if x)
            table.add_row(str(c["ref"]), c["file"], where, c["snippet"][:90].replace("\n", " ") + "…")
        console.print(table)


def _scope_for(engine, target: str):
    from doclamar.store import Scope

    if os.path.isfile(target):
        with console.status("Indexing document…"):
            info = engine.indexer.index_file(target)
        if info["status"] != "indexed":
            console.print(f"[red]Cannot chat with this file: {info.get('error') or info['status']}[/red]")
            sys.exit(1)
        return Scope(files=[target])
    if os.path.isdir(target):
        _sync(engine, target)
        return Scope(folder=target)
    console.print(f"[red]Not found: {target}[/red]")
    sys.exit(1)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="DocLAMAR — ask questions about your documents")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="index (or re-sync) a folder").add_argument("folder")
    sub.add_parser("status", help="show index status for a folder").add_argument("folder")
    ask = sub.add_parser("ask", help="ask one question about a folder")
    ask.add_argument("folder")
    ask.add_argument("question")
    sub.add_parser("chat", help="interactive chat with a folder or one file").add_argument("target")
    args = parser.parse_args()

    engine = _engine()
    try:
        if args.cmd == "index":
            _sync(engine, args.folder)
        elif args.cmd == "status":
            _print_status(engine, args.folder)
        elif args.cmd == "ask":
            scope = _scope_for(engine, args.folder)
            with console.status("Thinking…"):
                result = engine.answer(args.question, scope)
            _show(result)
        elif args.cmd == "chat":
            scope = _scope_for(engine, args.target)
            history = []
            console.print("[dim]Type your question, or 'exit' to quit.[/dim]")
            while True:
                try:
                    question = console.input("[bold cyan]You:[/bold cyan] ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question.lower() in ("exit", "quit"):
                    break
                if not question:
                    continue
                with console.status("Thinking…"):
                    result = engine.answer(question, scope, history[-6:])
                _show(result)
                history += [{"role": "user", "content": question},
                            {"role": "assistant", "content": result["answer"]}]
    finally:
        engine.close()


if __name__ == "__main__":
    main()
