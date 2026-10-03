from __future__ import annotations
import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.prompt import Prompt, IntPrompt
from rich import box

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

console = Console()


def _print_search_stats(stats: dict):
    if not stats:
        return
    method = stats.get("routing_method", "")
    label = "[cyan]vector index[/cyan]" if method == "vector_index" else "[yellow]live scan[/yellow]"
    console.print(Rule(f"[dim]Search Statistics ({label})[/dim]"))
    dirs = stats.get("directories_scanned", 0)
    if isinstance(dirs, int):
        console.print(f"  [dim]Directories scanned:[/dim] [cyan]{dirs}[/cyan]")
    for d in stats.get("directory_paths", []):
        console.print(f"    [dim]  {d}[/dim]")
    console.print(f"  [dim]Files matched:[/dim]        [green]{stats.get('files_matched', 0)}[/green]")
    for f in stats.get("matched_file_names", []):
        console.print(f"    [green]  {f}[/green]")
    console.print(f"  [dim]Keywords used:[/dim]        [magenta]{', '.join(str(k) for k in stats.get('keywords_used', []))}[/magenta]")
    console.print()


def _print_citations(citations: list):
    if not citations:
        return
    table = Table(title="Citations", box=box.SIMPLE, show_header=True)
    table.add_column("Ref",     style="cyan",    width=5)
    table.add_column("File",    style="green")
    table.add_column("Chunk",   style="yellow",  width=7)
    table.add_column("Score",   style="magenta", width=7)
    table.add_column("Preview", style="dim")
    for c in citations:
        table.add_row(
            f"[{c['ref_id']}]",
            c.get("file_name", ""),
            str(c.get("chunk_index", "")),
            str(c.get("score", "")),
            c.get("preview", "")[:60],
        )
    console.print(table)


def _print_timings(timings: dict):
    if not timings:
        return
    table = Table(title="Node Timings", box=box.SIMPLE)
    table.add_column("Node",    style="cyan")
    table.add_column("Time (s)", style="yellow")
    for node, t in timings.items():
        table.add_row(node, str(t))
    console.print(table)


def _build_initial_state(query: str, root_path: str, top_k: int) -> dict:
    return {
        "query":               query,
        "root_path":           root_path,
        "top_k":               top_k,
        "routing_plan":        None,
        "candidate_documents": None,
        "parsed_chunks":       None,
        "retrieved_chunks":    None,
        "reranked_chunks":     None,
        "final_answer":        None,
        "source_files":        None,
        "citations":           None,
        "search_stats":        None,
        "evaluation":          None,
        "error":               None,
        "retry_count":         0,
        "node_timings":        {},
    }


def _display_result(result: dict):
    answer    = str(result.get("final_answer") or "No answer generated.")
    error     = result.get("error") or ""
    stats     = result.get("search_stats") or {}
    citations = result.get("citations") or []
    timings   = result.get("node_timings") or {}
    evaluation = result.get("evaluation") or {}
    retries   = result.get("retry_count", 0)

    _print_search_stats(stats)

    if error and not answer:
        console.print(Panel(str(error), title="[bold red]Error", border_style="red"))

    console.print(Panel(answer, title="[bold green]Answer", border_style="green"))
    _print_citations(citations)
    _print_timings(timings)

    if evaluation:
        console.print(
            f"[dim]Chunks used: {evaluation.get('chunks_used')} | "
            f"Avg score: {evaluation.get('avg_retrieval_score')} | "
            f"Retries: {retries}[/dim]\n"
        )


def _execute_and_display(query: str, root_path: str, top_k: int = 5) -> dict:
    from graph.builder import build_graph
    graph = build_graph()

    with console.status("[bold green]Running pipeline..."):
        result = graph.invoke(_build_initial_state(query, root_path, top_k))

    _display_result(result)
    return result


def _handle_locate_query(query: str) -> List[Dict]:
    from fileindex.locator import locate_files, format_locate_results
    from fileindex.index_builder import index_exists

    if not index_exists():
        console.print(Panel(
            "No file index found.\n"
            "Run [bold cyan]python main.py index[/bold cyan] to build one first.",
            title="[bold yellow]File Index Required",
            border_style="yellow",
        ))
        return []

    with console.status("[bold green]Searching file index..."):
        results = locate_files(query, top_k=10)

    formatted = format_locate_results(results, query)
    console.print(Panel(formatted, title="[bold green]File Locations Found", border_style="green"))

    if results:
        table = Table(title="Matched Files", box=box.SIMPLE)
        table.add_column("#",        style="cyan",    width=4)
        table.add_column("Filename", style="green")
        table.add_column("Path",     style="dim")
        table.add_column("Size",     style="yellow",  width=10)
        table.add_column("Match",    style="magenta", width=8)
        for i, r in enumerate(results, 1):
            table.add_row(
                str(i),
                r.get("file_name", ""),
                r.get("file_path", ""),
                f"{r.get('size_kb', 0)} KB",
                f"{round(r.get('index_score', 0) * 100, 1)}%",
            )
        console.print(table)

    return results


def _offer_file_selection(
    source_files: List[str],
    top_k: int,
    original_result: dict,
):
    if not source_files:
        return

    unique_files = list(dict.fromkeys(source_files))

    console.print(Rule("[dim]File Actions[/dim]"))
    console.print("[dim]You can open a cited file to chat with it, or continue with a new query.[/dim]\n")

    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("#",        style="cyan",  width=4)
    table.add_column("Filename", style="green")
    table.add_column("Path",     style="dim")
    for i, fp in enumerate(unique_files, 1):
        table.add_row(str(i), os.path.basename(fp), fp)
    console.print(table)

    choices = [str(i) for i in range(1, len(unique_files) + 1)] + ["s", "n"]
    console.print(
        f"  Enter [cyan]1-{len(unique_files)}[/cyan] to chat with a file  |  "
        "[cyan]s[/cyan] to skip  |  [cyan]n[/cyan] for new query"
    )
    choice = Prompt.ask("Action", choices=choices, default="s")

    if choice in ("s", "n"):
        return choice

    selected_path = unique_files[int(choice) - 1]
    _run_inline_chat(selected_path, top_k, original_result, unique_files)
    return "done"


def _run_inline_chat(
    file_path: str,
    top_k: int,
    original_result: dict,
    original_files: List[str],
):
    from chat.document_chat import load_document, chat, clear_history

    console.print(Panel(
        f"[bold cyan]Chatting with:[/bold cyan] {os.path.basename(file_path)}\n"
        f"[dim]Path: {file_path}[/dim]\n\n"
        "[dim]Commands: 'back' return to results | 'clear' reset history | "
        "'info' file stats | 'switch' pick a different cited file | 'exit' quit[/dim]",
        title="[bold green]DocLAMAR — Document Chat",
        border_style="green",
    ))

    try:
        with console.status("[bold green]Loading and indexing document..."):
            session = load_document(file_path)
    except Exception as e:
        console.print(f"[red]Failed to load document: {e}[/red]")
        return

    console.print(
        f"[green]Ready![/green] [cyan]{session.file_name}[/cyan] "
        f"({session.file_size_kb} KB, {session.total_chunks} chunks indexed)\n"
    )

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Returning to results.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() == "back":
            console.print("[dim]Returning to your previous results.[/dim]\n")
            console.print(Rule("[dim]Previous Results[/dim]"))
            _display_result(original_result)
            _offer_file_selection(original_files, top_k, original_result)
            break

        if user_input.lower() == "exit":
            console.print("[dim]Goodbye.[/dim]")
            sys.exit(0)

        if user_input.lower() == "clear":
            clear_history(session)
            console.print("[yellow]Conversation history cleared.[/yellow]\n")
            continue

        if user_input.lower() == "info":
            console.print(
                f"[dim]File:[/dim]    {session.file_name}\n"
                f"[dim]Path:[/dim]    {session.file_path}\n"
                f"[dim]Size:[/dim]    {session.file_size_kb} KB\n"
                f"[dim]Chunks:[/dim]  {session.total_chunks}\n"
                f"[dim]Turns:[/dim]   {len(session.history) // 2}\n"
            )
            continue

        if user_input.lower() == "switch":
            unique_files = list(dict.fromkeys(original_files))
            available = [f for f in unique_files if f != file_path]
            if not available:
                console.print("[yellow]No other cited files to switch to.[/yellow]")
                continue
            table = Table(box=box.SIMPLE)
            table.add_column("#",        style="cyan", width=4)
            table.add_column("Filename", style="green")
            table.add_column("Path",     style="dim")
            for i, fp in enumerate(available, 1):
                table.add_row(str(i), os.path.basename(fp), fp)
            console.print(table)
            choices = [str(i) for i in range(1, len(available) + 1)] + ["c"]
            console.print(f"  [cyan]1-{len(available)}[/cyan] to switch  |  [cyan]c[/cyan] to cancel")
            sw = Prompt.ask("Switch to", choices=choices, default="c")
            if sw != "c":
                new_path = available[int(sw) - 1]
                console.print(f"[dim]Switching to {os.path.basename(new_path)}...[/dim]")
                _run_inline_chat(new_path, top_k, original_result, original_files)
                break
            continue

        with console.status("[bold green]Thinking..."):
            result = chat(session, user_input, top_k=top_k)

        console.print(Panel(
            result["answer"],
            title="[bold green]Assistant",
            border_style="green",
        ))

        if result.get("citations"):
            _print_citations(result["citations"])

        console.print()


def run_interactive():
    from fileindex.index_builder import index_exists, load_manifest

    idx_status = ""
    if index_exists():
        m = load_manifest()
        idx_status = (
            f"\n[dim]File index active — {m.get('files_indexed', 0)} files indexed. "
            f"Run [cyan]python main.py index[/cyan] to manage.[/dim]"
        )
    else:
        idx_status = (
            "\n[dim yellow]No file index. Run [cyan]python main.py index[/cyan] "
            "to build one for faster routing.[/dim yellow]"
        )

    console.print(Panel(
        "[bold cyan]DocLAMAR[/bold cyan] — Agentic Document Intelligence\n"
        "[dim]Ask anything about your documents, or ask where a file is.[/dim]"
        + idx_status,
        border_style="green",
    ))

    from graph.builder import build_graph
    from fileindex.locator import is_locate_query
    graph = build_graph()

    while True:
        console.print(Rule())

        query = Prompt.ask(
            "\n[bold cyan]Your query[/bold cyan] (or 'exit')"
        ).strip()

        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if not query:
            console.print("[yellow]Query cannot be empty.[/yellow]")
            continue

        if is_locate_query(query):
            console.print("[dim]Detected file location query — searching index...[/dim]\n")
            located = _handle_locate_query(query)

            if located:
                located_paths = [r["file_path"] for r in located]
                choices = [str(i) for i in range(1, len(located) + 1)] + ["s"]
                console.print(
                    f"\n  Enter [cyan]1-{len(located)}[/cyan] to chat with a file  |  [cyan]s[/cyan] to skip"
                )
                pick = Prompt.ask("Open file", choices=choices, default="s")
                if pick != "s":
                    chosen = located_paths[int(pick) - 1]
                    dummy_result = {"source_files": located_paths, "final_answer": "", "node_timings": {}}
                    _run_inline_chat(chosen, 5, dummy_result, located_paths)

            again = Prompt.ask("[dim]Another query?[/dim]", choices=["y", "n"], default="y")
            if again == "n":
                console.print("[dim]Goodbye.[/dim]")
                break
            continue

        root_path = Prompt.ask("[bold cyan]Folder path to search[/bold cyan]").strip()
        if not root_path:
            console.print("[yellow]Folder path cannot be empty.[/yellow]")
            continue
        if not os.path.isdir(root_path):
            console.print(f"[red]Path does not exist: {root_path}[/red]")
            continue

        top_k = IntPrompt.ask("[bold cyan]Number of results (chunks)[/bold cyan]", default=5)

        console.print()
        with console.status("[bold green]Running DocLAMAR pipeline..."):
            result = graph.invoke(_build_initial_state(query, root_path, top_k))

        _display_result(result)

        source_files = result.get("source_files") or []
        action = _offer_file_selection(source_files, top_k, result)

        if action == "n":
            console.print("[dim]Goodbye.[/dim]")
            break

        again = Prompt.ask("[dim]Another query?[/dim]", choices=["y", "n"], default="y")
        if again == "n":
            console.print("[dim]Goodbye.[/dim]")
            break


def run_chat(file_path: str, top_k: int = 5):
    from chat.document_chat import load_document, chat, clear_history

    console.print(Panel(
        f"[bold cyan]File:[/bold cyan] {file_path}\n"
        "[dim]Commands: 'exit' quit | 'clear' reset history | 'info' file stats[/dim]",
        title="[bold green]DocLAMAR -- Document Chat",
        border_style="green",
    ))

    try:
        with console.status("[bold green]Loading and indexing document..."):
            session = load_document(file_path)
    except Exception as e:
        console.print(f"[red]Failed to load document: {e}[/red]")
        sys.exit(1)

    console.print(
        f"[green]Ready![/green] Loaded [cyan]{session.file_name}[/cyan] "
        f"({session.file_size_kb} KB, {session.total_chunks} chunks indexed)\n"
    )

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            console.print("[dim]Session ended.[/dim]")
            break
        if user_input.lower() == "clear":
            clear_history(session)
            console.print("[yellow]Conversation history cleared.[/yellow]\n")
            continue
        if user_input.lower() == "info":
            console.print(
                f"[dim]File:[/dim]    {session.file_name}\n"
                f"[dim]Path:[/dim]    {session.file_path}\n"
                f"[dim]Size:[/dim]    {session.file_size_kb} KB\n"
                f"[dim]Chunks:[/dim]  {session.total_chunks}\n"
                f"[dim]Turns:[/dim]   {len(session.history) // 2}\n"
            )
            continue

        with console.status("[bold green]Thinking..."):
            result = chat(session, user_input, top_k=top_k)

        console.print(Panel(result["answer"], title="[bold green]Assistant", border_style="green"))

        if result.get("citations"):
            _print_citations(result["citations"])

        console.print()


def run_index_manager():
    from fileindex.index_builder import (
        index_exists, build_file_index, load_manifest,
        check_staleness, delete_index
    )
    from fileindex.index_search import invalidate_cache

    console.print(Panel(
        "[bold cyan]DocLAMAR File Index Manager[/bold cyan]\n"
        "[dim]Builds a semantic vector index of your documents for fast routing.[/dim]",
        border_style="cyan",
    ))

    while True:
        console.print(Rule())

        if index_exists():
            manifest = load_manifest()
            staleness = check_staleness()
            console.print(f"[green]Index exists[/green] — built at [cyan]{manifest.get('created_at', 'unknown')}[/cyan]")
            console.print(f"  Files indexed : [cyan]{manifest.get('files_indexed', 0)}[/cyan]")
            console.print(f"  Root paths    : [cyan]{', '.join(manifest.get('root_paths', []))}[/cyan]")
            if staleness.get("stale"):
                console.print(f"  [yellow]Warning: {staleness['changed_pct']}% of files changed since last index.[/yellow]")
            else:
                console.print(f"  [green]Index is up to date.[/green]")
        else:
            console.print("[yellow]No index found.[/yellow] Routing uses live directory scanning.")

        console.print()
        console.print("  [1] Build / Rebuild index")
        console.print("  [2] Check index status")
        console.print("  [3] Delete index")
        console.print("  [4] Back")

        choice = Prompt.ask("\nChoice", choices=["1", "2", "3", "4"], default="4")

        if choice == "4":
            break

        elif choice == "2":
            if not index_exists():
                console.print("[red]No index found.[/red]")
            else:
                staleness = check_staleness()
                console.print(f"  Created     : {staleness.get('created_at')}")
                console.print(f"  Total files : {staleness.get('total')}")
                console.print(f"  Changed     : {staleness.get('changed')}")
                console.print(f"  Missing     : {staleness.get('missing')}")
                console.print(f"  Stale       : {'Yes' if staleness.get('stale') else 'No'}")
                console.print(f"  Root paths  : {', '.join(staleness.get('root_paths', []))}")

        elif choice == "3":
            confirm = Prompt.ask("[red]Delete the index?[/red]", choices=["y", "n"], default="n")
            if confirm == "y":
                delete_index()
                invalidate_cache()
                console.print("[green]Index deleted.[/green]")

        elif choice == "1":
            console.print(
                "\n[dim]Enter one or more folders to index, separated by commas.\n"
                r"Example: C:\Users\Shlok\Documents, C:\Users\Shlok\Desktop[/dim]"
            )
            raw = Prompt.ask("[bold cyan]Folders to index[/bold cyan]").strip()
            paths = [p.strip() for p in raw.split(",") if p.strip()]

            invalid = [p for p in paths if not os.path.isdir(p)]
            if invalid:
                console.print(f"[red]These paths do not exist: {', '.join(invalid)}[/red]")
                continue

            def on_progress(current, total):
                pass

            with console.status("[bold green]Building file index..."):
                result = build_file_index(paths, on_progress=on_progress)

            invalidate_cache()

            if result["status"] == "built":
                console.print(
                    f"\n[green]Index built![/green] {result['files_indexed']} files indexed.\n"
                    f"[dim]Stored at: {result['index_dir']}[/dim]"
                )
            else:
                console.print("[yellow]No supported files found in the specified folders.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="DocLAMAR: Agentic Document Intelligence")
    subparsers = parser.add_subparsers(dest="mode")

    subparsers.add_parser("interactive", help="Interactive mode (default)")
    subparsers.add_parser("index",       help="Manage the file system vector index")

    qp = subparsers.add_parser("query", help="Single query via flags")
    qp.add_argument("--query",     "-q", required=True)
    qp.add_argument("--root_path", "-p", required=True)
    qp.add_argument("--top_k",     "-k", type=int, default=5)

    cp = subparsers.add_parser("chat", help="Chat with a specific document")
    cp.add_argument("--file",  "-f", required=True)
    cp.add_argument("--top_k", "-k", type=int, default=5)

    ep = subparsers.add_parser("eval", help="Run evaluation suite")
    ep.add_argument("--root_path", "-p", required=True)
    ep.add_argument("--top_k",     "-k", type=int, default=5)

    args = parser.parse_args()
    mode = args.mode or "interactive"

    if mode in ("interactive", None):
        run_interactive()

    elif mode == "query":
        result = _execute_and_display(args.query, args.root_path, args.top_k)
        if result.get("error") and not result.get("final_answer"):
            sys.exit(1)

    elif mode == "chat":
        run_chat(args.file, args.top_k)

    elif mode == "index":
        run_index_manager()

    elif mode == "eval":
        from tests.run_eval import TEST_CASES
        from tests.evaluator import run_eval_suite, print_eval_report
        results = run_eval_suite(TEST_CASES, args.root_path, args.top_k)
        print_eval_report(results)


if __name__ == "__main__":
    main()
