import argparse
from pathlib import Path
from typing import Any

from bench.common import RESULTS_DIR, load_results, of_kind, rate_key, serving_series
from bench.serve_bench import SYSTEMS

START_MARKER = "<!-- BENCH:START -->"
END_MARKER = "<!-- BENCH:END -->"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:,.1f}"


def _by_rate_table(results: list[dict[str, Any]], workload: str, metric: str) -> str | None:
    systems = [s for s in SYSTEMS if serving_series(results, workload, s, metric)]
    if not systems:
        return None
    columns = {s: dict(serving_series(results, workload, s, metric)) for s in systems}
    rates = sorted({rate for col in columns.values() for rate in col}, key=rate_key)
    rows = [[rate] + [_fmt(columns[s].get(rate)) for s in systems] for rate in rates]
    return _table(["request rate (req/s)"] + systems, rows)


def _agent_table(results: list[dict[str, Any]]) -> str | None:
    rows = []
    for system in SYSTEMS:
        for r in of_kind(results, "serving"):
            if r["config"]["workload"] == "agent" and r["config"]["system"] == system:
                res = r["result"]
                rows.append([system, r["config"]["request_rate"], _fmt(res["mean_ttft_ms"]),
                             _fmt(res["median_ttft_ms"]), _fmt(res["p99_ttft_ms"]), _fmt(res["output_throughput"])])
    header = ["system", "request rate", "mean TTFT (ms)", "median TTFT (ms)", "p99 TTFT (ms)", "output tok/s"]
    return _table(header, rows) if rows else None


def _concurrency_table(results: list[dict[str, Any]]) -> str | None:
    rows = [
        [r["config"]["layout"], str(r["config"]["kv_budget_gb"]), str(r["config"]["block_size"]),
         str(r["result"]["peak_running_seqs"]), str(r["result"]["num_preemptions"])]
        for r in of_kind(results, "max_concurrency")
    ]
    header = ["layout", "KV budget (GB)", "block size", "peak running seqs", "preemptions"]
    return _table(header, rows) if rows else None


def _profile_table(results: list[dict[str, Any]]) -> str | None:
    profiles = of_kind(results, "profile_decode")
    if not profiles:
        return None
    ops = profiles[-1]["result"]["top_ops"]
    use_device = any(op["self_device_us"] for op in ops)
    key = "self_device_us" if use_device else "self_cpu_us"
    total = sum(op[key] for op in ops) or 1.0
    rows = [[op["name"], str(op["count"]), _fmt(op[key]), f"{100 * op[key] / total:.1f}%"] for op in ops]
    label = "GPU" if use_device else "CPU"
    return _table(["op", "calls", f"self {label} time (us)", "share of listed ops"], rows)


def _metadata_table(results: list[dict[str, Any]]) -> str:
    rows = sorted(
        {
            (r["meta"]["kind"], r["meta"]["commit"][:10], str(r["meta"]["dirty"]),
             "; ".join(r["meta"]["hardware"]["gpu"] or [r["meta"]["hardware"]["cpu"]]), r["meta"]["date"][:10])
            for r in results
        }
    )
    return _table(["kind", "commit", "dirty tree", "hardware", "date"], [list(row) for row in rows])


def render_tables(results: list[dict[str, Any]]) -> str:
    sections = [
        ("Output throughput, ShareGPT (tokens/s)", _by_rate_table(results, "sharegpt", "output_throughput")),
        ("p99 time to first token, ShareGPT (ms)", _by_rate_table(results, "sharegpt", "p99_ttft_ms")),
        ("Agent workload: shared system prompt", _agent_table(results)),
        ("Peak concurrent sequences at a fixed KV budget", _concurrency_table(results)),
        ("Where one decode step spends its time", _profile_table(results)),
        ("Run metadata", _metadata_table(results)),
    ]
    return "\n\n".join(f"### {title}\n\n{body}" for title, body in sections if body)


def replace_between_markers(text: str, new_body: str) -> str:
    if START_MARKER not in text or END_MARKER not in text:
        raise ValueError(f"README must contain {START_MARKER} and {END_MARKER}")
    head, rest = text.split(START_MARKER, 1)
    _, tail = rest.split(END_MARKER, 1)
    return f"{head}{START_MARKER}\n\n{new_body}\n\n{END_MARKER}{tail}"


def main() -> None:
    p = argparse.ArgumentParser(description="Render markdown tables from bench/results")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--readme", type=Path, help="replace the section between the BENCH markers in this file")
    args = p.parse_args()
    tables = render_tables(load_results(args.results_dir))
    if args.readme:
        args.readme.write_text(replace_between_markers(args.readme.read_text(), tables))
    else:
        print(tables)


if __name__ == "__main__":
    main()
