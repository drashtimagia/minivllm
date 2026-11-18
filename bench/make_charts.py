import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench.common import RESULTS_DIR, load_results, of_kind, rate_key, serving_series  # noqa: E402
from bench.serve_bench import SYSTEMS  # noqa: E402

COLORS = dict(zip(SYSTEMS, ["#7f7f7f", "#1f77b4", "#2ca02c", "#17becf", "#d62728", "#ff7f0e", "#9467bd"]))
INF_LABEL = "inf"


def _rate_chart(results: list[dict[str, Any]], metric: str, ylabel: str, title: str, out: Path) -> bool:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rates = sorted(
        {r for s in SYSTEMS for r, _ in serving_series(results, "sharegpt", s, metric)}, key=rate_key
    )
    drew = False
    for system in SYSTEMS:
        points = dict(serving_series(results, "sharegpt", system, metric))
        if points:
            ax.plot(range(len(rates)), [points.get(r) for r in rates], marker="o", label=system, color=COLORS[system])
            drew = True
    if drew:
        ax.set_xticks(range(len(rates)), [r if r != "inf" else INF_LABEL for r in rates])
        ax.set(xlabel="request rate (req/s)", ylabel=ylabel, title=title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=150)
    plt.close(fig)
    return drew


def _agent_chart(results: list[dict[str, Any]], out: Path) -> bool:
    runs = [r for r in of_kind(results, "serving") if r["config"]["workload"] == "agent"]
    if not runs:
        return False
    runs.sort(key=lambda r: SYSTEMS.index(r["config"]["system"]))
    labels = [r["config"]["system"] for r in runs]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.38
    ax.bar([i - width / 2 for i in range(len(runs))], [r["result"]["mean_ttft_ms"] for r in runs], width, label="mean")
    ax.bar([i + width / 2 for i in range(len(runs))], [r["result"]["p99_ttft_ms"] for r in runs], width, label="p99")
    ax.set_xticks(range(len(runs)), labels, rotation=20)
    ax.set(ylabel="time to first token (ms)", title="Agent workload: shared system prompt")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def _concurrency_chart(results: list[dict[str, Any]], out: Path) -> bool:
    runs = of_kind(results, "max_concurrency")
    if not runs:
        return False
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.bar([r["config"]["layout"] for r in runs], [r["result"]["peak_running_seqs"] for r in runs])
    budget = runs[0]["config"]["kv_budget_gb"]
    ax.set(ylabel="peak running sequences", title=f"Concurrency at a {budget} GB KV budget")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Render charts from bench/results")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "charts")
    args = p.parse_args()
    results = load_results(args.results_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "throughput_vs_rate.png": _rate_chart(
            results, "output_throughput", "output tokens/s", "Throughput vs request rate", args.out_dir / "throughput_vs_rate.png"
        ),
        "p99_ttft_vs_rate.png": _rate_chart(
            results, "p99_ttft_ms", "p99 TTFT (ms)", "p99 time to first token vs request rate", args.out_dir / "p99_ttft_vs_rate.png"
        ),
        "agent_ttft.png": _agent_chart(results, args.out_dir / "agent_ttft.png"),
        "max_concurrency.png": _concurrency_chart(results, args.out_dir / "max_concurrency.png"),
    }
    for name, drawn in charts.items():
        print(f"{'wrote' if drawn else 'skipped (no data)'} {name}")


if __name__ == "__main__":
    main()
