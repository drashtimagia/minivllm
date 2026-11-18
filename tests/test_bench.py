import json
import sys
from pathlib import Path
from typing import Any

import pytest

from bench import common, make_charts, make_tables, serve_bench
from tests.conftest import TEST_MODEL


def _args(*extra: str) -> Any:
    return serve_bench.parse_args(
        ["--system", "paged", "--workload", "sharegpt", "--dataset-path", "sg.json", "--model", TEST_MODEL, *extra]
    )


def _flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_minivllm_server_commands_differ_only_where_the_system_differs() -> None:
    args = _args("--max-model-len", "2048")
    static = serve_bench.build_server_cmd("static", args)
    contiguous = serve_bench.build_server_cmd("continuous", args)
    paged = serve_bench.build_server_cmd("paged", args)
    prefix = serve_bench.build_server_cmd("paged_prefix", args)
    assert _flag_value(static, "--scheduler") == "static"
    assert _flag_value(contiguous, "--block-size") == "2048"
    assert _flag_value(paged, "--block-size") == "16"
    assert "--enable-prefix-caching" not in paged and "--enable-prefix-caching" in prefix
    assert _flag_value(prefix, "--kv-budget-gb") == _flag_value(paged, "--kv-budget-gb")


def test_phase_8_systems_differ_from_paged_only_by_their_flag() -> None:
    args = _args("--chunk-budget", "1024")
    paged = serve_bench.build_server_cmd("paged", args)
    chunked = serve_bench.build_server_cmd("paged_chunked", args)
    graphs = serve_bench.build_server_cmd("paged_graphs", args)
    both = serve_bench.build_server_cmd("paged_chunked_graphs", args)
    assert "--enable-chunked-prefill" in chunked and "--enable-cuda-graphs" not in chunked
    assert "--enable-cuda-graphs" in graphs and "--enable-chunked-prefill" not in graphs
    assert "--enable-chunked-prefill" in both and "--enable-cuda-graphs" in both
    assert _flag_value(chunked, "--max-num-batched-tokens") == "1024"
    assert _flag_value(graphs, "--max-num-batched-tokens") == _flag_value(paged, "--max-num-batched-tokens")


def test_vllm_gets_the_same_block_pool_as_the_budget_implies() -> None:
    args = _args("--kv-budget-gb", "0.5")
    cmd = serve_bench.build_server_cmd("vllm_eager", args)
    assert "--enforce-eager" in cmd and "--no-enable-prefix-caching" in cmd
    assert _flag_value(cmd, "--num-gpu-blocks-override") == str(
        common.kv_blocks_for_budget(TEST_MODEL, __import__("torch").bfloat16, 16, 0.5)
    )
    assert "--enable-prefix-caching" in serve_bench.build_server_cmd("vllm_prefix", args)


def test_client_commands_use_one_client_for_every_system(tmp_path: Path) -> None:
    args = _args("--output-len", "64")
    sharegpt = serve_bench.build_client_cmd("sharegpt", "inf", args, tmp_path)
    agent = serve_bench.build_client_cmd("agent", "4", args, tmp_path)
    assert sharegpt[:3] == ["vllm", "bench", "serve"] == agent[:3]
    assert "--ignore-eos" in sharegpt and _flag_value(sharegpt, "--sharegpt-output-len") == "64"
    assert _flag_value(sharegpt, "--request-rate") == "inf"
    assert _flag_value(agent, "--dataset-name") == "prefix_repetition"
    assert _flag_value(agent, "--prefix-repetition-prefix-len") == "3072"


def test_sharegpt_workload_needs_a_dataset() -> None:
    with pytest.raises(SystemExit):
        serve_bench.parse_args(["--system", "paged", "--workload", "sharegpt"])


def test_results_from_a_dirty_tree_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common, "git_state", lambda: ("a" * 40, True))
    with pytest.raises(RuntimeError, match="uncommitted"):
        common.write_result(tmp_path, "x.json", "serving", {}, {}, allow_dirty=False)
    path = common.write_result(tmp_path, "x.json", "serving", {}, {}, allow_dirty=True)
    assert json.loads(path.read_text())["meta"]["dirty"] is True


def _fake(kind: str, config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    meta = {"kind": kind, "commit": "c" * 40, "dirty": False, "date": "2000-01-01T00:00:00",
            "hardware": {"gpu": ["test gpu"], "cpu": "x"}}
    return {"meta": meta, "config": config, "result": result}


def _fake_results() -> list[dict[str, Any]]:
    serving = lambda system, workload, rate, thr: _fake(  # noqa: E731
        "serving", {"system": system, "workload": workload, "request_rate": rate},
        {"output_throughput": thr, "p99_ttft_ms": thr / 10, "mean_ttft_ms": 1.0, "median_ttft_ms": 1.0},
    )
    return [
        serving("paged", "sharegpt", "2", 20.0), serving("paged", "sharegpt", "inf", 40.0),
        serving("static", "sharegpt", "2", 10.0), serving("paged_prefix", "agent", "4", 30.0),
        _fake("max_concurrency", {"layout": "paged", "kv_budget_gb": 1.0, "block_size": 16},
              {"peak_running_seqs": 9, "num_preemptions": 0}),
        _fake("profile_decode", {}, {"top_ops": [{"name": "aten::mm", "count": 3, "self_cpu_us": 5.0, "self_device_us": 0.0}]}),
    ]


def test_tables_are_built_only_from_result_files() -> None:
    text = make_tables.render_tables(_fake_results())
    assert "| 2 | 10.0 | 20.0 |" in text
    assert "| inf | - | 40.0 |" in text, "a system with no run at a rate shows a dash, not a guess"
    assert "peak running seqs" in text and "aten::mm" in text and "test gpu" in text


def test_readme_section_is_replaced_between_markers() -> None:
    body = f"intro\n{make_tables.START_MARKER}\nold\n{make_tables.END_MARKER}\noutro"
    out = make_tables.replace_between_markers(body, "NEW")
    assert "NEW" in out and "old" not in out and out.startswith("intro") and out.endswith("outro")
    with pytest.raises(ValueError):
        make_tables.replace_between_markers("no markers", "NEW")


def test_charts_are_written_for_available_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for i, result in enumerate(_fake_results()):
        (results_dir / f"{i}.json").write_text(json.dumps(result))
    monkeypatch.setattr(sys, "argv", ["make_charts", "--results-dir", str(results_dir), "--out-dir", str(tmp_path / "c")])
    make_charts.main()
    assert {p.name for p in (tmp_path / "c").iterdir()} == {
        "throughput_vs_rate.png", "p99_ttft_vs_rate.png", "agent_ttft.png", "max_concurrency.png",
    }


def test_missing_results_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        common.load_results(tmp_path)
