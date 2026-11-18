from prometheus_client import CollectorRegistry, Gauge, Histogram, generate_latest

from minivllm.sequence import EngineStats

TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
TPOT_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.running = Gauge("minivllm_num_running_seqs", "Sequences in the running batch", registry=self.registry)
        self.waiting = Gauge("minivllm_num_waiting_seqs", "Sequences waiting for admission", registry=self.registry)
        self.kv_usage = Gauge(
            "minivllm_kv_cache_usage_ratio", "Fraction of KV blocks held by running sequences", registry=self.registry
        )
        self.hit_rate = Gauge(
            "minivllm_prefix_cache_hit_rate", "Prompt tokens served from the prefix cache over prompt tokens admitted",
            registry=self.registry,
        )
        self.ttft = Histogram(
            "minivllm_time_to_first_token_seconds", "Request arrival to first generated token",
            buckets=TTFT_BUCKETS, registry=self.registry,
        )
        self.tpot = Histogram(
            "minivllm_time_per_output_token_seconds", "Mean gap between generated tokens after the first",
            buckets=TPOT_BUCKETS, registry=self.registry,
        )

    def render(self, stats: EngineStats) -> bytes:
        self.running.set(stats.num_running)
        self.waiting.set(stats.num_waiting)
        # Cached free blocks hold reusable data but are free to be taken, so they do not
        # count as used.
        self.kv_usage.set(1 - stats.num_free_blocks / stats.num_blocks if stats.num_blocks else 0.0)
        self.hit_rate.set(stats.prefix_hit_rate)
        return generate_latest(self.registry)
