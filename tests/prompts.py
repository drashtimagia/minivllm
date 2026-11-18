from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    chat: bool


LONG_ARTICLE = """\
The history of computing hardware is usually told as a sequence of machines, but it is
more accurately a sequence of bottlenecks. Each generation of engineers found the part of
the system that limited everything else, removed it, and in doing so exposed the next one.
Early mechanical calculators were limited by the precision of gears and the patience of the
people turning them. Relays replaced gears and made switching faster, but relays wore out
and clicked loudly enough that operators could hear a program run. Vacuum tubes switched
thousands of times faster than relays, yet they burned out so often that machines like ENIAC
spent a large share of their time being repaired rather than computing. The transistor
solved the reliability problem and shrank the machine, and the integrated circuit placed
many transistors on a single piece of silicon, which removed the wiring between them as the
dominant cost.

Once logic became cheap, memory became the bottleneck. Core memory, made of tiny magnetic
rings threaded by hand, was reliable but slow and expensive to manufacture. Semiconductor
memory replaced it in the 1970s, and from then on processor speed and memory capacity grew
along two different curves. Processors became faster much more quickly than memory became
faster, and the gap between them, sometimes called the memory wall, shaped computer
architecture for decades. Caches were introduced so that the processor could keep a small
amount of recently used data close at hand. Prefetchers guessed what data would be needed
next. Out of order execution allowed the processor to keep working on independent
instructions while waiting for a slow load to complete.

The same pattern repeated in storage. Hard disks were limited by the time it took to move a
mechanical arm and wait for the platter to rotate, so file systems and databases were
designed around sequential access and large block sizes. When flash storage arrived, random
reads became cheap, and software that had been carefully tuned for spinning disks suddenly
had the wrong assumptions built in. Log structured designs, which had been invented to turn
random writes into sequential ones, turned out to fit flash well for a different reason:
flash cannot overwrite data in place and must erase large blocks at a time.

Networks followed the same arc. When links were slow, protocols were designed to send as
little as possible, and computation was cheap compared with communication. As links became
faster, the cost moved into the operating system: copying data between buffers, handling
interrupts, and switching between user and kernel mode. Techniques such as zero copy
transmission, interrupt coalescing, and kernel bypass were all responses to the discovery
that the network card was no longer the slow part.

Machine learning workloads are the most recent example. Training large models was first
limited by raw arithmetic throughput, which led to specialized accelerators with thousands
of parallel multiply units. As those units became faster, the limiting factor became the
speed at which weights and activations could be moved from memory into the arithmetic
units. Serving a trained language model is limited in yet another way. Generating text one
token at a time means that for each step the accelerator must read every weight of the model
from memory while doing relatively little arithmetic per byte read. Batching many requests
together amortizes that memory traffic, but each request also keeps a growing cache of
attention keys and values, and that cache competes for the same memory as the batch itself.
How that cache is stored, shared, and evicted therefore determines how many requests can be
served at once."""

LONG_CODE = '''\
import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Task:
    priority: int
    name: str = field(compare=False)
    duration: float = field(compare=False)


class Scheduler:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.queue: list[Task] = []
        self.clock = 0.0
        self.busy_until = [0.0] * workers
        self.log: list[tuple[float, int, str]] = []

    def submit(self, task: Task) -> None:
        heapq.heappush(self.queue, task)

    def run(self) -> float:
        while self.queue:
            task = heapq.heappop(self.queue)
            worker = min(range(self.workers), key=lambda w: self.busy_until[w])
            start = max(self.clock, self.busy_until[worker])
            self.busy_until[worker] = start + task.duration
            self.log.append((start, worker, task.name))
        return max(self.busy_until)


def simulate(tasks: list[tuple[int, str, float]], workers: int) -> float:
    scheduler = Scheduler(workers)
    for priority, name, duration in tasks:
        scheduler.submit(Task(priority, name, duration))
    return scheduler.run()


if __name__ == "__main__":
    jobs = [(2, "compile", 3.0), (1, "lint", 0.5), (3, "test", 6.0), (1, "format", 0.2)]
    print(simulate(jobs, workers=2))
'''

PROMPTS: list[Prompt] = [
    Prompt("short_fact", "The capital of France is", chat=False),
    Prompt("short_count", "1, 2, 3, 4, 5,", chat=False),
    Prompt("short_chat", "What is a hash table? Answer in two sentences.", chat=True),
    Prompt("story", "Once upon a time, in a small village by the sea, there lived", chat=False),
    Prompt("explain_kv", "Explain what a KV cache is in transformer inference.", chat=True),
    Prompt("list", "Give me five tips for writing clear technical documentation.", chat=True),
    Prompt("math", "If a train travels 120 km in 1.5 hours, what is its average speed? Show your work.", chat=True),
    Prompt("chinese", "请用简单的语言解释什么是机器学习。", chat=True),
    Prompt("french", "Écris un court paragraphe sur l'importance du sommeil.", chat=True),
    Prompt("spanish", "La inteligencia artificial está cambiando el mundo porque", chat=False),
    Prompt("japanese", "日本の四季について短く説明してください。", chat=True),
    Prompt("hindi", "भारत की राजधानी के बारे में दो वाक्य लिखिए।", chat=True),
    Prompt("german", "Die wichtigsten Erfindungen des zwanzigsten Jahrhunderts waren", chat=False),
    Prompt("code_python", "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", chat=False),
    Prompt("code_sql", "Write a SQL query that returns the top 3 customers by total order amount from tables customers(id, name) and orders(id, customer_id, amount).", chat=True),
    Prompt("code_c", "#include <stdio.h>\n\n/* Reverse a string in place. */\nvoid reverse(char *s) {\n", chat=False),
    Prompt("code_js", "Write a JavaScript function that debounces another function.", chat=True),
    Prompt("long_article", LONG_ARTICLE + "\n\nSummarize the main argument of this text in three sentences.", chat=True),
    Prompt("long_code", LONG_CODE + "\n\nExplain what this program prints and why.", chat=True),
    Prompt("long_continue", LONG_ARTICLE + "\n\nThe next bottleneck is likely to be", chat=False),
]


def encode_prompt(tokenizer: PreTrainedTokenizerBase, prompt: Prompt) -> list[int]:
    if prompt.chat:
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.text}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
        )
    return tokenizer.encode(prompt.text)
