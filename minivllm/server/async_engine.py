import asyncio
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from minivllm.engine import AnyEngine
from minivllm.sequence import RequestOutput, SamplingParams


class AsyncEngine:
    """Runs the synchronous engine in one background loop and hands each request its
    own asyncio queue of outputs.

    The engine is not thread safe, so it is only touched in two places that never
    overlap: the loop drains queued add/abort operations between steps, and engine.step
    runs on a single worker thread while the loop awaits it. Request handlers never call
    the engine directly; they only append to the operation queue."""

    def __init__(self, engine: AnyEngine) -> None:
        self.engine = engine
        self._ops: deque[Callable[[], None]] = deque()
        self._streams: dict[str, asyncio.Queue[RequestOutput | Exception]] = {}
        self._wakeup = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._task: asyncio.Task[None] | None = None
        self._failure: Exception | None = None

    @property
    def failure(self) -> Exception | None:
        return self._failure

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        self._task.add_done_callback(self._on_loop_exit)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._executor.shutdown(wait=True)

    def submit(
        self, request_id: str, prompt_token_ids: list[int], params: SamplingParams
    ) -> asyncio.Queue[RequestOutput | Exception]:
        if self._failure is not None:
            raise RuntimeError("the engine loop has stopped") from self._failure
        queue: asyncio.Queue[RequestOutput | Exception] = asyncio.Queue()
        self._streams[request_id] = queue
        self._ops.append(lambda: self.engine.add_request(request_id, prompt_token_ids, params))
        self._wakeup.set()
        return queue

    def abort(self, request_id: str) -> None:
        self._streams.pop(request_id, None)
        self._ops.append(lambda: self.engine.abort_request(request_id))
        self._wakeup.set()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not self._ops and not self.engine.has_unfinished_requests():
                self._wakeup.clear()
                await self._wakeup.wait()
            while self._ops:
                self._ops.popleft()()
            if not self.engine.has_unfinished_requests():
                continue
            outputs = await loop.run_in_executor(self._executor, self.engine.step)
            self._dispatch(outputs)

    def _dispatch(self, outputs: list[RequestOutput]) -> None:
        for out in outputs:
            queue = self._streams.get(out.request_id)
            if queue is None:
                continue
            queue.put_nowait(out)
            if out.finished:
                del self._streams[out.request_id]

    def _on_loop_exit(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        failure = task.exception()
        if failure is None:
            return
        # A dead loop would leave every waiting client hanging, so each one is handed
        # the error instead.
        self._failure = failure if isinstance(failure, Exception) else RuntimeError(str(failure))
        for queue in self._streams.values():
            queue.put_nowait(self._failure)
