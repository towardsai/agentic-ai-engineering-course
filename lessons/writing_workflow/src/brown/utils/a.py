"""Async utils."""

import asyncio
import concurrent.futures
from collections.abc import Callable
from functools import wraps
from itertools import zip_longest
from typing import Any, Awaitable, Coroutine, Iterable, List, Optional, TypeVar, cast


def asyncio_module(show_progress: bool = False) -> Any:
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        module = tqdm_asyncio
    else:
        module = asyncio

    return module


def _suppress_grpc_event_loop_errors(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """
    Custom exception handler to suppress gRPC-related event loop errors.

    gRPC's PollerCompletionQueue callbacks sometimes try to access event loops
    from thread pool workers, causing RuntimeError and BlockingIOError exceptions
    that are harmless but noisy. This handler silences them.
    """
    exception = context.get("exception")
    message = context.get("message", "")

    # Suppress gRPC PollerCompletionQueue errors
    if isinstance(exception, (RuntimeError, BlockingIOError)):
        if "PollerCompletionQueue" in message or "ThreadPoolExecutor" in str(exception):
            # Silently ignore these gRPC internal errors
            return

    # For all other exceptions, use the default handler
    loop.default_exception_handler(context)


def as_sync(func: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
    """
    Decorator to convert an async function into a synchronous function.

    This decorator wraps an async function to make it callable from synchronous code.
    It handles event loop management automatically, including cases where an event
    loop is already running by executing the coroutine in a separate thread.

    Args:
        func: The async function to wrap.

    Returns:
        A synchronous wrapper function that returns the result of the async function.

    Raises:
        RuntimeError: If nested async operations are detected and nest_asyncio is not available.

    Example:
        >>> @as_sync
        ... async def async_function(x: int) -> int:
        ...     await asyncio.sleep(0.1)
        ...     return x * 2
        ...
        >>> result = async_function(5)  # Can be called synchronously
        >>> print(result)  # 10
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Synchronous wrapper for the async function."""
        coro = func(*args, **kwargs)
        if not isinstance(coro, (Coroutine, Awaitable)):
            raise TypeError(f"Function {func.__name__} decorated with @as_sync must be async and return a coroutine, but got {type(coro)}")
        return asyncio_run(coro)

    return cast(Callable[..., Any], wrapper)


def asyncio_run(coro: Coroutine) -> Any:
    """
    Gets an existing event loop to run the coroutine.

    If there is no existing event loop, creates a new one.
    If an event loop is already running, uses threading to run in a separate thread.
    """

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, run in a new one in a separate thread.
            def run_coro_in_thread() -> Any:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.set_exception_handler(_suppress_grpc_event_loop_errors)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_coro_in_thread)
                return future.result()
        else:
            # If we're here, there's an existing loop but it's not running
            return loop.run_until_complete(coro)

    except RuntimeError:
        # If we can't get the event loop, create a new one.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(_suppress_grpc_event_loop_errors)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def run_async_tasks(
    tasks: List[Coroutine],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> List[Any]:
    """Run a list of async tasks."""
    tasks_to_execute: List[Any] = tasks
    if show_progress:
        try:
            import nest_asyncio
            from tqdm.asyncio import tqdm

            # jupyter notebooks already have an event loop running
            # we need to reuse it instead of creating a new one
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()

            async def _tqdm_gather() -> List[Any]:
                return await tqdm.gather(*tasks_to_execute, desc=progress_bar_desc)

            tqdm_outputs: List[Any] = loop.run_until_complete(_tqdm_gather())
            return tqdm_outputs
        # run the operation w/o tqdm on hitting a fatal
        # may occur in some environments where tqdm.asyncio
        # is not supported
        except Exception:
            pass

    async def _gather() -> List[Any]:
        return await asyncio.gather(*tasks_to_execute)

    outputs: List[Any] = asyncio_run(_gather())

    return outputs


def chunks(iterable: Iterable, size: int) -> Iterable:
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=None)


async def batch_gather(tasks: List[Coroutine], batch_size: int = 10, verbose: bool = False) -> List[Any]:
    output: List[Any] = []
    for task_chunk in chunks(tasks, batch_size):
        task_chunk = (task for task in task_chunk if task is not None)
        output_chunk = await asyncio.gather(*task_chunk)
        output.extend(output_chunk)
        if verbose:
            print(f"Completed {len(output)} out of {len(tasks)} tasks")
    return output


def get_asyncio_module(show_progress: bool = False) -> Any:
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        module = tqdm_asyncio
    else:
        module = asyncio

    return module


DEFAULT_NUM_WORKERS = 4

T = TypeVar("T")


async def run_jobs(
    jobs: List[Coroutine[Any, Any, T]],
    show_progress: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
    desc: Optional[str] = None,
) -> List[T]:
    """
    Run jobs.

    Args:
        jobs (List[Coroutine]):
            List of jobs to run.
        show_progress (bool):
            Whether to show progress bar.

    Returns:
        List[Any]:
            List of results.

    """
    semaphore = asyncio.Semaphore(workers)

    async def worker(job: Coroutine) -> Any:
        async with semaphore:
            return await job

    pool_jobs = [worker(job) for job in jobs]

    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        results = await tqdm_asyncio.gather(*pool_jobs, desc=desc)
    else:
        results = await asyncio.gather(*pool_jobs)

    return results
