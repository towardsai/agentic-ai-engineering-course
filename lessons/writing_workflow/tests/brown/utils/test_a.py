"""Tests for brown.utils.a module."""

import asyncio
from unittest.mock import patch

import pytest

from brown.utils.a import as_sync, asyncio_run, batch_gather, run_async_tasks, run_jobs


class TestAsSync:
    """Test the as_sync decorator."""

    @as_sync
    async def async_function(self, x: int) -> int:
        """Test async function."""
        await asyncio.sleep(0.01)
        return x * 2

    def test_as_sync_decorator(self) -> None:
        """Test that as_sync decorator converts async to sync."""
        result = self.async_function(5)
        assert result == 10

    def test_as_sync_with_exception(self) -> None:
        """Test as_sync decorator with exception handling."""

        @as_sync
        async def failing_function() -> int:
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_as_sync_invalid_function(self) -> None:
        """Test as_sync decorator with non-async function."""

        @as_sync
        def sync_function() -> int:
            return 42

        with pytest.raises(TypeError, match="must be async"):
            sync_function()


class TestAsyncioRun:
    """Test the asyncio_run function."""

    async def _test_coro(self) -> int:
        """Test coroutine."""
        await asyncio.sleep(0.01)
        return 42

    def test_asyncio_run_new_loop(self) -> None:
        """Test asyncio_run with new event loop."""
        result = asyncio_run(self._test_coro())
        assert result == 42

    def test_asyncio_run_with_exception(self) -> None:
        """Test asyncio_run with exception."""

        async def failing_coro() -> int:
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            asyncio_run(failing_coro())

    def test_asyncio_run_existing_loop(self) -> None:
        """Test asyncio_run with existing event loop."""

        # This test runs in an existing event loop (pytest-asyncio)
        async def test_coro() -> int:
            return 42

        result = asyncio_run(test_coro())
        assert result == 42


class TestRunAsyncTasks:
    """Test the run_async_tasks function."""

    async def _test_task(self) -> int:
        """Test async task."""
        await asyncio.sleep(0.01)
        return 5 * 2

    def test_run_async_tasks(self) -> None:
        """Test running multiple async tasks."""
        tasks = [self._test_task() for _ in range(5)]
        results = run_async_tasks(tasks)
        assert results == [10, 10, 10, 10, 10]

    def test_run_async_tasks_empty(self) -> None:
        """Test running empty task list."""
        results = run_async_tasks([])
        assert results == []

    def test_run_async_tasks_with_progress(self) -> None:
        """Test running tasks with progress bar."""

        async def test_task() -> int:
            await asyncio.sleep(0.01)
            return 5 * 2

        tasks = [test_task() for _ in range(3)]
        # Test that the function works with progress enabled
        results = run_async_tasks(tasks, show_progress=True)
        assert results == [10, 10, 10]


class TestBatchGather:
    """Test the batch_gather function."""

    async def _test_coro(self) -> int:
        """Test coroutine."""
        await asyncio.sleep(0.01)
        return 5 * 2

    def test_batch_gather(self) -> None:
        """Test batch gathering with size limits."""
        tasks = [self._test_coro() for _ in range(10)]
        results = asyncio_run(batch_gather(tasks, batch_size=3))
        assert results == [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    def test_batch_gather_single_batch(self) -> None:
        """Test batch gathering with large batch size."""
        tasks = [self._test_coro() for _ in range(3)]
        results = asyncio_run(batch_gather(tasks, batch_size=10))
        assert results == [10, 10, 10]

    def test_batch_gather_with_verbose(self) -> None:
        """Test batch gathering with verbose output."""
        tasks = [self._test_coro() for _ in range(3)]
        with patch("builtins.print") as mock_print:
            results = asyncio_run(batch_gather(tasks, batch_size=2, verbose=True))
            assert results == [10, 10, 10]
            # Should print progress messages
            assert mock_print.called


class TestRunJobs:
    """Test the run_jobs function."""

    async def _test_job(self) -> int:
        """Test job coroutine."""
        await asyncio.sleep(0.01)
        return 5 * 2

    def test_run_jobs(self) -> None:
        """Test running jobs with semaphore."""
        jobs = [self._test_job() for _ in range(5)]
        results = asyncio_run(run_jobs(jobs, workers=2))
        assert results == [10, 10, 10, 10, 10]

    def test_run_jobs_single_worker(self) -> None:
        """Test running jobs with single worker."""
        jobs = [self._test_job() for _ in range(3)]
        results = asyncio_run(run_jobs(jobs, workers=1))
        assert results == [10, 10, 10]

    def test_run_jobs_with_progress(self) -> None:
        """Test running jobs with progress bar."""
        jobs = [self._test_job() for _ in range(3)]
        with patch("tqdm.asyncio.tqdm_asyncio") as mock_tqdm:
            mock_tqdm.gather.return_value = asyncio.gather(*jobs)
            results = asyncio_run(run_jobs(jobs, show_progress=True, desc="Test jobs"))
            assert results == [10, 10, 10]
            mock_tqdm.gather.assert_called_once()

    def test_run_jobs_empty(self) -> None:
        """Test running empty job list."""
        results = asyncio_run(run_jobs([]))
        assert results == []

    def test_run_jobs_with_exception(self) -> None:
        """Test running jobs with exception."""

        async def failing_job() -> int:
            await asyncio.sleep(0.01)
            raise ValueError("Job failed")

        jobs = [self._test_job(), failing_job(), self._test_job()]
        with pytest.raises(ValueError, match="Job failed"):
            asyncio_run(run_jobs(jobs, workers=1))
