"""Tests for brown.evals.tasks module."""

from pathlib import Path

from brown.evals.tasks import create_evaluation_task, evaluation_task


class TestEvaluationTask:
    """Test the evaluation_task function."""

    def test_evaluation_task_creation(self) -> None:
        """Test creating an evaluation task function."""
        task_func = create_evaluation_task(
            cache_dir=Path("/tmp/test"),
            read_from_cache=False,
            clean_cache=False,
            debug=False,
        )

        assert callable(task_func)
        # The function is a partial, so it doesn't have __name__ attribute
        assert hasattr(task_func, "func")  # partial functions have a 'func' attribute

    def test_evaluation_task_function_signature(self) -> None:
        """Test that evaluation_task has the correct signature."""
        import inspect

        sig = inspect.signature(evaluation_task)
        params = list(sig.parameters.keys())

        expected_params = ["sample", "cache_dir", "read_from_cache", "clean_cache", "debug"]

        for param in expected_params:
            assert param in params

    def test_create_evaluation_task_parameters(self) -> None:
        """Test create_evaluation_task with different parameters."""
        # Test with different parameter combinations
        task1 = create_evaluation_task(
            cache_dir=Path("/tmp/test1"),
            read_from_cache=True,
            clean_cache=True,
            debug=True,
        )

        task2 = create_evaluation_task(
            cache_dir=Path("/tmp/test2"),
            read_from_cache=False,
            clean_cache=False,
            debug=False,
        )

        assert callable(task1)
        assert callable(task2)
        assert task1 != task2  # Different partial functions

    def test_evaluation_task_async(self) -> None:
        """Test that evaluation_task is a regular function (not coroutine)."""
        import inspect

        # evaluation_task is decorated with @a.as_sync, so it's not a coroutine function
        assert not inspect.iscoroutinefunction(evaluation_task)
        assert callable(evaluation_task)
