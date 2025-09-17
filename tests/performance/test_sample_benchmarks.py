"""
Sample performance benchmarks to validate CI/CD pipeline.
These benchmarks will be used for regression detection.
"""

import pytest
import time
import numpy as np


@pytest.mark.benchmark
def test_basic_computation_benchmark(benchmark):
    """Benchmark basic computation for regression detection."""

    def compute_operation():
        # Simple computation to establish baseline
        data = np.random.random((1000, 1000))
        result = np.dot(data, data.T)
        return result.sum()

    result = benchmark(compute_operation)
    assert result > 0


@pytest.mark.benchmark
@pytest.mark.slow
def test_memory_allocation_benchmark(benchmark):
    """Benchmark memory allocation patterns."""

    def allocate_memory():
        # Test memory allocation performance
        arrays = []
        for i in range(100):
            arr = np.zeros((1000, 1000))
            arrays.append(arr)
        return len(arrays)

    result = benchmark(allocate_memory)
    assert result == 100


@pytest.mark.benchmark
def test_string_operations_benchmark(benchmark):
    """Benchmark string operations for baseline."""

    def string_operations():
        # Test string manipulation performance
        data = []
        for i in range(10000):
            s = f"test_string_{i}_benchmark"
            data.append(s.upper().lower().split('_'))
        return len(data)

    result = benchmark(string_operations)
    assert result == 10000


@pytest.mark.benchmark
@pytest.mark.gpu
def test_gpu_computation_benchmark(benchmark):
    """Benchmark GPU computation if available."""

    def gpu_computation():
        try:
            import torch
            if torch.cuda.is_available():
                # Simple GPU computation
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.randn(1000, 1000, device='cuda')
                result = torch.mm(x, y)
                return result.sum().item()
            else:
                # Fallback to CPU
                x = torch.randn(1000, 1000)
                y = torch.randn(1000, 1000)
                result = torch.mm(x, y)
                return result.sum().item()
        except ImportError:
            # PyTorch not available - use numpy
            x = np.random.randn(1000, 1000)
            y = np.random.randn(1000, 1000)
            result = np.dot(x, y)
            return result.sum()

    result = benchmark(gpu_computation)
    assert isinstance(result, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only"])