import time

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


def benchmark_function(iterations, func, *args, **kwargs):
    """
    Benchmarks a function over different iteration counts
    and plots the execution times.

    Parameters:
        iterations (list[int]):
            List of iteration counts.

        func (callable):
            Function to benchmark.

        *args:
            Positional arguments passed to the function.

        **kwargs:
            Keyword arguments passed to the function.
    """

    times = []

    for n in iterations:
        start = time.perf_counter()

        for _ in range(n):
            func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"{n} iterations -> {elapsed:.6f} sec")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, times, marker='o')

    plt.title(f"Performance of {func.__name__}")
    plt.xlabel("Iterations")
    plt.ylabel("Time (seconds)")
    plt.grid(True)

    plt.show()

    return times