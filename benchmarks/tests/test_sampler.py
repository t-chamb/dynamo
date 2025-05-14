import numpy as np
from collections import Counter

from benchmarks.data_synth.sampler import EmpiricalSampler


def test_empirical_sampler_distribution():
    # Create a test array with equal numbers of 1, 2, and 3
    test_data = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])

    # Create the sampler
    sampler = EmpiricalSampler(test_data)

    # Sample 1000 times
    samples = [sampler.sample() for _ in range(1000)]

    # Count occurrences of each value
    counts = Counter(samples)

    # Verify each number (1, 2, 3) appears between 300 and 400 times
    for value in [1, 2, 3]:
        assert (
            300 <= counts[value] <= 400
        ), f"Value {value} appeared {counts[value]} times, expected 300-400 times"

    # Verify no other values appear in the samples
    assert set(counts.keys()) == {
        1,
        2,
        3,
    }, f"Unexpected values in samples: {set(counts.keys()) - {1, 2, 3}}"
