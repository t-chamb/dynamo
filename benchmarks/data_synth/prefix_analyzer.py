import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm


class PrefixAnalyzer:
    """
    A class for analyzing dataset characteristics related to prefixes, hash IDs, and cache hit rates.
    """

    def __init__(self, dataset_path, block_size=1):
        """
        Initialize the analyzer with dataset path and block size.

        Args:
            dataset_path: Path to the JSONL dataset file
            block_size: Size of each block for prefix calculation
        """
        self.dataset_path = dataset_path
        self.block_size = block_size
        self.dataset = self._load_dataset()
        self.hash_counter = self._build_hash_counter()
        self.repeated_hash_ids = self._find_repeated_hash_ids()

    def _load_dataset(self):
        print(f"Loading dataset from {self.dataset_path}...")
        dataset = []
        with open(self.dataset_path, "r") as f:
            for line in f:
                dataset.append(json.loads(line))
        print(f"Dataset loaded: {len(dataset)} examples")
        return dataset

    def _build_hash_counter(self):
        all_hash_ids = []
        for item in tqdm(self.dataset, desc="Processing hash IDs"):
            all_hash_ids.extend(item["hash_ids"])
        counter = Counter(all_hash_ids)
        print(f"Hash counter built: {len(counter)} unique hash IDs")
        return counter

    def _find_repeated_hash_ids(self):
        return {hash_id for hash_id, count in self.hash_counter.items() if count > 1}

    def analyze_dataset_lengths(self):
        """
        Analyze dataset to extract various length metrics and print statistics.

        Returns:
            Tuple of lists: (input_lengths, prefix_lengths, user_prompt_lengths, output_lengths)
        """
        # Extract input and output lengths directly from fields
        input_lengths = [item["input_length"] for item in self.dataset]
        output_lengths = [item["output_length"] for item in self.dataset]

        # Calculate prefix length and user prompt length for each row
        prefix_lengths = []
        user_prompt_lengths = []

        for i, item in tqdm(
            enumerate(self.dataset),
            total=len(self.dataset),
            desc="Analyzing dataset lengths",
        ):
            input_len = item["input_length"]
            hash_ids = item["hash_ids"]
            assert len(hash_ids) * self.block_size >= input_len

            # Special case: if all hash IDs in the row are repeated elsewhere
            if all(hash_id in self.repeated_hash_ids for hash_id in hash_ids):
                prefix_len = input_len  # Set prefix length to input length
                user_prompt_len = 0  # Set user prompt length to 0
            else:
                # Count how many hash IDs in this row are repeated elsewhere in the dataset
                repeated_count = sum(
                    1 for hash_id in hash_ids if hash_id in self.repeated_hash_ids
                )
                prefix_len = repeated_count * self.block_size
                user_prompt_len = input_len - prefix_len

            prefix_lengths.append(prefix_len)
            user_prompt_lengths.append(user_prompt_len)

            # Check if prefix length is greater than input length
            if prefix_len > input_len:
                print(f"WARNING: Line {i}: {json.dumps(item)}")

        # Print statistics table
        metrics = {
            "Input Length": input_lengths,
            "Prefix Length": prefix_lengths,
            "User Prompt Length": user_prompt_lengths,
            "Output Length": output_lengths,
        }

        print_statistics_table(metrics)

        return input_lengths, prefix_lengths, user_prompt_lengths, output_lengths

    def visited_radix_lens(self, ax=None, legend=None):
        """
        Analyze radix lengths based on hash IDs with the same repetition count as the first hash.

        Args:
            ax: Matplotlib axis handle for plotting (if None, creates a new figure)
            legend: Legend label for this dataset in the plot

        Returns:
            The matplotlib axis handle
        """
        # For each row, calculate the radix length based on the first hash's repetition count
        radix_lengths = []

        for item in tqdm(self.dataset):
            # Skip if there are no hash_ids
            if len(item["hash_ids"]) == 0:
                continue

            # Get the repetition count of the first hash ID
            first_hash = item["hash_ids"][0]
            first_hash_repetition_count = self.hash_counter[first_hash]
            if first_hash_repetition_count <= 1:
                continue

            # Count how many hash IDs in this row have the same repetition count
            matching_hash_ids = sum(
                1
                for hash_id in item["hash_ids"]
                if self.hash_counter[hash_id] == first_hash_repetition_count
            )

            # Calculate radix length
            radix_length = matching_hash_ids * self.block_size
            radix_lengths.append((first_hash, radix_length))

        # Count occurrences of each (first_hash, radix_length) tuple
        radix_lens_counter = Counter(radix_lengths)

        # Create a new figure if no axis is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Extract x and y values for plotting
        x_values = [tup[1] for tup in radix_lens_counter.keys()]
        y_values = list(radix_lens_counter.values())

        # Plot with legend if provided
        scatter = ax.scatter(
            x_values, y_values, alpha=0.7, label=legend + " (trunk)" if legend else None
        )

        # Now analyze based on the last hash with nonzero repeat count
        radix_lengths_last = []
        for item in tqdm(self.dataset):
            # Skip if there are no hash_ids
            if len(item["hash_ids"]) == 0:
                continue

            # Find the last hash with nonzero repeat count
            last_hash = None
            for hash_id in reversed(item["hash_ids"]):
                if self.hash_counter[hash_id] > 1:
                    last_hash = hash_id
                    break

            if last_hash is None:
                continue

            last_hash_repetition_count = self.hash_counter[last_hash]

            # Count how many hash IDs in this row have repeat count greater than the last hash's
            matching_hash_ids = sum(
                1
                for hash_id in item["hash_ids"]
                if self.hash_counter[hash_id] > last_hash_repetition_count
            )

            # Calculate radix length
            radix_length = matching_hash_ids * self.block_size
            radix_lengths_last.append((last_hash, radix_length))

        # Count occurrences of each (last_hash, radix_length) tuple
        radix_lens_last_counter = Counter(radix_lengths_last)

        # Extract x and y values for plotting
        x_values_last = [tup[1] for tup in radix_lens_last_counter.keys()]
        y_values_last = list(radix_lens_last_counter.values())

        # Plot with legend if provided
        scatter_last = ax.scatter(
            x_values_last,
            y_values_last,
            alpha=0.7,
            marker="x",
            label=legend + " (branch)" if legend else None,
        )

        # Add legend if any labels exist
        if legend is not None:
            ax.legend()

        ax.set_xlabel("Radix Length")
        ax.set_ylabel("Visited")
        ax.set_xscale("log")
        # ax.set_yscale('log')
        ax.grid(True, linestyle="--", alpha=0.7)

        # Return the axis for further modifications
        return ax

    def analyze_cache_hit_rates(self):
        """
        Analyze theoretical cache hit rates based on hash ID repetition.

        Returns:
            List of cache hit rates for each row in the dataset
        """
        # Set to track all hash IDs we've seen
        seen_hash_ids = set()

        # Store cache hit rates for each row
        cache_hit_rates = []

        for item in tqdm(self.dataset, desc="Calculating cache hit rates"):
            hash_ids = item["hash_ids"]

            # Skip if there are no hash IDs
            if len(hash_ids) == 0:
                continue

            # Find the first index where the hash ID hasn't been seen before
            first_unseen_idx = len(hash_ids)  # Default if all are seen
            for idx, hash_id in enumerate(hash_ids):
                if hash_id not in seen_hash_ids:
                    first_unseen_idx = idx
                    break

            # Calculate cache hit rate
            cache_hit_rate = first_unseen_idx / len(hash_ids)
            cache_hit_rates.append(cache_hit_rate)

            # Add all hash IDs from this row to the seen set
            seen_hash_ids.update(hash_ids)

        # Create histogram of cache hit rates
        plt.figure(figsize=(10, 6))
        plt.hist(
            cache_hit_rates, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        plt.xlabel("Cache Hit Rate")
        plt.ylabel("Frequency")
        plt.title("Theoretical Cache Hit Rates")

        # Add statistics text to the plot directly
        stats_text = (
            f"Mean: {np.mean(cache_hit_rates):.4f}\n"
            f"Median: {np.median(cache_hit_rates):.4f}\n"
            f"Min: {np.min(cache_hit_rates):.4f}\n"
            f"Max: {np.max(cache_hit_rates):.4f}\n"
            f"Std Dev: {np.std(cache_hit_rates):.4f}"
        )

        # Position the text in the upper right corner with some padding
        plt.text(
            0.95,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig("theoretical_hit_rates.png", dpi=300, bbox_inches="tight")
        plt.close()

        return cache_hit_rates


def print_statistics_table(metrics):
    """
    Print a formatted table of statistics for the given metrics.

    Args:
        metrics: Dictionary mapping metric names to lists of values
    """
    stats_data = []
    for metric_name, values in metrics.items():
        stats_data.append(
            {
                "Metric": metric_name,
                "Mean": np.mean(values),
                "Std Dev": np.std(values),
                "Min": np.min(values),
                "P25": np.percentile(values, 25),
                "Median": np.median(values),
                "P75": np.percentile(values, 75),
                "Max": np.max(values),
            }
        )

    # Create DataFrame from the collected statistics
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.set_index("Metric")
    stats_df = stats_df.round(2)

    # Print the table using tabulate with a pretty format
    print(tabulate(stats_df, headers="keys", tablefmt="pretty"))


if __name__ == "__main__":
    # Main routine that uses the specified dataset with block size of 16
    block_size = 16
    dataset_path = f"../datasets/avian_r100000_bs{block_size}_synth.jsonl"
    # dataset_path = "/home/rupei/nova-benchmarking/datasets/gen_prompts_32k_2_languages_16.jsonl"

    print(f"Analyzing dataset: {dataset_path}")
    print(f"Using block size: {block_size}")
    print()

    # Create analyzer instance
    analyzer = PrefixAnalyzer(dataset_path, block_size=block_size)

    # Run analyses
    input_lens, prefix_lens, user_prompt_lens, output_lens = (
        analyzer.analyze_dataset_lengths()
    )
    analyzer.analyze_cache_hit_rates()

    print(f"\nAnalysis complete. Processed {len(input_lens)} examples.")
