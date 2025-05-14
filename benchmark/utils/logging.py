from typing import Dict, List, Any
import numpy as np
import pandas as pd
from tabulate import tabulate


def calculate_and_print_statistics(metrics: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Calculate statistics for a dictionary of metrics and print them in a tabular format.

    Args:
        metrics: Dictionary where keys are metric names and values are lists of metric values

    Returns:
        pandas.DataFrame: DataFrame containing the calculated statistics
    """
    metric_names = []
    stats_data = []

    # Calculate statistics for each metric
    for metric_name, values in metrics.items():
        metric_names.append(metric_name)
        stats_data.append(
            {
                "Mean": np.mean(values),
                "Std Dev": np.std(values),
                "Min": np.min(values),
                "P25": np.percentile(values, 25),
                "Median": np.median(values),
                "P75": np.percentile(values, 75),
                "Max": np.max(values),
            }
        )

    stats_df = pd.DataFrame(stats_data, index=metric_names)
    print(tabulate(stats_df.round(2), headers="keys", tablefmt="pretty"))

    return stats_df
