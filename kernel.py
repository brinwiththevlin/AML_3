import numpy as np
from collections import Counter
from typing import List, Literal, TextIO


def buildKernel(D: np.ndarray, type: Literal["dot", "poly"]):
    if type == "dot":

        return np.matmul(D, D.T)
    else:
        return (np.matmul(D, D.T) + 1) ** 2


def extractSubmatrices(
    K: np.ndarray, class_sizes: Counter, class_keys: List[int]
) -> np.ndarray:

    starts = [0]
    for key in class_keys[:-1]:
        starts.append(starts[-1] + class_sizes[key])

    sub_matrices = {
        key: K[
            starts[i]: starts[i] + class_sizes[key],
            starts[i]: starts[i] + class_sizes[key],
        ]
        for i, key in enumerate(class_keys)
    }
    return sub_matrices


def descriptiveStats(kernel_mat: np.ndarray, class_label: str, results: TextIO) -> None:
    mean_value = np.mean(kernel_mat)
    median_value = np.median(kernel_mat)
    std_deviation = np.std(kernel_mat)
    min_value = np.min(kernel_mat)
    max_value = np.max(kernel_mat)

    results.writelines(f"class_matrix: {class_label}\n")
    results.writelines(f"Mean: {mean_value}\n")
    results.writelines(f"Median: {median_value}\n")
    results.writelines(f"Standard Deviation: {std_deviation}\n")
    results.writelines(f"Minimum Value: {min_value}\n")
    results.writelines(f"Maximum Value: {max_value}\n")
    results.writelines("\n")

