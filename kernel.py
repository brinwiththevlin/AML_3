import numpy as np
from collections import Counter
from typing import List, Literal


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


def descriptiveStats(kernel_mat: np.ndarray, class_label: str) -> None:
    mean_value = np.mean(kernel_mat)
    median_value = np.median(kernel_mat)
    std_deviation = np.std(kernel_mat)
    min_value = np.min(kernel_mat)
    max_value = np.max(kernel_mat)

    print("class_matrix", class_label)
    print("Mean:", mean_value)
    print("Median:", median_value)
    print("Standard Deviation:", std_deviation)
    print("Minimum Value:", min_value)
    print("Maximum Value:", max_value)
    print()
