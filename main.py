# %%
import numpy as np
from pprint import pprint
import os
import pandas as pd
import re
from typing import List
from collections import Counter


def vectorize(filename: str) -> np.ndarray:
    f = open(filename)
    content = f.read().split()
    vector = [content.count(word) for word in D0]
    np_vector = np.array(vector)
    return np_vector


def findStopWords(vectors: List[np.ndarray]) -> list:
    """returns the index of the stop words"""
    top_ten_sets = [set(np.argsort(v)[:3]) for v in vectors]
    common_indexes = set.intersection(*top_ten_sets)
    return list(common_indexes)


def classCounts(files: List[str]) -> Counter:
    # Regular expression pattern to extract class information
    pattern = r"(\d+)-"

    # Use list comprehension to extract class information from each file name
    class_numbers = [int(re.search(pattern, file_name).group(1)) for file_name in files]
    return Counter(class_numbers)


def extractSubmatrices(K):

    starts = [0]
    for key in class_keys[:-1]:
        starts.append(starts[-1] + class_sizes[key])

    sub_matrices = {
        key: K[
            starts[i] : starts[i] + class_sizes[key],
            starts[i] : starts[i] + class_sizes[key],
        ]
        for i, key in enumerate(class_keys)
    }
    return sub_matrices


def descriptiveStats(kernel_mat, class_label):
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


if __name__ == "__main__":
    # list of filenames
    files = [f"./data/{f}" for f in os.listdir("./data")]
    files.remove("./data/.DS_Store")
    class_sizes = classCounts(files)
    class_keys = sorted(list(class_sizes.keys()))
    print(class_keys)

    # dictionary of terms across all files
    D0 = set()
    [D0 := D0.union(set(open(f).read().split())) for f in files]
    D0 = list(D0)

    vectors: List[np.ndarray] = [vectorize(f) for f in files]

    # revise dictionary
    stop_indexes = findStopWords(vectors)
    D0 = [D0[i] for i in range(len(D0)) if i not in stop_indexes]

    # re-vectorize with revised dictionary
    vectors: List[np.ndarray] = [vectorize(f) for f in files]

    # D is the Document-term matrix
    D = np.vstack(vectors)
    # K is the kernel matrix
    K0 = np.matmul(D, D.T)

    sub_matrices = extractSubmatrices(K0)
    [
        descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]
# %%
