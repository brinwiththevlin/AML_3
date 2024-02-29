import numpy as np
from collections import Counter
import re
from typing import List, Literal, Optional, Dict


def buildKernel(D: np.ndarray, type: Literal["dot", "poly"]):
    if type == "dot":
        return np.matmul(D, D.T)
    else:
        return (np.matmul(D, D.T) + 1) ** 2


def dictionary(
    files: List[str],
    stem: Optional[bool] = None,
    adv_stem: Optional[Dict[str, str]] = None,
):
    assert type(stem) in [bool, None], "stem must be boolean"

    # dictionary of terms across all files
    vocabulary = set()
    [vocabulary := vocabulary.union(set(open(f).read().split())) for f in files]
    vocabulary = list(vocabulary)

    vectors: List[np.ndarray] = [vectorize(f, vocabulary) for f in files]
    # revise dictionary
    stop_indexes = findStopWords(vectors)

    if stem:
        # TODO: implement stemming for part 2
        pass
    elif adv_stem:
        # TODO: implement stemming for part 3
        pass
    vocabulary = [
        vocabulary[i] for i in range(len(vocabulary)) if i not in stop_indexes
    ]

    return vocabulary


def vectorize(filename: str, dict: List[str]) -> np.ndarray:
    f = open(filename)
    content = f.read().split()
    vector = [content.count(word) for word in dict]
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


def extractSubmatrices(
    K: np.ndarray, class_sizes: Counter, class_keys: List[int]
) -> np.ndarray:

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
