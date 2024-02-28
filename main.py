import numpy as np
import os
from typing import List


def vectorize(filename: str, D0: list) -> np.ndarray:
    f = open(filename)
    content = f.read().split()
    vector = [content.count(word) for word in D0]
    np_vector = np.array(vector)
    return np_vector


def find_stop_words(vectors: List[np.ndarray], D0: list) -> list:
    """ returns the index of the stop words """
    top_ten_sets = [set(np.argsort(v)[:10]) for v in vectors]
    common_indexes = set.intersection(*top_ten_sets)
    return list(common_indexes)


if __name__ == "__main__":
    # list of filenames
    files = [f"./data/{f}" for f in os.listdir("./data")]
    files.remove("./data/.DS_Store")

    # dictionary of terms across all files
    D0 = set()
    [D0 := D0.union(set(open(f).read().split())) for f in files]
    D0 = list(D0)

    vectors: List[np.ndarray] = [vectorize(f, D0) for f in files]

    # revise dictionary
    stop_indexes = find_stop_words(vectors, D0)
    D0 = [D0[i] for i in len(D0) if i not in stop_indexes]

    # re-vectorize with revised dictionary
    vectors: List[np.ndarray] = [vectorize(f, D0) for f in files]
