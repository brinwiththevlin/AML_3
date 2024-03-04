import utils as u
import numpy as np
import os
from typing import List


if __name__ == "__main__":
    # list of filenames
    files = [f"./data/{f}" for f in os.listdir("./data")]
    files.remove("./data/.DS_Store")
    class_sizes = u.classCounts(files)
    class_keys = sorted(list(class_sizes.keys()))

    """part 1"""
    print("PART 1!!:")
    # dictionary of terms across all files
    D0 = u.dictionary(files)
    # re-vectorize with revised dictionay
    vectors: List[np.ndarray] = [u.vectorize(f, D0) for f in files]

    # D is the Document-term matrix
    D = np.vstack(vectors)

    # K is the kernel matrix
    K0 = u.buildKernel(D, "dot")
    print("dot kernel")
    sub_matrices = u.extractSubmatrices(K0, class_sizes, class_keys)
    [
        u.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    Kp = u.buildKernel(D, "poly")

    print("poly kernel")
    sub_matrices = u.extractSubmatrices(Kp, class_sizes, class_keys)
    [
        u.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    """part 2"""

    print("PART 2!!:")
    # dictionary of terms across all files
    D1 = u.dictionary(files, stem=True)
    # re-vectorize with revised dictionay
    vectors: List[np.ndarray] = [u.vectorize(f, D1) for f in files]

    # D is the Document-term matrix
    D = np.vstack(vectors)

    # K is the kernel matrix
    K0 = u.buildKernel(D, "dot")
    print("dot kernel")
    sub_matrices = u.extractSubmatrices(K0, class_sizes, class_keys)
    [
        u.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    Kp = u.buildKernel(D, "poly")

    print("poly kernel")
    sub_matrices = u.extractSubmatrices(Kp, class_sizes, class_keys)
    [
        u.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]
    """part 3"""
    print("PART 3!!:")
    # dictionary of terms across all files
    D1 = u.dictionary(files, adv_stem=True)
    # re-vectorize with revised dictionay
    vectors: List[np.ndarray] = [u.vectorize(f, D1) for f in files]

    # D is the Document-term matrix
    D = np.vstack(vectors)

    # K is the kernel matrix
    K0 = u.buildKernel(D, "dot")
    print("dot kernel")
    sub_matrices = u.extractSubmatrices(K0, class_sizes, class_keys)
    [
        u.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    Kp = u.buildKernel(D, "poly")

    print("poly kernel")
    sub_matrices = u.extractSubmatrices(Kp, class_sizes, class_keys)
    [
        u.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]
