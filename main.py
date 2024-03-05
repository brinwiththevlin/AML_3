import kernel
import vectorize
import vocab
import numpy as np
import os
from typing import List


if __name__ == "__main__":
    # list of filenames
    files = [f"./data/{f}" for f in os.listdir("./data")]
    files.remove("./data/.DS_Store")
    class_sizes = vectorize.classCounts(files)
    class_keys = sorted(list(class_sizes.keys()))

    """part 1"""
    print("PART 1!!:")
    # dictionary of terms across all files
    D0 = vocab.dictionary(files)
    # re-vectorize with revised dictionay
    vectors: List[np.ndarray] = [vectorize.vectorize(f, D0) for f in files]

    # D is the Document-term matrix
    D = np.vstack(vectors)

    # K is the kernel matrix
    K0 = kernel.buildKernel(D, "dot")
    print("dot kernel")
    sub_matrices = kernel.extractSubmatrices(K0, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    Kp = kernel.buildKernel(D, "poly")

    print("poly kernel")
    sub_matrices = kernel.extractSubmatrices(Kp, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    """part 2"""
    print("PART 2!!:")
    # dictionary of terms across all files
    D1 = vocab.dictionary(files, stem=True)
    # re-vectorize with revised dictionay
    vectors: List[np.ndarray] = [vectorize.vectorize(f, D1) for f in files]

    # D is the Document-term matrix
    D = np.vstack(vectors)

    # K is the kernel matrix
    K0 = kernel.buildKernel(D, "dot")
    print("dot kernel")
    sub_matrices = kernel.extractSubmatrices(K0, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    Kp = kernel.buildKernel(D, "poly")

    print("poly kernel")
    sub_matrices = kernel.extractSubmatrices(Kp, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    """part 3"""
    print("PART 3!!:")
    # dictionary of terms across all files
    D2 = vocab.dictionary(files, stem=True)
    # re-vectorize with revised dictionay
    vectors: List[np.ndarray] = [vectorize.vectorize(f, D2) for f in files]

    # D is the Document-term matrix
    D = np.vstack(vectors)

    # K is the kernel matrix
    K0 = kernel.buildKernel(D, "dot")
    print("dot kernel")
    sub_matrices = kernel.extractSubmatrices(K0, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    Kp = kernel.buildKernel(D, "poly")

    print("poly kernel")
    sub_matrices = kernel.extractSubmatrices(Kp, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label)
        for class_label, kernel_mat in sub_matrices.items()
    ]
