import kernel
import spectral
import vectorize
import vocab
import numpy as np
import os
from typing import List


def silhouette(K: np.ndarray, clusters: np.ndarray) -> float:
    """
    Compute the silhouette score
    """
    # compute average distane between points in the same cluster
    d = np.zeros(K.shape[0])
    for i in range(K.shape[0]):
        d[i] = np.mean(K[i, clusters == clusters[i]])

    # compute the smallest average distance between a point and any other cluster
    D = np.zeros(K.shape[0])
    for i in range(K.shape[0]):
        D[i] = np.min(
            [np.mean(K[i, clusters == j])
             for j in set(clusters) if j != clusters[i]]
        )

    # compute the silhouette score
    s = (D - d) / np.maximum(d, D)
    return np.mean(s)


if __name__ == "__main__":

    # list of filenames
    files = [f"./data/{f}" for f in os.listdir("./data")]
    files.remove("./data/.DS_Store")
    class_sizes = vectorize.classCounts(files)
    class_keys = sorted(list(class_sizes.keys()))
    results = open("results.txt", "w")
    """part 1"""
    results.writelines("PART 1!!:\n")
    # dictionary of terms across all files
    D0 = vocab.dictionary(files)
    # re-vectorize with revised dictionay
    vectors: List[np.ndarray] = [vectorize.vectorize(f, D0) for f in files]

    # D is the Document-term matrix
    D = np.vstack(vectors)

    # K is the kernel matrix
    K0 = kernel.buildKernel(D, "dot")
    results.writelines("dot kernel\n")
    sub_matrices = kernel.extractSubmatrices(K0, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label, results)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    for k in range(2, 5):
        clusters = spectral.spectralClustering(K0, k)
        silhouette_score = silhouette(K0, clusters)
        results.writelines(f"spectral clustering k={k}\n")
        results.writelines(f"silhouette score: {silhouette_score}\n")
        results.writelines("\n")

    Kp = kernel.buildKernel(D, "poly")

    results.writelines("poly kernel\n")
    sub_matrices = kernel.extractSubmatrices(Kp, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label, results)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    for k in range(2, 5):
        clusters = spectral.spectralClustering(Kp, k)
        silhouette_score = silhouette(Kp, clusters)
        results.writelines(f"spectral clustering k={k}\n")
        results.writelines(f"silhouette score: {silhouette_score}\n")
        results.writelines("\n")

    """part 2"""
    results.writelines("PART 2!!:\n")
    # dictionary of terms across all files
    D1 = vocab.dictionary(files, stem=True)
    # re-vectorize with revised dictionay
    vectors: List[np.ndarray] = [vectorize.vectorize(f, D1) for f in files]

    # D is the Document-term matrix
    D = np.vstack(vectors)

    # K is the kernel matrix
    K0 = kernel.buildKernel(D, "dot")
    results.writelines("dot kernel\n")
    sub_matrices = kernel.extractSubmatrices(K0, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label, results)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    for k in range(2, 5):
        clusters = spectral.spectralClustering(K0, k)
        silhouette_score = silhouette(K0, clusters)
        results.writelines(f"spectral clustering k={k}\n")
        results.writelines(f"silhouette score: {silhouette_score}\n")
        results.writelines("\n")

    Kp = kernel.buildKernel(D, "poly")

    results.writelines("poly kernel\n")
    sub_matrices = kernel.extractSubmatrices(Kp, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label, results)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    for k in range(2, 5):
        clusters = spectral.spectralClustering(Kp, k)
        silhouette_score = silhouette(Kp, clusters)
        results.writelines(f"spectral clustering k={k}\n")
        results.writelines(f"silhouette score: {silhouette_score}\n")
        results.writelines("\n")

    """part 3"""
    results.writelines("PART 3!!:\n")
    # dictionary of terms across all files
    D2 = vocab.dictionary(files, stem=True)
    # re-vectorize with revised dictionay
    vectors: List[np.ndarray] = [vectorize.vectorize(f, D2) for f in files]

    # D is the Document-term matrix
    D = np.vstack(vectors)

    # K is the kernel matrix
    K0 = kernel.buildKernel(D, "dot")
    results.writelines("dot kernel\n")
    sub_matrices = kernel.extractSubmatrices(K0, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label, results)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    for k in range(2, 5):
        clusters = spectral.spectralClustering(K0, k)
        silhouette_score = silhouette(K0, clusters)
        results.writelines(f"spectral clustering k={k}\n")
        results.writelines(f"silhouette score: {silhouette_score}\n")
        results.writelines("\n")

    Kp = kernel.buildKernel(D, "poly")

    results.writelines("poly kernel\n")
    sub_matrices = kernel.extractSubmatrices(Kp, class_sizes, class_keys)
    [
        kernel.descriptiveStats(kernel_mat, class_label, results)
        for class_label, kernel_mat in sub_matrices.items()
    ]

    for k in range(2, 5):
        clusters = spectral.spectralClustering(Kp, k)
        silhouette_score = silhouette(Kp, clusters)
        results.writelines(f"spectral clustering k={k}\n")
        results.writelines(f"silhouette score: {silhouette_score}\n")
        results.writelines("\n")

    results.close()
