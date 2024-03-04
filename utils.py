import numpy as np
from collections import Counter
import re
from typing import List, Literal, Optional, Dict


def buildKernel(D: np.ndarray, type: Literal["dot", "poly"]):
    if type == "dot":
        kernel = np.matmul(D, D.T)

        return np.matmul(D, D.T)
    else:
        return (np.matmul(D, D.T) + 1) ** 2


def part2_stem(vocab: List[str]):
    add = [term for term in vocab if "add" in term]
    sub = [term for term in vocab if "sub" in term]
    mul = [term for term in vocab if "mul" in term]
    div = [term for term in vocab if "div" in term]
    jump = [term for term in vocab if term.startswith("j")]
    push = [term for term in vocab if "push" in term]
    mov = [term for term in vocab if "mov" in term]
    num = [term for term in vocab if term.isnumeric()]

    new_vocab = {}
    new_vocab.update(dict.fromkeys(add, "add"))
    new_vocab.update(dict.fromkeys(sub, "sub"))
    new_vocab.update(dict.fromkeys(mul, "mul"))
    new_vocab.update(dict.fromkeys(div, "div"))
    new_vocab.update(dict.fromkeys(jump, "jump"))
    new_vocab.update(dict.fromkeys(push, "push"))
    new_vocab.update(dict.fromkeys(mov, "mov"))
    new_vocab.update(dict.fromkeys(num, "num"))
    return new_vocab


def part3_stem(vocab: List[str]):
    arith = [
        term for term in vocab if any(s in term for s in ["add", "sub", "mul", "div"])
    ]
    jump = [term for term in vocab if term.startswith("j")]
    data = [term for term in vocab if any(s in term for s in ["mov", "push"])]
    num = [term for term in vocab if term.isnumeric()]

    new_vocab = {}
    new_vocab.update(dict.fromkeys(num, "num"))
    new_vocab.update(dict.fromkeys(arith, "arith"))
    new_vocab.update(dict.fromkeys(jump, "jump"))
    new_vocab.update(dict.fromkeys(data, "data"))

    return new_vocab


def dictionary(
    files: List[str],
    stem: Optional[bool] = False,
    adv_stem: Optional[bool] = False,
):
    assert type(stem) in [bool, False], "stem must be boolean"
    assert type(adv_stem) in [bool, False], "adv_stem must be boolean"

    # dictionary of terms across all files
    vocab = set()
    [vocab := vocab.union(set(open(f).read().split())) for f in files]
    vocab = list(vocab)

    vectors: List[np.ndarray] = [vectorize(f, vocab) for f in files]
    # revise dictionary
    stop_indexes = findStopWords(vectors)

    vocab = [vocab[i] for i in range(len(vocab)) if i not in stop_indexes]

    if stem:
        # TODO: implement stemming for part 2
        vocab = part2_stem(vocab)
    elif adv_stem:
        # TODO: implement stemming for part 3
        vocab = part3_stem(vocab)

    return vocab


def vectorize(filename: str, dict: List[str]) -> np.ndarray:
    f = open(filename)
    content = f.read().split()
    vector = [content.count(word) for word in dict]
    np_vector = np.array(vector)
    # normalize to avoid overflow errors
    return np_vector / np.linalg.norm(np_vector)


def findStopWords(vectors: List[np.ndarray]) -> list:
    """returns the index of the stop words"""
    top_ten_sets = [set(np.argsort(v)[:5]) for v in vectors]
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
