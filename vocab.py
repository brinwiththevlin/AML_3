from typing import List, Optional
import numpy as np
import vectorize


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

    vectors: List[np.ndarray] = [vectorize.vectorize(f, vocab) for f in files]
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


def findStopWords(vectors: List[np.ndarray]) -> list:
    """returns the index of the stop words"""
    top_ten_sets = [set(np.argsort(v)[:5]) for v in vectors]
    common_indexes = set.intersection(*top_ten_sets)
    return list(common_indexes)
