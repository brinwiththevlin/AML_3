from collections import Counter
import numpy as np
from typing import Literal, List
import re


def classCounts(files: List[str]) -> Counter:
    # Regular expression pattern to extract class information
    pattern = r"(\d+)-"

    # Use list comprehension to extract class information from each file name
    class_numbers = [int(re.search(pattern, file_name).group(1)) for file_name in files]
    return Counter(class_numbers)


def vectorize(filename: str, dict: List[str]) -> np.ndarray:
    f = open(filename)
    content = f.read().split()
    vector = [content.count(word) for word in dict]
    np_vector = np.array(vector)
    # normalize to avoid overflow errors
    return np_vector / np.linalg.norm(np_vector)
