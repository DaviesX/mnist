import numpy as np


def to_one_hot(labels: np.ndarray, num_categories: int) -> np.ndarray:
    """Employ one hot encoding to categorical data.

    Arguments:
        labels {np.ndarray}
            -- labels to be encoded.
        num_categories {int}
            -- number of categories possible.

    Returns:
        np.ndarray
            -- the encoded labels.
    """

    m = np.zeros([num_categories, num_categories], dtype=np.int64)
    for i in range(num_categories):
        m[i][i] = 1
    return m[labels]


if __name__ == "__main__":
    print(to_one_hot(np.array([0, 3, 2, 1, 4, 5, 6, 7, 8, 9]), 10))
