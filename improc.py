import numpy as np
import cv2


def preprocess(img: np.ndarray) -> np.ndarray:
    """pre-process the image to match the mnist dataset.

    Arguments:
        img {np.ndarray}
            -- the image to be preprocessed.

    Returns:
        np.ndarray
            -- the preprocessed image.
    """
    img = cv2.resize(img, (28, 28))
    return img.astype(dtype=np.uint8)
