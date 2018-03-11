import numpy as np


class if_mnist:
    """mnist classifier interface
    """

    def __init__(self):
        pass

    def name(self) -> str:
        """model name.
        """
        pass

    def fit(self,
            tr_imgs: np.ndarray, tr_labels: np.ndarray,
            te_imgs: np.ndarray, te_labels: np.ndarray,
            sess_file: str) -> None:
        """fitting a model.

        Arguments:
            tr_imgs {np.ndarray}
                -- training images.
            tr_labels {np.ndarray} 
                -- training labels.
            te_imgs {np.ndarray}
                -- testing images.
            te_labels {np.ndarray}
                -- testing labels.
            sess_file {str}
                -- where to checkpoint the model params.
        """
        pass

    def infer(self, imgs: np.ndarray, sess_file: str) -> np.ndarray:
        """produce inference on an array of images.

        Arguments:
            imgs {np.ndarray}
                -- images to be inferred.
            sess_file {str}
                -- where to restore model params checkpoint.

        Returns:
            np.ndarray
                -- an array of pmfs.
        """
        pass
