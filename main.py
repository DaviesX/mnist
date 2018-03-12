import sys
import os.path
import numpy as np
import cv2
from mnist.loader import MNIST as mnist_dataset
from improc import preprocess
from ifmnist import if_mnist
from impl.mnistfc import mnistfc
from impl.mnistcnn import mnist_cnn


def run(agent: if_mnist) -> None:
    if not os.path.isfile("./" + agent.name() + ".ckpt.meta"):
        print("start training process")
        dataset = mnist_dataset("./dataset")
        tr_imgs, tr_labels = dataset.load_training()
        te_imgs, te_labels = dataset.load_testing()
        tr_imgs = np.array(tr_imgs)
        tr_labels = np.array(tr_labels)
        te_imgs = np.array(te_imgs)
        te_labels = np.array(te_labels)

        agent.fit(tr_imgs, tr_labels, te_imgs, te_labels,
                  "./" + agent.name() + ".ckpt")
    else:
        print("start inference process")
        if len(sys.argv) > 1:
            # process the image.
            test_file = sys.argv[1]
            test_img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)

            pmf = agent.infer(np.reshape(preprocess(test_img), [1, 28, 28]),
                              "./" + agent.name() + ".ckpt")
            best_prob = np.max(pmf)
            best_label = np.where(pmf[0, :] == best_prob)[0]
            print("best label: " + str(best_label))
            print("best prob: " + str(best_prob))
        else:
            # use the webcam.
            cap = cv2.VideoCapture(0)

            while True:
                # Capture frame-by-frame
                ok, frame = cap.read()
                if not ok:
                    raise Exception("Failed to capture web cam.")

                # Our operations on the frame come here
                test_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Display the resulting frame
                cv2.imshow('frame', test_img)
                pmf = agent.infer(np.reshape(preprocess(test_img), [1, 28, 28]),
                                  "./" + agent.name() + ".ckpt")
                best_prob = np.max(pmf)
                best_label = np.where(pmf[0, :] == best_prob)[0]
                print("best label: " + str(best_label))
                print("best prob: " + str(best_prob))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run(mnist_cnn())
