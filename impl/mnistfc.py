import cv2
import math as m
import numpy as np
import tensorflow as tf
from ifmnist import if_mnist


def img(xs: int):
    with tf.name_scope("input"):
        return tf.placeholder(tf.float32, shape=[None, xs], name="x")


def label(num_classes: int):
    with tf.name_scope("output"):
        return tf.placeholder(tf.int64, [None], name="output")


def nn_layers(img_node, xs: int, h1s: int, h2s: int, num_classes: int):
    with tf.name_scope("h1"):
        weights = tf.Variable(tf.truncated_normal([xs, h1s],
                                                  stddev=1.0 / m.sqrt(float(xs))),
                              name="w")
        biases = tf.Variable(tf.zeros([h1s]), name="b")
        hidden1 = tf.nn.relu(tf.matmul(img_node, weights) + biases)

    with tf.name_scope("h2"):
        weights = tf.Variable(tf.truncated_normal([h1s, h2s], stddev=1.0 / m.sqrt(float(h1s))),
                              name="w")
        biases = tf.Variable(tf.zeros([h2s]), name="b")
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope("sm"):
        weights = tf.Variable(tf.truncated_normal([h2s, num_classes], stddev=1.0 / m.sqrt(float(h2s))),
                              name="w")
        biases = tf.Variable(tf.zeros([num_classes]), name="b")
        logits_node = tf.add(tf.matmul(hidden2, weights),
                             biases, name="logits")
        sm_node = tf.nn.softmax(logits_node, name="sm")
        return logits_node, sm_node


def loss(logits_node, label_node):
    return tf.losses.sparse_softmax_cross_entropy(labels=tf.to_int64(label_node), logits=logits_node)


def opt(loss_node, learning_rate: float):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    opt_node = optimizer.minimize(loss_node, global_step=global_step)
    return opt_node


def evaluate(logits_node, label_node):
    correct = tf.nn.in_top_k(logits_node, label_node, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


class mnistfc(if_mnist):
    """[summary]

    Arguments:
        if_mnist {[type]} -- [description]
    """

    def __init__(self, ws=28, hs=28, num_classes=10,
                 num_iters=20000, batch_size=1000):
        if_mnist.__init__(self)
        self.ws = ws
        self.hs = hs
        self.num_classes = num_classes
        self.num_iters = num_iters
        self.batch_size = batch_size

    def name(self) -> str:
        return "fc2"

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
        # construct the architecture.
        input_node = img(int(self.ws*self.hs))
        label_node = label(self.num_classes)
        logits_node, _ = nn_layers(input_node,
                                   int(self.ws * self.hs),
                                   int((self.ws / 2 - 4) * (self.ws / 2 - 4)),
                                   int(2*self.num_classes + 5),
                                   self.num_classes)
        loss_node = loss(logits_node, label_node)
        opt_node = opt(loss_node, 0.001)
        eval_node = evaluate(logits_node, label_node)

        # start parameter fitting.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.num_iters):
                batch_idx = np.random.randint(0, len(tr_imgs),
                                              size=self.batch_size)
                batch_img = tr_imgs[batch_idx]
                batch_label = tr_labels[batch_idx]
                opt_node.run(feed_dict={input_node: batch_img,
                                        label_node: batch_label})
                if i % 100 == 0:
                    # Evaluate current parameterization.
                    batch_idx = np.random.randint(0, len(te_imgs),
                                                  size=self.batch_size)
                    tr_accuracy = eval_node.eval(feed_dict={input_node: batch_img,
                                                            label_node: batch_label})/self.batch_size
                    te_accuracy = eval_node.eval(feed_dict={input_node: te_imgs[batch_idx],
                                                            label_node: te_labels[batch_idx]})/self.batch_size
                    print("iteration: " + str(i))
                    print("training accuracy: " + str(tr_accuracy))
                    print("test accuracy: " + str(te_accuracy))

            saver.save(sess, sess_file)

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
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(sess_file + ".meta")
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            graph = tf.get_default_graph()

            input_node = graph.get_tensor_by_name("input/x:0")
            sm_node = graph.get_tensor_by_name("sm/sm:0")
            result = sess.run(sm_node, feed_dict={input_node: imgs})
            return result
