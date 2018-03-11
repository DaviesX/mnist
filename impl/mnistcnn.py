import math as m
import numpy as np
import tensorflow as tf
import cv2
from enc import to_one_hot
from ifmnist import if_mnist


def img(ws: int, hs: int):
    with tf.name_scope("input"):
        return tf.placeholder(tf.float32, shape=[None, ws, hs, 1], name="x")


def one_hot(num_classes: int):
    with tf.name_scope("one_hot"):
        return tf.placeholder(tf.int64, [None, num_classes], name="one_hot")


def label():
    with tf.name_scope("label"):
        return tf.placeholder(tf.int64, [None], name="label")


def keep_prob():
    with tf.name_scope("hyper"):
        return tf.placeholder(tf.float32, name="keep_prob")


def nn_layers(img_node, keep_prob_node,
              ws: int, hs: int,
              c1: int, c1s: int, c1d: int, p1s: int,
              c2: int, c2s: int, c2d: int, p2s: int,
              fc1s: int,
              num_classes: int):
    with tf.name_scope("c1"):
        w1 = tf.Variable(tf.truncated_normal([c1, c1, 1, c1d], stddev=0.1),
                         name="w")
        b1 = tf.Variable(tf.zeros([c1d]), name="b")
        conv1 = tf.nn.relu(tf.nn.conv2d(img_node, w1,
                                        strides=[c1s, c1s, c1s, c1s],
                                        padding="SAME",
                                        name="conv") + b1)

    with tf.name_scope("p1"):
        pool1 = tf.nn.max_pool(conv1,
                               ksize=[1, p1s, p1s, 1],
                               strides=[1, p1s, p1s, 1],
                               padding="SAME",
                               name="pool")

    with tf.name_scope("c2"):
        w2 = tf.Variable(tf.truncated_normal([c2, c2, c1d, c2d], stddev=0.1),
                         name="w")
        b2 = tf.Variable(tf.zeros([c2d]), name="b")
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w2,
                                        strides=[c2s, c2s, c2s, c2s],
                                        padding="SAME",
                                        name="conv") + b2)

    with tf.name_scope("p2"):
        pool2 = tf.nn.max_pool(conv2,
                               ksize=[1, p2s, p2s, 1],
                               strides=[1, p2s, p2s, 1],
                               padding="SAME",
                               name="pool")

    with tf.name_scope("fc"):
        fws = ws // p1s // p2s
        fhs = hs // p1s // p2s
        fc1is = fws*fhs*c2d
        w3 = tf.Variable(tf.truncated_normal([fc1is, fc1s], stddev=1.0/m.sqrt(float(fc1is))),
                         name="w")
        b3 = tf.Variable(tf.zeros([fc1s]), name="b")
        pool2_f = tf.reshape(pool2, [-1, fc1is])
        fc = tf.nn.relu(tf.matmul(pool2_f, w3) + b3, name="fc")

    with tf.name_scope("dropout"):
        fc_drop = tf.nn.dropout(fc, keep_prob_node)

    with tf.name_scope("output"):
        w4 = tf.Variable(tf.truncated_normal([fc1s, num_classes], stddev=1.0 / m.sqrt(float(fc1s))),
                         name="w")
        b4 = tf.Variable(tf.zeros([num_classes]), name="b")
        # output_node = tf.nn.softmax(tf.add(tf.matmul(fc_drop, w4),
        #                                   b4), name="output")
        return tf.add(tf.matmul(fc_drop, w4), b4, name="output")


def loss(output_node, one_hot_node):
    ce = tf.losses.softmax_cross_entropy(one_hot_node, output_node)
    return tf.reduce_mean(ce)


def opt(loss_node, learning_rate: float):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    opt_node = optimizer.minimize(loss_node, global_step=global_step)
    return opt_node


def evaluate(output_node, label_node):
    correct = tf.equal(tf.argmax(output_node, axis=1), label_node)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


class mnist_cnn(if_mnist):
    """2-layer CNN model.
    """

    def __init__(self, ws=28, hs=28, num_classes=10,
                 num_iters=20000, batch_size=20):
        if_mnist.__init__(self)
        self.ws = ws
        self.hs = hs
        self.num_classes = num_classes
        self.num_iters = num_iters
        self.batch_size = batch_size

    def name(self) -> str:
        """model name.
        """
        return "cnn"

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
        input_node = img(self.ws, self.hs)
        one_hot_node = one_hot(self.num_classes)
        label_node = label()
        keep_prob_node = keep_prob()
        output_node = nn_layers(input_node, keep_prob_node,
                                self.ws, self.hs,
                                5, 1, 32, 2,
                                5, 1, 64, 2,
                                1024,
                                self.num_classes)
        loss_node = loss(output_node, one_hot_node)
        opt_node = opt(loss_node, 0.001)
        eval_node = evaluate(output_node, label_node)

        # one-hot.
        tr_one_hots = to_one_hot(tr_labels, self.num_classes)

        # start parameter fitting.
        saver = tf.train.Saver()

        # fix shape
        tr_imgs = np.reshape(tr_imgs, [len(tr_imgs), self.ws, self.hs, 1])
        te_imgs = np.reshape(te_imgs, [len(te_imgs), self.ws, self.hs, 1])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.num_iters):
                batch_idx = np.random.randint(0, len(tr_imgs),
                                              size=self.batch_size)
                batch_img = tr_imgs[batch_idx]
                batch_one_hots = tr_one_hots[batch_idx]
                opt_node.run(feed_dict={input_node: batch_img,
                                        one_hot_node: batch_one_hots,
                                        keep_prob_node: 0.5})
                if i % 100 == 0:
                    # Evaluate current parameterization.
                    batch_idx2 = np.random.randint(0, len(te_imgs),
                                                   size=self.batch_size)
                    tr_accuracy = eval_node.eval(feed_dict={input_node: batch_img,
                                                            label_node: tr_labels[batch_idx],
                                                            keep_prob_node: 1.0})/self.batch_size
                    te_accuracy = eval_node.eval(feed_dict={input_node: te_imgs[batch_idx2],
                                                            label_node: te_labels[batch_idx2],
                                                            keep_prob_node: 1.0})/self.batch_size
                    l = loss_node.eval(feed_dict={input_node: batch_img,
                                                  one_hot_node: batch_one_hots,
                                                  keep_prob_node: 1.0})
                    print("iteration: " + str(i))
                    print("training accuracy: " + str(tr_accuracy))
                    print("test accuracy: " + str(te_accuracy))
                    print("loss: " + str(l))

            saver.save(sess, sess_file)

            final_te_accuracy = eval_node.eval(feed_dict={input_node: te_imgs,
                                                          label_node: te_labels,
                                                          keep_prob_node: 1.0}) / te_labels.shape[0]
            print("final test accuracy: " + str(final_te_accuracy))

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
            keep_prob_node = graph.get_tensor_by_name("hyper/keep_prob:0")
            output_node = graph.get_tensor_by_name("output/output:0")

            imgs = np.reshape(imgs, [len(imgs), self.ws, self.hs, 1])
            result = sess.run(output_node, feed_dict={input_node: imgs,
                                                      keep_prob_node: 1.0})
            return result
