from mnist.loader import MNIST as mnist_dataset
import sys
import os.path
import cv2
import math as m
import numpy as np
import tensorflow as tf


def img(xs):
    with tf.name_scope("input"):
        return tf.placeholder(tf.float32, shape=[None, xs], name="x")


def label(num_classes):
    with tf.name_scope("output"):
        return tf.placeholder(tf.int64, [None], name="output")


def nn_layers(img_node, xs, h1s, h2s, num_classes):
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


def opt(loss_node, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    opt_node = optimizer.minimize(loss_node, global_step=global_step)
    return opt_node


def evaluate(logits_node, label_node):
    correct = tf.nn.in_top_k(logits_node, label_node, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def fit(input_node, label_node, opt_node, eval_node, batch_size: int,
        tr_imgs, tr_labels, te_imgs, te_labels,
        sess_file):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch_idx = np.random.randint(0, len(tr_imgs), size=batch_size)
            batch_img = tr_imgs[batch_idx]
            batch_label = tr_labels[batch_idx]
            opt_node.run(feed_dict={input_node: batch_img,
                                    label_node: batch_label})
            if i % 100 == 0:
                batch_idx = np.random.randint(0, len(te_imgs), size=batch_size)
                tr_accuracy = eval_node.eval(feed_dict={input_node: batch_img,
                                                        label_node: batch_label})/batch_size
                te_accuracy = eval_node.eval(feed_dict={input_node: te_imgs[batch_idx],
                                                        label_node: te_labels[batch_idx]})/batch_size
                print("iteration: " + str(i))
                print("training accuracy: " + str(tr_accuracy))
                print("test accuracy: " + str(te_accuracy))
        saver.save(sess, sess_file)


def inference(imgs, sess_file):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(sess_file + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()

        input_node = graph.get_tensor_by_name("input/x:0")
        sm_node = graph.get_tensor_by_name("sm/sm:0")
        result = sess.run(sm_node, feed_dict={input_node: imgs})
        return result

    return None


def preprocess(img: np.ndarray):
    img = cv2.resize(img, (28, 28))
    return img.astype(dtype=np.uint8)


if __name__ == "__main__":
    sys.argv.append("IMG_0081.PNG")
    if not os.path.isfile("checkpoint"):
        print("start training process")
        dataset = mnist_dataset("./dataset")
        tr_imgs, tr_labels = dataset.load_training()
        te_imgs, te_labels = dataset.load_testing()
        tr_imgs = np.reshape(tr_imgs, [len(tr_imgs), 28*28])
        tr_labels = np.array(tr_labels)
        te_imgs = np.reshape(te_imgs, [len(te_imgs), 28*28])
        te_labels = np.array(te_labels)

        labels = [l for l in range(10)]
        input_node = img(28*28)
        label_node = label(len(labels))
        logits_node, _ = nn_layers(
            input_node, 28*28, 10*10, 5*5, len(labels))
        loss_node = loss(logits_node, label_node)
        opt_node = opt(loss_node, 0.001)
        eval_node = evaluate(logits_node, label_node)

        fit(input_node, label_node, opt_node, eval_node,
            1000,
            tr_imgs, tr_labels, te_imgs, te_labels,
            "./model.ckpt")
    else:
        print("start inference process")
        if len(sys.argv) > 1:
            # process the image.
            test_file = sys.argv[1]
            test_img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)

            x = np.reshape(preprocess(test_img), [1, 28*28])
            pmf = inference(x, "./model.ckpt")
            best_prob = np.max(pmf)
            best_label = np.where(pmf[0, :] == best_prob)[0]
            print("best label: " + str(best_label))
            print("best prob: " + str(best_prob))
        else:
            # use the webcam.
            cap = cv2.VideoCapture(0)

            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Display the resulting frame
                cv2.imshow('frame', gray)
                x = np.reshape(preprocess(test_img), [1, 28*28])
                pmf = inference(x, "./model.ckpt")
                best_prob = np.max(pmf)
                best_label = np.where(pmf == best_prob)
                print("best label: " + str(best_label))
                print("best prob: " + str(best_prob))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
