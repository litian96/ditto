import os
import numpy as np
import tensorflow as tf
from tqdm import trange


from flearn.utils.tf_utils import graph_size, process_grad
from flearn.utils.model_utils import process_x, process_y

IMAGE_SIZE = 84
IMAGES_DIR = os.path.join('..', 'data', 'celeba', 'data', 'raw', 'img_align_celeba')


class Model(object):
    def __init__(self, num_classes, q, optimizer, seed=1):
        # params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + seed)
            self.features, self.labels, self.output2, self.train_op, self.grads, self.kl_grads, self.eval_metric_ops, \
                self.loss, self.kl_loss, self.soft_max, self.predictions = self.create_model(q, optimizer)
            self.saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

    def create_model(self, q, optimizer):
        input_ph = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
        output2 = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='output2')
        out = input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        logits = tf.layers.dense(out, self.num_classes)
        label_ph = tf.placeholder(tf.int64, shape=(None,))
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label_ph, logits=logits)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(label_ph, tf.argmax(input=logits, axis=1)))

        kl_loss = tf.keras.losses.KLD(predictions['probabilities'], output2) + tf.keras.losses.KLD(output2, predictions[
            'probabilities'])
        kl_grads_and_vars = optimizer.compute_gradients(kl_loss)
        kl_grads, _ = zip(*kl_grads_and_vars)

        return input_ph, label_ph, output2, train_op, grads, kl_grads, eval_metric_ops, loss, \
               kl_loss, predictions['probabilities'], predictions['classes']



    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data):

        with self.graph.as_default():
            grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data[0], self.labels: data[1]})

        return grads

    def get_kl_gradients(self, data, output2):
        with self.graph.as_default():
            kl_grads = self.sess.run(self.kl_grads,
                                    feed_dict={self.features: process_x(data['x']),
                                               self.labels: process_y(data['y']),
                                               self.output2: output2})

        return kl_grads

    def get_softmax(self, data):
        with self.graph.as_default():
            soft_max = self.sess.run(self.soft_max, feed_dict={self.features: process_x(data['x']),
                                                               self.labels: process_y(data['y'])})
        return soft_max

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                           feed_dict={self.features: mini_batch_data[0],
                                                      self.labels: mini_batch_data[1]})
        weights = self.get_params()
        return grads, loss, weights

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                                              feed_dict={self.features: process_x(data['x']),
                                                         self.labels: process_y(data['y'])})
        return tot_correct, loss

    def get_loss(self, data):
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={self.features: process_x(data['x']),
                                                       self.labels: process_y(data['y'])})
        return loss

    def close(self):
        self.sess.close()