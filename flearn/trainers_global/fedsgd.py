import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using fair fed avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        batches = {}
        for c in self.clients:
            batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds + 2, 0)

        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0:
                num_test, num_correct_test = self.test()  # have set the latest model for all clients
                num_train, num_correct_train = self.train_error()
                tqdm.write('At round {} testing accuracy: {}'.format(i,
                                                                     np.sum(np.array(num_correct_test)) * 1.0 / np.sum(
                                                                         np.array(num_test))))
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(
                    np.array(num_correct_train)) * 1.0 / np.sum(np.array(num_train))))

            # weighted sampling
            indices, selected_clients = self.select_clients(round=i, num_clients=self.clients_per_round)

            cgrads = []
            for idx in indices:
                c = self.clients[idx]
                c.set_params(self.latest_model)
                _, grads, _ = c.solve_sgd(next(batches[c]))
                cgrads.append(grads[1])

            avg_grad = self.simple_average(cgrads)

            for layer in range(len(avg_grad)):
                self.latest_model[layer] = self.latest_model[layer] - self.learning_rate * avg_grad[layer]





