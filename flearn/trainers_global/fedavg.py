import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, l2_clip, get_stdev
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, gen_batch_celeba
from flearn.utils.language_utils import letter_to_vec, word_to_indices


def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch



class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using fed avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        np.random.seed(1234567+self.seed)
        corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted, replace=False)
        print(corrupt_id)

        if self.dataset == 'shakespeare':
            for c in self.clients:
                c.train_data['y'], c.train_data['x'] = process_y(c.train_data['y']), process_x(c.train_data['x'])
                c.test_data['y'], c.test_data['x'] = process_y(c.test_data['y']), process_x(c.test_data['x'])

        batches = {}

        for idx, c in enumerate(self.clients):
            if idx in corrupt_id:
                c.train_data['y'] = np.asarray(c.train_data['y'])
                if self.dataset == 'celeba':
                    c.train_data['y'] = 1 - c.train_data['y']
                elif self.dataset == 'femnist':
                    c.train_data['y'] = np.random.randint(0, 62, len(c.train_data['y']))
                elif self.dataset == 'shakespeare':
                    c.train_data['y'] = np.random.randint(0, 80, len(c.train_data['y']))
                elif self.dataset == "vehicle":
                    c.train_data['y'] = c.train_data['y'] * -1
                elif self.dataset == 'fmnist': 
                    c.train_data['y'] = np.random.randint(0, 10, len(c.train_data['y']))

            if self.dataset == 'celeba':  # need to deal with celeba data loading a bit differently
                batches[c] = gen_batch_celeba(c.train_data, self.batch_size, self.num_rounds * self.local_iters)
            else:
                batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds * self.local_iters)

        for i in range(self.num_rounds+1):
            if i % self.eval_every == 0:
                num_test, num_correct_test, test_loss_vector = self.test()  # have set the latest model for all clients
                avg_test_loss = np.dot(test_loss_vector, num_test) / np.sum(num_test)
                num_train, num_correct_train, train_loss_vector = self.train_error()
                avg_train_loss = np.dot(train_loss_vector, num_train) / np.sum(num_train)
    
                tqdm.write('At round {} training accu: {}, loss: {}'.format(i, np.sum(num_correct_train) * 1.0 / np.sum(
                    num_train), avg_train_loss))
                tqdm.write('At round {} test loss: {}'.format(i, avg_test_loss))
                tqdm.write('At round {} test accu: {}'.format(i, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
                non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
                tqdm.write('At round {} malicious test accu: {}'.format(i, np.sum(
                    num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
                tqdm.write('At round {} benign test accu: {}'.format(i, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
                tqdm.write('At round {} variance: {}'.format(i, np.var(
                    num_correct_test[non_corrupt_id] * 1.0 / num_test[non_corrupt_id])))
            
            indices, selected_clients = self.select_clients(round=i, corrupt_id=corrupt_id, num_clients=self.clients_per_round)

            csolns = []
            losses = []

            for idx in indices:
                c = self.clients[idx]

                # communicate the latest model
                c.set_params(self.latest_model)
                weights_before = copy.deepcopy(self.latest_model)

                loss = c.get_loss()  # training loss
                losses.append(loss)

                for _ in range(self.local_iters):
                    data_batch = next(batches[c])
                    _, grads, _ = c.solve_sgd(data_batch)

                w_global = c.get_params()

                grads = [(u - v) * 1.0 for u, v in zip(w_global, weights_before)]

                if idx in corrupt_id:
                    if self.boosting:  # model replacement
                        grads = [self.clients_per_round * u for u in grads]
                    elif self.random_updates:
                        # send random updates
                        stdev_ = get_stdev(grads)
                        grads = [np.random.normal(0, stdev_, size=u.shape) for u in grads]

                if self.q == 0:
                    csolns.append(grads)
                else:
                    csolns.append((np.exp(self.q * loss), grads))

            if self.q != 0:
                overall_updates = self.aggregate(csolns)
            else:
                if self.gradient_clipping:
                    csolns = l2_clip(csolns)

                expected_num_mali = int(self.clients_per_round * self.num_corrupted / len(self.clients))

                if self.median:
                    overall_updates = self.median_average(csolns)
                elif self.k_norm:
                    overall_updates = self.k_norm_average(self.clients_per_round - expected_num_mali, csolns)
                elif self.k_loss:
                    overall_updates = self.k_loss_average(self.clients_per_round - expected_num_mali, losses, csolns)
                elif self.krum:
                    overall_updates = self.krum_average(self.clients_per_round - expected_num_mali - 2, csolns)
                elif self.mkrum:
                    m = self.clients_per_round - expected_num_mali
                    overall_updates = self.mkrum_average(self.clients_per_round - expected_num_mali - 2, m, csolns)
                elif self.fedmgda:
                    overall_updates = self.fedmgda_average(csolns)
                else:
                    overall_updates = self.simple_average(csolns)

            self.latest_model = [(u + v) for u, v in zip(self.latest_model, overall_updates)]



                    



