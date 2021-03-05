import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, l2_clip, get_stdev
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, gen_batch_celeba


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using mean-regularized multi-task learning to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('---{} workers per communication round---'.format(self.clients_per_round))

        np.random.seed(1234567)
        corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted)
        print(corrupt_id)

        batches = {}
        for idx, c in enumerate(self.clients):
            if idx in corrupt_id:
                c.train_data['y'] = np.asarray(c.train_data['y'])
                if self.dataset == 'celeba':
                    c.train_data['y'] = 1 - c.train_data['y']
                elif self.dataset == 'femnist':
                    c.train_data['y'] = np.random.randint(0, 62, len(c.train_data['y']))  # [0, 62)
                elif self.dataset == 'shakespeare':
                    c.train_data['y'] = np.random.randint(0, 80, len(c.train_data['y']))
                elif self.dataset == "vehicle":
                    c.train_data['y'] = c.train_data['y'] * -1
                elif self.dataset == "fmnist":
                    c.train_data['y'] = np.random.randint(0, 10, len(c.train_data['y']))

            if self.dataset == 'celeba':
                # due to a different data storage format
                batches[c] = gen_batch_celeba(c.train_data, self.batch_size, self.num_rounds * self.local_iters)
            else:
                batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds * self.local_iters)

        initialization = copy.deepcopy(self.clients[0].get_params())

        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0 and i > 0:
                tmp_models = []
                for idx in range(len(self.clients)):
                    a = []
                    for layer in range(len(self.local_models[idx])):
                          a.append(self.alpha * self.local_models[idx][layer] + (1-self.alpha) * self.global_model[layer])
                    tmp_models.append(a)
                num_test, num_correct_test = self.test(tmp_models)
                num_train, num_correct_train, loss_vector = self.train_error(tmp_models)
                avg_loss = np.dot(loss_vector, num_train) / np.sum(num_train)
                print(num_correct_test / num_test)

                tqdm.write('At round {} training accu: {}, loss: {}'.format(i, np.sum(num_correct_train) * 1.0 / np.sum(
                    num_train), avg_loss))
                tqdm.write('At round {} test accu: {}'.format(i, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
                non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
                tqdm.write('At round {} malicious test accu: {}'.format(i, np.sum(num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
                tqdm.write('At round {} benign test accu: {}'.format(i, np.sum(num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
                print("variance of the performance: ", np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))

            # weighted sampling
            indices, selected_clients = self.select_clients(round=i, num_clients=self.clients_per_round)

            csolns = []

            for idx in indices:
                c = self.clients[idx]

                # server sends the current global model to selected devices
                w_global_idx = copy.deepcopy(self.global_model)

                self.client_model.set_params(self.global_model)

                for _ in range(self.local_iters):
                    # first sample a mini-batch
                    data_batch = next(batches[c])
                    # optimize the global model

                    self.client_model.set_params(w_global_idx)
                    _, grads, _ = c.solve_sgd(data_batch)  # grads: (num_samples, real_grads)
                    w_global_idx = self.client_model.get_params()

                    # optimize for the local model (wrt to the interpolation)
                    self.client_model.set_params(self.interpolation[idx])
                    _, grads, _ = c.solve_sgd(data_batch)
                    for layer in range(len(self.local_models[idx])):
                        self.local_models[idx][layer] = self.local_models[idx][layer] - self.alpha * self.learning_rate * grads[1][layer]
#
                    # update the interpolation
                    for layer in range(len(self.local_models[idx])):
                        self.interpolation[idx][layer] = self.alpha * self.local_models[idx][layer] + (1-self.alpha) * w_global_idx[layer]

                diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]

                # send the malicious updates
                if idx in corrupt_id:
                    if self.boosting:
                        # scale malicious updates
                        diff = [self.clients_per_round * u for u in diff]
                    elif self.random_updates:
                        # send random updates
                        stdev_ = get_stdev(diff)
                        diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]

                csolns.append(diff)

            if self.gradient_clipping:
                csolns = l2_clip(csolns)
            if self.median:
                avg_updates = self.median_average(csolns)
            else:
                avg_updates = self.simple_average(csolns)
            for layer in range(len(avg_updates)):
                self.global_model[layer] += avg_updates[layer]
