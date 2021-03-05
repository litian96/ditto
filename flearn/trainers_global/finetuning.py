import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, l2_clip, get_stdev
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, gen_batch_celeba


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using fair fed avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))

        np.random.seed(1234567+self.seed)
        corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted, replace=False)
        print(corrupt_id)

        batches = {}
        for idx, c in enumerate(self.clients):
            if idx in corrupt_id:
                c.train_data['y'] = np.asarray(c.train_data['y'])
                if self.dataset == 'celeba':
                    c.train_data['y'] = 1 - c.train_data['y']
                elif self.dataset == 'femnist':
                    c.train_data['y'] = np.random.randint(0, 62, len(c.train_data['y']))  # [0, 62)
                elif self.dataset == 'fmnist': # fashion mnist
                    c.train_data['y'] = np.random.randint(0, 10, len(c.train_data['y']))


            if self.dataset == 'celeba':
                batches[c] = gen_batch_celeba(c.train_data, self.batch_size, self.num_rounds * self.local_iters + 350)
            else:
                batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds * self.local_iters + 350)

        initialization = copy.deepcopy(self.clients[0].get_params())

        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0:
                num_test, num_correct_test, _ = self.test()  # have set the latest model for all clients
                num_train, num_correct_train, loss_vector = self.train_error()

                avg_loss = np.dot(loss_vector, num_train) / np.sum(num_train)

                tqdm.write('At round {} training accu: {}, loss: {}'.format(i, np.sum(num_correct_train) * 1.0 / np.sum(
                    num_train), avg_loss))
                tqdm.write('At round {} test accu: {}'.format(i, np.sum(num_correct_test) * 1.0 / np.sum(num_test)))
                non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
                tqdm.write('At round {} malicious test accu: {}'.format(i, np.sum(
                    num_correct_test[corrupt_id]) * 1.0 / np.sum(num_test[corrupt_id])))
                tqdm.write('At round {} benign test accu: {}'.format(i, np.sum(
                    num_correct_test[non_corrupt_id]) * 1.0 / np.sum(num_test[non_corrupt_id])))
                print("variance of the performance: ",
                      np.var(num_correct_test[non_corrupt_id] / num_test[non_corrupt_id]))

            indices, selected_clients = self.select_clients(round=i, corrupt_id=corrupt_id,
                                                            num_clients=self.clients_per_round)

            csolns = []
            losses = []

            for idx in indices:
                c = self.clients[idx]

                # communicate the latest model
                c.set_params(self.latest_model)
                weights_before = copy.deepcopy(self.latest_model)
                loss = c.get_loss()  # compute loss on the whole training data
                losses.append(loss)

                for _ in range(self.local_iters):
                    data_batch = next(batches[c])
                    _, _, _ = c.solve_sgd(data_batch)

                new_weights = c.get_params()

                grads = [(u - v) * 1.0 for u, v in zip(new_weights, weights_before)]


                if idx in corrupt_id:
                    if self.boosting:  # model replacement
                        grads = [self.clients_per_round * u for u in grads]
                    elif self.random_updates:
                        # send random updates
                        stdev_ = get_stdev(grads)
                        grads = [np.random.normal(0, stdev_, size=u.shape) for u in grads]

                if self.q > 0:
                    csolns.append((np.exp(self.q * loss), grads))
                else:
                    csolns.append(grads)

            if self.q > 0:
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
                else:
                    overall_updates = self.simple_average(csolns)

            self.latest_model = [(u + v) for u, v in zip(self.latest_model, overall_updates)]

            distance = np.linalg.norm(process_grad(self.latest_model) - process_grad(initialization))
            if i % self.eval_every == 0:
                print('distance to initialization:', distance)


        # local finetuning
        init_model = copy.deepcopy(self.latest_model)

        after_test_accu = []
        test_samples = []
        for idx, c in enumerate(self.clients):
            c.set_params(init_model)
            local_model = copy.deepcopy(init_model)
            for _ in range(max(int(self.finetune_iters * c.train_samples / self.batch_size), self.finetune_iters)):
                c.set_params(local_model)
                data_batch = next(batches[c])
                _, grads, _ = c.solve_sgd(data_batch)
                for j in range(len(grads[1])):
                    eff_grad = grads[1][j] + self.lam * (local_model[j] - init_model[j])
                    local_model[j] = local_model[j] - self.learning_rate * self.decay_factor * eff_grad
            c.set_params(local_model)
            tc, _, num_test = c.test()
            after_test_accu.append(tc)
            test_samples.append(num_test)

        after_test_accu = np.asarray(after_test_accu)
        test_samples = np.asarray(test_samples)
        tqdm.write('final test accu: {}'.format(np.sum(after_test_accu) * 1.0 / np.sum(test_samples)))
        tqdm.write('final malicious test accu: {}'.format(np.sum(
            after_test_accu[corrupt_id]) * 1.0 / np.sum(test_samples[corrupt_id])))
        tqdm.write('final benign test accu: {}'.format(np.sum(
            after_test_accu[non_corrupt_id]) * 1.0 / np.sum(test_samples[non_corrupt_id])))
        print("variance of the performance: ",
              np.var(after_test_accu[non_corrupt_id] / test_samples[non_corrupt_id]))








