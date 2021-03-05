import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import copy

from flearn.trainers_global.fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad, cosine_sim, softmax, norm_grad, l2_clip, get_stdev
from flearn.utils.model_utils import batch_data, gen_batch, gen_epoch, gen_batch_celeba


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using symmetrized KL')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        print('---{} workers per communication round---'.format(self.clients_per_round))

        np.random.seed(1234567+self.seed)
        corrupt_id = np.random.choice(range(len(self.clients)), size=self.num_corrupted, replace=False)

        batches = {}
        for idx, c in enumerate(self.clients):
            if idx in corrupt_id:
                c.train_data['y'] = np.asarray(c.train_data['y'])
                if self.dataset == 'celeba':
                    c.train_data['y'] = 1 - c.train_data['y']
                elif self.dataset == 'femnist':
                    c.train_data['y'] = np.random.randint(0, 62, len(c.train_data['y']))

            if self.dataset == 'celeba':
                batches[c] = gen_batch_celeba(c.train_data, self.batch_size, self.num_rounds * self.local_iters + 300)
            else:
                batches[c] = gen_batch(c.train_data, self.batch_size, self.num_rounds * self.local_iters + 300)


        for i in range(self.num_rounds + 1):
            if i % self.eval_every == 0 and i > 0:

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

            # weighted sampling
            indices, selected_clients = self.select_clients(round=i, corrupt_id=corrupt_id, num_clients=self.clients_per_round)

            csolns = []
            for idx in indices:
                w_global_idx = copy.deepcopy(self.latest_model)
                c = self.clients[idx]
                c.set_params(w_global_idx)
                for _ in range(self.local_iters):
                    data_batch = next(batches[c])
                    _, grads, _ = c.solve_sgd(data_batch)
                w_global_idx = self.client_model.get_params()

                # get the difference (global model updates)
                diff = [u - v for (u, v) in zip(w_global_idx, self.latest_model)]

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

            avg_updates = self.simple_average(csolns)

            # update the global model
            for layer in range(len(avg_updates)):
                self.latest_model[layer] += avg_updates[layer]


        # local finetuning based on KL
        after_test_accu = []
        test_samples = []
        for idx, c in enumerate(self.clients):

            c.set_params(self.latest_model)
            output2 = copy.deepcopy(c.get_softmax()) 
            # start to finetune
            local_model = copy.deepcopy(self.latest_model)

            for _ in range(max(int(self.finetune_iters * c.train_samples / self.batch_size), self.finetune_iters)):
                data_batch = next(batches[c])
                c.set_params(local_model)
                kl_grads = c.get_kl_grads(output2)
                _, grads, _ = c.solve_sgd(data_batch)

                for j in range(len(grads[1])):
                   eff_grad = grads[1][j] + self.lam * kl_grads[j]
                   local_model[j] = local_model[j] - self.learning_rate * eff_grad

            c.set_params(local_model)
            tc, _, num_test = c.test()
            after_test_accu.append(tc)
            test_samples.append(num_test)


        non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
        after_test_accu = np.asarray(after_test_accu)
        test_samples = np.asarray(test_samples)
        tqdm.write('final test accu: {}'.format(np.sum(after_test_accu) * 1.0 / np.sum(test_samples)))
        tqdm.write('final malicious test accu: {}'.format(np.sum(
            after_test_accu[corrupt_id]) * 1.0 / np.sum(test_samples[corrupt_id])))
        tqdm.write('final benign test accu: {}'.format(np.sum(
            after_test_accu[non_corrupt_id]) * 1.0 / np.sum(test_samples[non_corrupt_id])))
        print("variance of the performance: ",
              np.var(after_test_accu[non_corrupt_id] / test_samples[non_corrupt_id]))



