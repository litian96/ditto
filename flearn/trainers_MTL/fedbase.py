import numpy as np
import tensorflow as tf
from tqdm import tqdm
import copy

from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad, norm_grad, norm_grad_sparse


class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.q, self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.dynamic_lam, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = copy.deepcopy(self.client_model.get_params())  # self.latest_model is the global model
        self.local_models = []
        self.interpolation = []
        self.global_model = copy.deepcopy(self.latest_model)
        for _ in self.clients:
            self.local_models.append(copy.deepcopy(self.latest_model))
            self.interpolation.append(copy.deepcopy(self.latest_model))

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, dynamic_lam, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], dynamic_lam, model) for u, g in zip(users, groups)]
        return all_clients

    def train_error(self, models):
        num_samples = []
        tot_correct = []
        losses = []

        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            ct, cl, ns = c.train_error()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        return np.array(num_samples), np.array(tot_correct), np.array(losses)

    def test(self, models):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []

        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            ct, cl, ns = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        return np.array(num_samples), np.array(tot_correct), np.array(losses)

    def validate(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []

        for idx, c in enumerate(self.clients):
            self.client_model.set_params(self.local_models[idx])
            ct, ns = c.validate()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
        return np.array(num_samples), np.array(tot_correct)


    def save(self):
        pass

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients

        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            indices: an array of indices
            self.clients[]
        '''
        num_clients = min(num_clients, len(self.clients))

        np.random.seed(round + 4)

        if self.sampling == 1:
            pk = np.ones(num_clients) / num_clients
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]

        elif self.sampling == 2:
            num_samples = []
            for client in self.clients:
                num_samples.append(client.train_samples)
            total_samples = np.sum(np.asarray(num_samples))
            pk = [item * 1.0 / total_samples for item in num_samples]
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]


    def aggregate(self, wsolns):

        total_weight = 0.0
        base = [0] * len(wsolns[0][1])

        for (w, soln) in wsolns:
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w * v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    def simple_average(self, parameters):

        base = [0] * len(parameters[0])

        for p in parameters:  # for each client
            for i, v in enumerate(p):
                base[i] += v.astype(np.float64)   # the i-th layer

        averaged_params = [v / len(parameters) for v in base]

        return averaged_params

    def median_average(self, parameters):

        num_layers = len(parameters[0])
        aggregated_models = []
        for i in range(num_layers):
            a = []
            for j in range(len(parameters)):
                a.append(parameters[j][i].flatten())
            aggregated_models.append(np.reshape(np.median(a, axis=0), newshape=parameters[0][i].shape))

        return aggregated_models

    def krum_average(self, k, parameters):
        # krum: return the parameter which has the lowest score defined as the sum of distance to its closest k vectors
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        distance = np.zeros((len(parameters), len(parameters)))
        for i in range(len(parameters)):
            for j in range(i+1, len(parameters)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]

        score = np.zeros(len(parameters))
        for i in range(len(parameters)):
            score[i] = np.sum(np.sort(distance[i])[:k+1])  # the distance including itself, so k+1 not k

        selected_idx = np.argsort(score)[0]

        return parameters[selected_idx]

    def mkrum_average(self, k, m, parameters):
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        distance = np.zeros((len(parameters), len(parameters)))
        for i in range(len(parameters)):
            for j in range(i + 1, len(parameters)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]

        score = np.zeros(len(parameters))
        for i in range(len(parameters)):
            score[i] = np.sum(np.sort(distance[i])[:k + 1])  # the distance including itself, so k+1 not k

        # multi-krum selects top-m 'good' vectors (defined by socre) (m=1: reduce to krum)
        selected_idx = np.argsort(score)[:m]
        selected_parameters = []
        for i in selected_idx:
            selected_parameters.append(parameters[i])

        return self.simple_average(selected_parameters)

