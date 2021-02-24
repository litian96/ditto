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
        for key, val in params.items(): setattr(self, key, val);

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.q, self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.dynamic_lam, self.client_model)
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = copy.deepcopy(self.client_model.get_params())

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, dynamic=0, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], dynamic, model) for u, g in zip(users, groups)]
        return all_clients

    def train_error(self):
        num_samples = []
        tot_correct = []
        losses = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, cl, ns = c.train_error()
            tot_correct.append(ct*1.0)
            losses.append(cl * 1.0)
            num_samples.append(ns)

        return np.array(num_samples), np.array(tot_correct), np.array(losses)


    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, cl, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
        return np.array(num_samples), np.array(tot_correct), np.array(losses)


    def validate(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.validate()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        return np.array(num_samples), np.array(tot_correct)

    def test_resulting_model(self):
        num_samples = []
        tot_correct = []
        #  self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass

    def select_clients(self, round, corrupt_id, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            indices: an array of indices
            self.clients[]
        '''
        num_clients = min(num_clients, len(self.clients)) # number of selected clients per round
        np.random.seed(round+4)
        non_corrupt_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
        corrupt_fraction = len(corrupt_id) / len(self.clients)
        num_selected_corrupted = int(num_clients * corrupt_fraction)


        if self.sampling == 0:
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]
               
        elif self.sampling == 1:
            num_samples = []
            for client in self.clients:
                num_samples.append(client.train_samples)
            total_samples = np.sum(np.asarray(num_samples))
            pk = [item * 1.0 / total_samples for item in num_samples]
            indices1 = np.random.choice(corrupt_id, num_selected_corrupted, replace=False, p=np.asarray(pk)[corrupt_id] / sum(np.asarray(pk)[corrupt_id]))
            indices2 = np.random.choice(non_corrupt_id, num_clients-num_selected_corrupted, replace=False, p=np.asarray(pk)[non_corrupt_id] / sum(np.asarray(pk)[non_corrupt_id]))
            indices = np.concatenate((indices1, indices2))
            #print(indices1, indices2)
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
        base = [0]*len(wsolns[0][1])

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
            score[i] = np.sum(np.sort(distance[i])[:k+1]) 

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
            score[i] = np.sum(np.sort(distance[i])[:k + 1])  

        # multi-krum selects top-m 'good' vectors (defined by socre) (m=1: reduce to krum)
        selected_idx = np.argsort(score)[:m]
        selected_parameters = []
        for i in selected_idx:
            selected_parameters.append(parameters[i])

        return self.simple_average(selected_parameters)

    def k_norm_average(self, num_benign, parameters):
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        norms = [np.linalg.norm(u) for u in flattened_grads]
        selected_idx = np.argsort(norms)[:num_benign]  # filter out the updates with large norms
        selected_parameters = []
        for i in selected_idx:
            selected_parameters.append(parameters[i])

        return self.simple_average(selected_parameters)


    def k_loss_average(self, num_benign, losses, parameters):
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        selected_idx = np.argsort(losses)[num_benign-1]  # select the update with largest loss among the benign devices
        return parameters[selected_idx]


    def fedmgda_average(self, parameters):

        from cvxopt import solvers, matrix

        flattened_grads = []
        for i in range(len(parameters)):
            tmp = process_grad(parameters[i])  # 1-D array of the model updates
            scale = np.linalg.norm(tmp)
            tmp = tmp / scale  # server first normalizes the gradients
            flattened_grads.append(tmp) # flattened_grads used for calculating the lambda for aggregation
            for layer in range(len(parameters[i])):
                parameters[i][layer] = parameters[i][layer] / scale 

        # then solve a SDP to obtain \lambda^*
        n = len(parameters)
        all_g = np.asarray(flattened_grads)
        P = matrix(np.matmul(all_g, all_g.T).astype(np.double))
        q = matrix([0.0] * n)
        I = matrix(0.0, (n, n))
        I[::n+1] = 1.0 # I is identity
        G = matrix([-I, -I, I])
        h = matrix([0.0] * n + [self.fedmgda_eps-1.0/n] * n + [self.fedmgda_eps+1.0/n] * n)
        A = matrix(1.0, (1, n))
        b = matrix(1.0)
        sol_lambda = solvers.qp(P, q, G, h, A, b)['x']

        # print(sol_lambda)

        # return aggregated gradients using sol_lambda
        return self.aggregate([(sol_lambda[i], parameters[i]) for i in range(n)])


    def aggregate2(self, weights_before, Deltas, hs): 
        
        demominator = np.sum(np.asarray(hs))
        num_clients = len(Deltas)
        scaled_deltas = []
        for client_delta in Deltas:
            scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

        updates = []
        for i in range(len(Deltas[0])):
            tmp = scaled_deltas[0][i]
            for j in range(1, len(Deltas)):
                tmp += scaled_deltas[j][i]
            updates.append(tmp)

        new_solutions = [(u - v) * 1.0 for u, v in zip(weights_before, updates)]

        return new_solutions

