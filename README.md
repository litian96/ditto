<img width="200" height="150" src="https://user-images.githubusercontent.com/14993256/109053987-54418f80-76ab-11eb-98bd-2c119d8a61ce.gif">

# Ditto: Fair and Robust Federated Learning Through Personalization

This repository contains the code and experiments for the manuscript:

> [Ditto: Fair and Robust Federated Learning Through Personalization](https://arxiv.org/abs/2012.04221)
>

Fairness and robustness are two important concerns for federated learning systems.
In this work, we identify that *robustness* to data and model poisoning attacks and *fairness*, measured as the uniformity of performance across devices, are competing constraints in statistically heterogeneous networks. 
To address these constraints, we propose employing a simple, general framework for personalized federated learning, Ditto, and develop a scalable solver for it. 
Theoretically, we  analyze the ability of Ditto to achieve
fairness and robustness simultaneously on a class of linear problems.
Empirically, across a suite of federated datasets, we show that Ditto not only achieves competitive performance relative to recent personalization methods, but also enables more accurate, robust, and fair models relative to state-of-the-art fair or robust baselines.



### *We also provide [Pytorch implementation](https://github.com/s-huu/Ditto)*



## Preparation

### Dataset generation

For each dataset, we provide links to downloadable datasets used in our experiments. We describe in our paper and the `REAME` files in separate `ditto/data/$dataset` folders on how these datasets are generated, and provide instructions and scripts on preprocessing and/or sampling data.


### Downloading dependencies

```
pip3 install -r requirements.txt
``` 

## Run the point estimation example

We provide a jupyter notebook that simulates the federated point estimation problem. To run that, make sure you are under the `ditto` folder, and 

```
jupyter notebook
```
then open `point_estimation.ipynb`, and directly run the notebook cell by cell to reproduce the results.

## Run on federated benchmarks

(A subset of) Options in `run.sh`:

* `dataset` chosen from `[femnist, fmnist, celeba, vehicle]`, where fmnist is short for Fashion MNIST.
*  `model` should be the corresponding model of that dataset. You can find it the model name under `flearn/models/$dataset/$model.py`, and take `$model`.
* `$optimizer` chosen from `['ditto', 'apfl', 'ewc', 'kl', 'l2sgd', 'mapper', 'meta', 'fedavg', 'finetuning']`
* `fedavg` is training global models, `ditto` with `lam=0` corresponds to training separate local models
* `$lambda` is the lambda we use for ditto (can use dynamic lambdas by setting `--dynamic_lam` to 1)
* `num_corrupted` is the number of corrupted devices (see the total number of devices in paper)
* `random_updates` indicates whether we launch Attack 2 (Def 1 in paper)
* `boosting` indicates whether we launch Attack 3 (Def 1 in paper)
* If both `random_updates` and `boosting` is set to 0, then we default to Attack 1 (Def 1 paper)
* By default, we disable all robust baselines. If you want to test any of them, set `--optimizer=fedavg`, and set any of the robust baselines to 1 (chosen from `gradient_clipping, krum, mkrum, median, k_norm, k_loss, fedmgda` in `run.sh`). For `fedmgda`, one needs to set an additional `fedmgda_eps` hyperparameter, chosen from the continuous range of [0, 1]. For our experiments, we pick the best `fedmgda_eps` among {0, 0.1, 0.5, 1} based on validation performance on benign devices.

#### Some example instructions on Fashion MNIST
* Download datasets (link and instructions under `ditto/data/fmnist/README.md`)
* Fashion MNIST, Ditto, without attacks, lambda=1: `bash run_fashion_clean_ditto_lam1.sh`
* Fashion MNIST, Ditto, A1 (50% adversaries), lambda=1: `bash run_fashion_a1_50_ditto_lam1.sh`

#### Some example instructions on Vehicle
* Download datasets (link and instructions under `ditto/data/vehicle/README.md`)
* Vehicle, Ditto, without attacks, lambda=1: `bash run_vehicle_clean_ditto_lam1.sh`
* Vehicle, Ditto, A1 (50% adversaries), lambda=1: `bash run_vehicle_a1_50_ditto_lam1.sh`

